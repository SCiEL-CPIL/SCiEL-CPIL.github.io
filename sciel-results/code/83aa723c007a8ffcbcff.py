"""3D Container Loading – grid-accelerated packing with batched re-insertion."""
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    def check_deadline():
        return time.time() - start_time > DEADLINE

    def get_rotations(w, h, d):
        """Return unique axis rotations."""
        seen = set()
        rots = []
        for p in itertools.permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots

    # Grid-based occupancy for fast overlap checks
    MAX_GRID_DIM = 200

    def make_grid(W, H, D):
        if W <= MAX_GRID_DIM and H <= MAX_GRID_DIM and D <= MAX_GRID_DIM:
            return np.zeros((W, H, D), dtype=bool)
        return None

    def grid_check_and_mark(grid, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        grid[x:x+w, y:y+h, z:z+d] = True
        return True

    def grid_unmark(grid, x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = False

    def list_check(placed, x, y, z, w, h, d):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def generate_candidates(placed, W, H, D):
        cands = set()
        cands.add((0, 0, 0))

        for (px, py, pz, pw, ph, pd) in placed:
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
            cands.add((0, py, pz))
            cands.add((px, 0, pz))
            cands.add((px, py, 0))

        valid = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    def score_pos(x, y, z):
        return y * 1000 + x * 10 + z

    def try_place(w, h, d, placed, grid, W, H, D):
        rots = get_rotations(w, h, d)
        best = None
        best_sc = float('inf')

        cands = generate_candidates(placed, W, H, D)

        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
            for (x, y, z) in cands:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if grid is not None:
                    if grid[x:x+rw, y:y+rh, z:z+rd].any():
                        continue
                else:
                    if not list_check(placed, x, y, z, rw, rh, rd):
                        continue
                sc = score_pos(x, y, z)
                if sc < best_sc:
                    best_sc = sc
                    best = (x, y, z, rw, rh, rd)
        return best

    def pack_order(order, boxes, W, H, D, grid):
        placed = []
        result = [None] * len(boxes)

        for idx in order:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            pos = try_place(w, h, d, placed, grid, W, H, D)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if grid is not None:
                    grid[x:x+rw, y:y+rh, z:z+rd] = True
                result[idx] = (0, x, y, z, rw, rh, rd)

        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        return result

    def filled_volume(result):
        return sum(w * h * d for (b, x, y, z, w, h, d) in result if b == 0)

    def batched_reinsertion(result, boxes, W, H, D, grid):
        """Attempt to insert left-out boxes using grid checks."""
        if grid is None:
            return result, filled_volume(result)

        placed_list = []
        for (b, x, y, z, w, h, d) in result:
            if b == 0:
                placed_list.append((x, y, z, w, h, d))

        # Collect left-out boxes
        left_out = [i for i in range(len(boxes)) if result[i][0] == -1]

        # Sort left-out by volume descending
        left_out.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

        new_result = [list(t) for t in result]
        improved = False

        for idx in left_out:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            pos = try_place(w, h, d, placed_list, grid, W, H, D)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed_list.append((x, y, z, rw, rh, rd))
                grid[x:x+rw, y:y+rh, z:z+rd] = True
                new_result[idx] = [0, x, y, z, rw, rh, rd]
                improved = True

        if improved:
            return [tuple(t) for t in new_result], filled_volume(new_result)
        return result, filled_volume(result)

    # Sort strategies
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    def sort_vol(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))

    def sort_max(lst):
        return sorted(lst, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_surf(lst):
        return sorted(lst, key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))

    def sort_min(lst):
        return sorted(lst, key=lambda t: (-min(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))

    def sort_base(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))

    def sort_aspect(lst):
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return (dims[2]/dims[0] if dims[0] > 0 else 0)
        return sorted(lst, key=lambda t: (aspect(t), -t[1]*t[2]*t[3]))

    strategies = [sort_vol, sort_max, sort_surf, sort_min, sort_base, sort_aspect]

    grid_template = make_grid(W, H, D)

    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        if grid_template is not None:
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res = pack_order(order, boxes, W, H, D, g)
        # Batched reinsertion
        res, vol = batched_reinsertion(res, boxes, W, H, D, g)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts
    rng = random.Random(123)
    max_restarts = 300

    for _ in range(max_restarts):
        if check_deadline():
            break
        shuffled = list(indexed)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        n = len(shuffled)
        for _ in range(n // 2):
            i = rng.randint(0, n - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]

        if grid_template is not None:
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res_r = pack_order(order_r, boxes, W, H, D, g)
        res_r, vol_r = batched_reinsertion(res_r, boxes, W, H, D, g)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
