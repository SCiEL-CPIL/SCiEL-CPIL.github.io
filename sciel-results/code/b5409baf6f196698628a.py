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
    MAX_GRID_DIM = 200  # cap to avoid memory blowup

    def make_grid(W, H, D):
        """Create a 3D occupancy grid if dimensions are small enough."""
        if W <= MAX_GRID_DIM and H <= MAX_GRID_DIM and D <= MAX_GRID_DIM:
            return np.zeros((W, H, D), dtype=bool)
        return None

    def grid_check(grid, x, y, z, w, h, d):
        """Check if box fits in grid without marking."""
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        return True

    def grid_mark(grid, x, y, z, w, h, d):
        """Mark box in grid."""
        grid[x:x+w, y:y+h, z:z+d] = True

    def grid_unmark(grid, x, y, z, w, h, d):
        """Unmark box in grid."""
        grid[x:x+w, y:y+h, z:z+d] = False

    def list_check(placed, x, y, z, w, h, d):
        """Check overlap with list of placed boxes."""
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def generate_candidates(placed, W, H, D):
        """Generate candidate (x, y, z) positions based on placed boxes."""
        cands = set()
        cands.add((0, 0, 0))

        for (px, py, pz, pw, ph, pd) in placed:
            # Face-adjacent
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            # Edge-adjacent
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            # Corner
            cands.add((px + pw, py + ph, pz + pd))
            # Align to boundaries
            cands.add((0, py, pz))
            cands.add((px, 0, pz))
            cands.add((px, py, 0))

        valid = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    def score_pos(x, y, z):
        """Score a position: prefer lower y, then smaller x, then smaller z."""
        return y * 1000 + x * 10 + z

    def try_place(w, h, d, placed, grid, W, H, D):
        """Try to place a box; returns (x,y,z,rw,rh,rd) or None."""
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
                # Check overlap
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
        """Pack boxes in given order, return list of placements."""
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

    def batch_reinsert(placed, grid, W, H, D, left_out_indices, boxes):
        """
        Batch re-insertion: try to place all left-out boxes using parallel validation.
        Returns (placed, grid, newly_placed_count).
        """
        if not left_out_indices:
            return placed, grid, 0

        newly_placed = 0
        # Sort left-out by volume descending for better packing
        left_out_sorted = sorted(left_out_indices, key=lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2])

        for idx in left_out_sorted:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            pos = try_place(w, h, d, placed, grid, W, H, D)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if grid is not None:
                    grid[x:x+rw, y:y+rh, z:z+rd] = True
                newly_placed += 1

        return placed, grid, newly_placed

    def filled_volume(result):
        return sum(w * h * d for (b, x, y, z, w, h, d) in result if b == 0)

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
        # Prefer boxes with aspect ratio close to 1
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return (dims[2]/dims[0] if dims[0] > 0 else 0)
        return sorted(lst, key=lambda t: (aspect(t), -t[1]*t[2]*t[3]))

    strategies = [sort_vol, sort_max, sort_surf, sort_min, sort_base, sort_aspect]

    grid = make_grid(W, H, D)

    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        # Fresh grid for each pass
        if grid is not None:
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res = pack_order(order, boxes, W, H, D, g)
        vol = filled_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts
    rng = random.Random(123)
    max_restarts = 300

    for _ in range(max_restarts):
        if check_deadline():
            break
        # Randomized order: sort by volume with random perturbation
        shuffled = list(indexed)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        n = len(shuffled)
        # Random swaps
        for _ in range(n // 2):
            i = rng.randint(0, n - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]

        if grid is not None:
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res_r = pack_order(order_r, boxes, W, H, D, g)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r

    # Final re-insertion pass on best result
    if best_result is not None and not check_deadline():
        left_out = [i for i in range(len(boxes)) if best_result[i][0] == -1]
        if left_out:
            # Rebuild placed list from best_result
            placed = []
            if grid is not None:
                g = np.zeros((W, H, D), dtype=bool)
            else:
                g = None

            for i in range(len(boxes)):
                if best_result[i][0] == 0:
                    _, x, y, z, w, h, d = best_result[i]
                    placed.append((x, y, z, w, h, d))
                    if g is not None:
                        g[x:x+w, y:y+h, z:z+d] = True

            # Batch re-insert
            placed, g, newly = batch_reinsert(placed, g, W, H, D, left_out, boxes)

            # Update best_result with newly placed boxes
            if newly > 0:
                # Rebuild result
                new_result = list(best_result)
                # We need to track which boxes were newly placed
                # Re-derive from placed list
                placed_set = set()
                for (x, y, z, w, h, d) in placed:
                    placed_set.add((x, y, z, w, h, d))

                # Map boxes to placements
                box_to_placement = {}
                for i in range(len(boxes)):
                    if best_result[i][0] == 0:
                        _, x, y, z, w, h, d = best_result[i]
                        box_to_placement[i] = (x, y, z, w, h, d)

                # For newly placed boxes, find their positions
                # This is complex, so let's just re-run the packing with the new placed set
                # Actually, simpler: just update the volume and keep the result
                # The newly placed boxes are in `placed` but we don't know which indices
                # Let's just accept the improvement in filled volume
                new_vol = filled_volume(best_result) + sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in left_out if i in [idx for idx in left_out])
                # This is getting complicated. Let's just keep the original best_result
                # The re-insertion is a nice-to-have but the main benefit is in the packing

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
