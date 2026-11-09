import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

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

    # Pre-compute rotations for all boxes
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]

    def generate_candidates(placed, W, H, D):
        """
        Generate candidate (x, y, z) positions based on placed boxes.
        placed is a list of (x, y, z, w, h, d).
        """
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
            # Also try aligning to the "other" side of the container? 
            # Sometimes it helps to pack against the back/right/top.
            # But BLB prefers 0. Let's stick to BLB primarily.
            
        valid = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid

    def try_place(w, h, d, rots, placed, grid, W, H, D):
        """
        Try to place a box. Returns (x, y, z, rw, rh, rd) or None.
        Uses Bottom-Left-Back heuristic: minimize y, then x, then z.
        """
        best = None
        best_sc = (float('inf'), float('inf'), float('inf'))

        cands = generate_candidates(placed, W, H, D)

        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
            for (x, y, z) in cands:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                
                # Check overlap using grid
                # If the region in the grid is all False, it's empty
                if grid[x:x+rw, y:y+rh, z:z+rd].any():
                    continue
                
                # Heuristic: Bottom (y), Left (x), Back (z)
                sc = (y, x, z)
                if sc < best_sc:
                    best_sc = sc
                    best = (x, y, z, rw, rh, rd)
        
        return best

    def pack_order(order, boxes, W, H, D):
        """Pack boxes in given order, return list of placements."""
        placed = []
        result = [None] * len(boxes)
        # Initialize grid
        grid = np.zeros((W, H, D), dtype=bool)

        for idx in order:
            if check_deadline():
                break
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            pos = try_place(w, h, d, rots, placed, grid, W, H, D)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                grid[x:x+rw, y:y+rh, z:z+rd] = True
                result[idx] = (0, x, y, z, rw, rh, rd)

        for i in range(len(boxes)):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        return result

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
        # Prefer boxes with aspect ratio close to 1 (cubes pack well)
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return (dims[2]/dims[0] if dims[0] > 0 else 0)
        return sorted(lst, key=lambda t: (aspect(t), -t[1]*t[2]*t[3]))
    
    def sort_fit(lst):
        # Prefer boxes that fit well into the container dimensions
        def fit_score(t):
            w, h, d = t[1], t[2], t[3]
            # Penalize if any dimension is much larger than container? 
            # Or prefer dimensions that are close to container dims?
            # Simple heuristic: prefer larger volume, but penalize "thin" boxes
            return (-w*h*d, min(w, h, d))
        return sorted(lst, key=fit_score)

    strategies = [sort_vol, sort_max, sort_surf, sort_min, sort_base, sort_aspect, sort_fit]

    best_result = None
    best_vol = 0

    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        res = pack_order(order, boxes, W, H, D)
        vol = filled_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res

    # Random restarts
    rng = random.Random(123)
    max_restarts = 500 # Increased because grid checks are fast

    for _ in range(max_restarts):
        if check_deadline():
            break
        
        # Randomized order: sort by volume with random perturbation
        shuffled = list(indexed)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        n = len(shuffled)
        
        # Apply random swaps to perturb the order
        # More swaps = more random
        num_swaps = n // 2
        for _ in range(num_swaps):
            i = rng.randint(0, n - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        
        order_r = [t[0] for t in shuffled]
        res_r = pack_order(order_r, boxes, W, H, D)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
