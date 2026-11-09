"""3D Container Loading – direct structural packing with grid-accelerated search and final polish."""
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

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
    # We use a 3D array of 0/1 with cell size = 1 unit.
    MAX_GRID_DIM = 200  # cap to avoid memory blowup

    def make_grid(W, H, D):
        """Create a 3D occupancy grid if dimensions are small enough."""
        if W <= MAX_GRID_DIM and H <= MAX_GRID_DIM and D <= MAX_GRID_DIM:
            return np.zeros((W, H, D), dtype=bool)
        return None

    def grid_check_and_mark(grid, x, y, z, w, h, d):
        """Check if box fits in grid; if yes, mark and return True."""
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        grid[x:x+w, y:y+h, z:z+d] = True
        return True

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

    def try_place_bruteforce(w, h, d, grid, W, H, D):
        """
        Brute force search for a valid position for a specific box.
        Only used in final polish when standard candidates fail.
        Returns (x,y,z,rw,rh,rd) or None.
        """
        if grid is None:
            return None
            
        rots = get_rotations(w, h, d)
        best = None
        best_sc = float('inf')
        
        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
            
            # Iterate through all possible positions
            # Optimization: Skip if box is too big
            if rw > W or rh > H or rd > D:
                continue
                
            # Use numpy to check all positions? No, that's complex.
            # Simple loop with early termination.
            # To make this fast, we can check the grid slice.
            # If the container is small, this is fine.
            for x in range(W - rw + 1):
                if check_deadline():
                    return None
                for y in range(H - rh + 1):
                    if check_deadline():
                        return None
                    for z in range(D - rd + 1):
                        if not grid[x:x+rw, y:y+rh, z:z+rd].any():
                            sc = score_pos(x, y, z)
                            if sc < best_sc:
                                best_sc = sc
                                best = (x, y, z, rw, rh, rd)
                            # We can break early if we find (0,0,0)
                            if x == 0 and y == 0 and z == 0:
                                return best
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

    def final_polish(result, boxes, W, H, D):
        """
        Attempt to place left-out boxes using brute force or extended search.
        Modifies result and returns updated volume.
        """
        if check_deadline():
            return result
            
        # Reconstruct grid from result
        grid = np.zeros((W, H, D), dtype=bool)
        placed = []
        
        # First pass: mark all placed boxes
        for i in range(len(result)):
            b, x, y, z, w, h, d = result[i]
            if b == 0:
                grid[x:x+w, y:y+h, z:z+d] = True
                placed.append((x, y, z, w, h, d))
                
        # Identify unplaced boxes
        unplaced_indices = [i for i in range(len(result)) if result[i][0] == -1]
        
        if not unplaced_indices:
            return result
            
        # Sort unplaced by volume descending (big ones first might be harder, 
        # but small ones might fit in gaps. Let's try smallest first to fill cracks?)
        # Actually, big boxes are harder to place. If we place small ones first, 
        # we might block big ones. 
        # Standard heuristic: Try to place the "hardest" boxes first? 
        # Or just iterate. Let's try sorting by volume ascending (smallest first) 
        # to see if they fit in the small gaps left by the greedy packing.
        unplaced_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
        
        # Threshold for brute force: if grid size is small enough
        grid_size = W * H * D
        use_bruteforce = grid_size < 500000 # ~0.5M cells, manageable
        
        for idx in unplaced_indices:
            if check_deadline():
                break
                
            w, h, d = boxes[idx]
            
            # 1. Try standard candidate generation first (fast)
            pos = try_place(w, h, d, placed, grid, W, H, D)
            
            # 2. If failed, try brute force if allowed
            if pos is None and use_bruteforce:
                pos = try_place_bruteforce(w, h, d, grid, W, H, D)
                
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                # Update result
                result[idx] = (0, x, y, z, rw, rh, rd)
                # Update grid and placed list
                grid[x:x+rw, y:y+rh, z:z+rd] = True
                placed.append((x, y, z, rw, rh, rd))
                
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
        # Prefer boxes with aspect ratio close to 1
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
        # Fresh grid for each pass
        if grid_template is not None:
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

        if grid_template is not None:
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res_r = pack_order(order_r, boxes, W, H, D, g)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    else:
        # Apply final polish to the best result found
        if grid_template is not None:
            best_result = final_polish(best_result, boxes, W, H, D)
            # Check if polish improved volume (it should, or stay same)
            # Note: final_polish modifies best_result in place

    return best_result
