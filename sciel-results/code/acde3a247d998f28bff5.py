import time
import random
import itertools
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute volumes
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    # --- Phase 1: Fast Greedy Construction with Voxel Grid ---
    # We use a coarse voxel grid for fast rejection.
    # Resolution: max(W, H, D) / 50 or so, to keep grid size manageable (e.g., 50x50x50 = 125k voxels)
    # Actually, for exactness in final check, we rely on interval checks.
    # But for the *search* phase, we can use a grid to quickly find empty spots.
    
    # Let's stick to a hybrid: 
    # 1. Generate a few good initial solutions using a fast EPA (Extreme Point Approach) with interval checks.
    # 2. Use a Voxel Grid to speed up the "Move" operations in a local search / Simulated Annealing.
    
    def _rotations(w, h, d):
        seen = set()
        result = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                result.append(perm)
        return result

    def _fits(box_dims, pos, container):
        W, H, D = container
        x, y, z = pos
        w, h, d = box_dims
        return x + w <= W and y + h <= H and z + d <= D

    def _no_overlap(box_dims, pos, placed):
        x, y, z = pos
        w, h, d = box_dims
        for (px, py, pz, pw, ph, pd) in placed:
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def _extreme_points(placed, container):
        W, H, D = container
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            points = [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                (px + pw, py + ph, pz),
                (px, py + ph, pz + pd),
                (px + pw, py, pz + pd),
                (px + pw, py + ph, pz + pd),
            ]
            for (cx, cy, cz) in points:
                if cx <= W and cy <= H and cz <= D:
                    candidates.add((cx, cy, cz))
        return list(candidates)

    def _generate_initial_solution(boxes, container, order_idx):
        W, H, D = container
        placed = []
        placements = [None] * len(boxes)
        
        for orig_idx in order_idx:
            w, h, d = boxes[orig_idx]
            rots = _rotations(w, h, d)
            candidates = _extreme_points(placed, container)
            
            best_pos = None
            best_dim = None
            best_score = None
            
            for (cx, cy, cz) in candidates:
                for (rw, rh, rd) in rots:
                    if not _fits((rw, rh, rd), (cx, cy, cz), container):
                        continue
                    if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                        continue
                    
                    # Score: prefer lower z, then lower y, then lower x
                    score = (cz, cy, cx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_dim = (rw, rh, rd)
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2],
                              best_dim[0], best_dim[1], best_dim[2]))
                placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                       best_dim[0], best_dim[1], best_dim[2])
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        for i in range(len(boxes)):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
        return placements, volume_filled

    # Generate initial solutions
    orders = []
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    rng_init = random.Random(42)
    for _ in range(3):
        ord = list(range(n))
        rng_init.shuffle(ord)
        orders.append(ord)
        
    best_placements = None
    best_volume = 0
    initial_solutions = []
    
    for order in orders:
        if time.time() > deadline:
            break
        placements, vol = _generate_initial_solution(boxes, container, order)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
        initial_solutions.append((placements, vol))
        
    if not initial_solutions:
        # Emergency fallback
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]

    # Sort by volume
    initial_solutions.sort(key=lambda x: x[1], reverse=True)
    
    # --- Phase 2: Voxel-Accelerated Local Search ---
    # Strategy: Use a voxel grid to quickly check if a box can fit at a position.
    # We will run a "shake" algorithm on the best initial solution.
    # 1. Remove a random placed box.
    # 2. Try to re-insert it at a random valid position (found via grid scan or random sampling).
    # 3. If it fits, accept. If not, revert.
    # 4. Also try to insert left-out boxes.
    
    # Setup Voxel Grid
    # To keep memory and speed reasonable, we scale the container to a grid of size G.
    # If container is 100x100x100, G=100 is 1M voxels (manageable).
    # If container is 1000x1000x1000, G=100 is still 1M voxels (coarse).
    # We use the grid for *candidate generation* and *fast rejection*, but final validity
    # is always checked with exact interval arithmetic to ensure no overlap.
    
    # Let's use a grid resolution that allows integer coordinates to map to voxels.
    # Actually, for this specific task, the coordinates are likely integers or small floats.
    # We assume integer coordinates for simplicity in grid mapping, or we can just use
    # the grid to guide random sampling.
    
    # A more robust approach for "Voxel Map" strategy:
    # Maintain a NumPy array `grid` of shape (W, H, D) or scaled down.
    # When a box is placed, mark its voxels as occupied.
    # When checking a new box, check if all its voxels are 0.
    
    # Given the 120s limit and 60 boxes, we can afford a reasonably fine grid.
    # Let's scale the grid so that the largest dimension is 100.
    scale = 100.0 / max(W, H, D)
    GW = max(1, int(W * scale))
    GH = max(1, int(H * scale))
    GD = max(1, int(D * scale))
    
    # To avoid issues with scaling, let's just use the raw dimensions if they are small enough,
    # otherwise scale down.
    if W * H * D < 1000000:
        GW, GH, GD = W, H, D
    else:
        GW, GH, GD = int(W * scale), int(H * scale), int(D * scale)
        
    grid = np.zeros((GW, GH, GD), dtype=np.uint8)
    
    def update_grid(placement, val):
        """Update grid for a placement. placement: (x, y, z, w, h, d)"""
        if val == 0:
            return
        x, y, z, w, h, d = placement
        # Map to grid coordinates
        gx1 = int(x * scale) if W*H*D >= 1000000 else int(x)
        gy1 = int(y * scale) if W*H*D >= 1000000 else int(y)
        gz1 = int(z * scale) if W*H*D >= 1000000 else int(z)
        
        gx2 = int((x + w) * scale) if W*H*D >= 1000000 else int(x + w)
        gy2 = int((y + h) * scale) if W*H*D >= 1000000 else int(y + h)
        gz2 = int((z + d) * scale) if W*H*D >= 1000000 else int(z + d)
        
        # Clip to grid bounds
        gx1 = max(0, min(GW, gx1))
        gy1 = max(0, min(GH, gy1))
        gz1 = max(0, min(GD, gz1))
        gx2 = max(0, min(GW, gx2))
        gy2 = max(0, min(GH, gy2))
        gz2 = max(0, min(GD, gz2))
        
        grid[gx1:gx2, gy1:gy2, gz1:gz2] = val

    def grid_fits(x, y, z, w, h, d):
        """Check if box fits in grid (fast pre-check)."""
        gx1 = int(x * scale) if W*H*D >= 1000000 else int(x)
        gy1 = int(y * scale) if W*H*D >= 1000000 else int(y)
        gz1 = int(z * scale) if W*H*D >= 1000000 else int(z)
        
        gx2 = int((x + w) * scale) if W*H*D >= 1000000 else int(x + w)
        gy2 = int((y + h) * scale) if W*H*D >= 1000000 else int(y + h)
        gz2 = int((z + d) * scale) if W*H*D >= 1000000 else int(z + d)
        
        if gx1 < 0 or gy1 < 0 or gz1 < 0:
            return False
        if gx2 > GW or gy2 > GH or gz2 > GD:
            return False
            
        # Check if any voxel in the subgrid is occupied
        if np.any(grid[gx1:gx2, gy1:gy2, gz1:gz2] > 0):
            return False
        return True

    # Initialize grid with the best initial solution
    current_placements = [p[:] for p in initial_solutions[0][0]]
    current_vol = initial_solutions[0][1]
    
    # Rebuild grid from scratch to be sure
    grid[:] = 0
    for p in current_placements:
        if p[0] == 0:
            update_grid((p[1], p[2], p[3], p[4], p[5], p[6]), 1)
            
    local_best_placements = [p[:] for p in current_placements]
    local_best_vol = current_vol
    
    # Local Search Loop
    # We use a simple hill-climbing / randomized restart approach.
    # Moves:
    # 1. Move a placed box to a new random position (if it fits).
    # 2. Try to place a left-out box at a random position (if it fits).
    
    rng = random.Random(123)
    
    # Pre-compute rotations for all boxes to save time
    all_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    while time.time() < deadline:
        # Randomly choose a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Move a placed box
            placed_indices = [i for i, p in enumerate(current_placements) if p[0] == 0]
            if not placed_indices:
                continue
            i = rng.choice(placed_indices)
            
            # Remove from grid
            old_p = current_placements[i]
            update_grid((old_p[1], old_p[2], old_p[3], old_p[4], old_p[5], old_p[6]), 0)
            
            # Try to find a new position
            w, h, d = boxes[i]
            rots = all_rots[i]
            new_pos = None
            new_dim = None
            
            # Try a few random positions
            for _ in range(20):
                rx = rng.randint(0, max(0, W - 1))
                ry = rng.randint(0, max(0, H - 1))
                rz = rng.randint(0, max(0, D - 1))
                
                for rw, rh, rd in rots:
                    if rx + rw > W or ry + rh > H or rz + rd > D:
                        continue
                    
                    # Fast grid check
                    if grid_fits(rx, ry, rz, rw, rh, rd):
                        # Exact check (just in case grid is coarse or has errors)
                        # Since we update grid exactly, grid_fits should be sufficient for validity
                        # IF the grid resolution is 1:1. If scaled, it's an approximation.
                        # To be safe, we should do exact check if grid says "fits".
                        
                        # For speed, we trust the grid if it's fine-grained enough.
                        # Given the constraints, let's assume the grid is accurate enough for rejection.
                        # If it accepts, we must verify no overlap with existing boxes.
                        
                        # Verification:
                        valid = True
                        for j, p in enumerate(current_placements):
                            if j == i or p[0] != 0:
                                continue
                            px, py, pz = p[1], p[2], p[3]
                            pw, ph, pd = p[4], p[5], p[6]
                            if (rx < px + pw and rx + rw > px and
                                ry < py + ph and ry + rh > py and
                                rz < pz + pd and rz + rd > pz):
                                valid = False
                                break
                                
                        if valid:
                            new_pos = (rx, ry, rz)
                            new_dim = (rw, rh, rd)
                            break
                if new_pos:
                    break
            
            if new_pos:
                # Accept move
                current_placements[i] = (0, new_pos[0], new_pos[1], new_pos[2],
                                        new_dim[0], new_dim[1], new_dim[2])
                update_grid((new_pos[0], new_pos[1], new_pos[2], new_dim[0], new_dim[1], new_dim[2]), 1)
                
                # Volume doesn't change for move, but we might have unblocked space?
                # No, volume is constant for move. But we might allow *other* boxes to be placed later.
                # So we don't update volume here.
                
            else:
                # Revert
                current_placements[i] = old_p
                update_grid((old_p[1], old_p[2], old_p[3], old_p[4], old_p[5], old_p[6]), 1)
                
        else:
            # Try to place a left-out box
            left_out_indices = [i for i, p in enumerate(current_placements) if p[0] == -1]
            if not left_out_indices:
                continue
            i = rng.choice(left_out_indices)
            
            w, h, d = boxes[i]
            rots = all_rots[i]
            
            found = False
            for _ in range(20):
                rx = rng.randint(0, max(0, W - 1))
                ry = rng.randint(0, max(0, H - 1))
                rz = rng.randint(0, max(0, D - 1))
                
                for rw, rh, rd in rots:
                    if rx + rw > W or ry + rh > H or rz + rd > D:
                        continue
                    
                    if grid_fits(rx, ry, rz, rw, rh, rd):
                        # Exact check
                        valid = True
                        for j, p in enumerate(current_placements):
                            if p[0] != 0:
                                continue
                            px, py, pz = p[1], p[2], p[3]
                            pw, ph, pd = p[4], p[5], p[6]
                            if (rx < px + pw and rx + rw > px and
                                ry < py + ph and ry + rh > py and
                                rz < pz + pd and rz + rd > pz):
                                valid = False
                                break
                                
                        if valid:
                            current_placements[i] = (0, rx, ry, rz, rw, rh, rd)
                            update_grid((rx, ry, rz, rw, rh, rd), 1)
                            current_vol += w * h * d
                            
                            if current_vol > local_best_vol:
                                local_best_vol = current_vol
                                local_best_placements = [p[:] for p in current_placements]
                                
                            found = True
                            break
                if found:
                    break
                    
    # Return the best solution found
    if local_best_vol > best_volume:
        return local_best_placements
    else:
        return best_placements
