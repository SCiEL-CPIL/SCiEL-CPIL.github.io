"""3D Container Loading with 3D occupancy grid and extreme-point placement."""
import time
import itertools
import numpy as np


# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Precompute unique orientations for each box
    box_orients = []
    for w, h, d in boxes:
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        box_orients.append(list(opts))

    # 3D occupancy grid
    # Use numpy array for fast operations
    # Grid dimensions: (D, H, W) - z, y, x
    grid = np.zeros((D, H, W), dtype=np.uint8)

    def can_place(x, y, z, pw, ph, pd):
        """Check if a box of size (pw, ph, pd) can be placed at (x, y, z)."""
        if x + pw > W or y + ph > H or z + pd > D:
            return False
        # Check if all cells in the box region are free
        # grid is indexed [z, y, x]
        region = grid[z:z+pd, y:y+ph, x:x+pw]
        if np.any(region > 0):
            return False
        return True

    def place(x, y, z, pw, ph, pd, box_id):
        """Place a box of size (pw, ph, pd) at (x, y, z) with given box_id."""
        grid[z:z+pd, y:y+ph, x:x+pw] = box_id + 1  # use 1-indexed for non-zero

    def unplace(x, y, z, pw, ph, pd, box_id):
        """Remove a box of size (pw, ph, pd) at (x, y, z)."""
        grid[z:z+pd, y:y+ph, x:x+pw] = 0

    def get_extreme_points():
        """Get all extreme points (potential placement positions)."""
        # Extreme points are positions where a box can be placed such that
        # at least one of its faces touches the container wall or another box.
        # A simpler approach: scan all possible positions, but that's expensive.
        # Instead, we'll use a heuristic: try positions near existing boxes and walls.
        
        candidates = set()
        
        # Add corners and edges
        for x in [0, W]:
            for y in [0, H]:
                for z in [0, D]:
                    candidates.add((x, y, z))
        
        # Add positions adjacent to placed boxes
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    if grid[z, y, x] > 0:
                        box_id = grid[z, y, x] - 1
                        # Find the box dimensions - we need to track this
                        # For now, add neighbors
                        candidates.add((x, y, z))
                        if x + 1 < W:
                            candidates.add((x + 1, y, z))
                        if y + 1 < H:
                            candidates.add((x, y + 1, z))
                        if z + 1 < D:
                            candidates.add((x, y, z + 1))
        
        # Filter candidates that are within bounds
        valid_candidates = []
        for (x, y, z) in candidates:
            if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
                valid_candidates.append((x, y, z))
        
        return valid_candidates

    def find_best_placement(bi, orients):
        """Find the best (x, y, z, orientation) for box bi."""
        best_score = -1
        best_pos = None
        best_orient = None
        
        # Get candidate positions
        candidates = get_extreme_points()
        
        # Also add a few random positions for diversity
        for _ in range(5):
            x = np.random.randint(0, max(1, W))
            y = np.random.randint(0, max(1, H))
            z = np.random.randint(0, max(1, D))
            candidates.append((x, y, z))
        
        for (x, y, z) in candidates:
            for oi, (pw, ph, pd) in enumerate(orients):
                if can_place(x, y, z, pw, ph, pd):
                    # Score: prefer positions that fill gaps (lower z, then y, then x)
                    # and that maximize contact with existing boxes
                    score = -z * 1.0 - y * 0.5 - x * 0.1
                    # Bonus for being adjacent to existing boxes
                    if x > 0 and np.any(grid[z:z+pd, y:y+ph, x-1] > 0):
                        score += 2.0
                    if y > 0 and np.any(grid[z:z+pd, y-1:y+ph, x:x+pw] > 0):
                        score += 2.0
                    if z > 0 and np.any(grid[z-1, y:y+ph, x:x+pw] > 0):
                        score += 2.0
                    if x + pw < W and np.any(grid[z:z+pd, y:y+ph, x+pw] > 0):
                        score += 1.0
                    if y + ph < H and np.any(grid[z:z+pd, y+ph:y+ph, x:x+pw] > 0):
                        score += 1.0
                    if z + pd < D and np.any(grid[z+pd, y:y+ph, x:x+pw] > 0):
                        score += 1.0
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        best_orient = oi
        
        return best_pos, best_orient

    # Multiple sorting strategies
    strategies = []

    # 1. Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(vol_order)

    # 2. Largest dimension first
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(maxdim_order)

    # 3. Volume ascending
    vol_asc = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    strategies.append(vol_asc)

    # 4. Original order
    strategies.append(list(range(n)))

    # 5. Sorted by (max_dim, volume)
    mixed_order = sorted(range(n), key=lambda i: (max(boxes[i]), boxes[i][0]*boxes[i][1]*boxes[i][2]), reverse=True)
    strategies.append(mixed_order)

    # 6. Surface area descending
    area_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append(area_order)

    # 7. Volume descending with smallest-dimension tiebreak
    vol_smallest = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], min(boxes[i])), reverse=True)
    strategies.append(vol_smallest)

    best_vol = 0
    best_placements = None

    for order in strategies:
        if time.time() > deadline:
            break
        
        # Reset grid
        grid[:] = 0
        
        placements = [None] * n
        placed_vol = 0
        
        for bi in order:
            if time.time() > deadline:
                break
            
            orients = box_orients[bi]
            best_pos, best_orient = find_best_placement(bi, orients)
            
            if best_pos is not None:
                x, y, z = best_pos
                pw, ph, pd = orients[best_orient]
                place(x, y, z, pw, ph, pd, bi)
                placements[bi] = (0, x, y, z, pw, ph, pd)
                placed_vol += pw * ph * pd
            else:
                w, h, d = boxes[bi]
                placements[bi] = (-1, 0, 0, 0, w, h, d)
        
        # Fill in any unplaced boxes
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        if placed_vol > best_vol:
            best_vol = placed_vol
            best_placements = placements[:]

    # If nothing was placed, fall back to a simple approach
    if best_placements is None:
        best_placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                best_placements.append((-1, 0, 0, 0, w, h, d))
                continue
            best_placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return best_placements
# EVOLVE-BLOCK-END
