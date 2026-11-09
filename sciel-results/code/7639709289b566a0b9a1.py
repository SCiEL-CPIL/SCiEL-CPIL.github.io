"""3D container loading: spatial-grid accelerated first-fit with dense candidates and multi-strategy restarts."""
import time
import math
import random
from itertools import product

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    n = len(boxes)
    container_vol = W * H * D

    # Precompute all 6 rotations per box
    def rotations(w, h, d):
        seen = set()
        out = []
        for a, b, c in ((w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)):
            key = (a, b, c)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    # Spatial grid for fast overlap checking
    # Grid cell size: use a reasonable resolution
    def make_grid():
        # Use integer grid with cell size 1 for simplicity, or adaptive
        # For efficiency, track occupied cells
        return {}

    def grid_check_and_mark(grid, x, y, z, w, h, d, mark=True):
        """Check if region (x,y,z,w,h,d) overlaps any marked region. If mark=True, mark it."""
        # Convert to integer coordinates for grid
        xi, yi, zi = int(x), int(y), int(z)
        x2, y2, z2 = int(x + w), int(y + h), int(z + d)
        
        # Check bounds
        if xi < 0 or yi < 0 or zi < 0:
            return False
        if x2 > W or y2 > H or z2 > D:
            return False
        
        # Check against all marked cells in the region
        for cx in range(xi, x2):
            for cy in range(yi, y2):
                for cz in range(zi, z2):
                    key = (cx, cy, cz)
                    if key in grid:
                        return False
                    if mark:
                        grid[key] = True
        return True

    def try_place_all(order, deadline):
        """Try to place all boxes in given order. Returns (placements, filled_volume)."""
        placed = []
        grid = {}
        placements = [None] * n
        filled_vol = 0

        for idx in order:
            if time.time() > deadline:
                break
            w, h, d = boxes[idx]
            rots = rotations(w, h, d)
            best = None
            best_key = None

            for rw, rh, rd in rots:
                # Generate dense candidate positions
                cands = set()
                cands.add((0, 0, 0))
                
                # Add positions adjacent to each placed box
                for px, py, pz, pw, ph, pd in placed:
                    # Right
                    cands.add((px + pw, py, pz))
                    # On top
                    cands.add((px, py + ph, pz))
                    # Front
                    cands.add((px, py, pz + pd))
                    # Diagonals for better coverage
                    cands.add((px + pw, py + ph, pz))
                    cands.add((px + pw, py, pz + pd))
                    cands.add((px, py + ph, pz + pd))
                    cands.add((px + pw, py + ph, pz + pd))
                
                # Add some grid-aligned positions
                # Sample positions along each axis
                step_x = max(1, rw // 3)
                step_y = max(1, rh // 3)
                step_z = max(1, rd // 3)
                
                for px, py, pz, pw, ph, pd in placed:
                    # Positions offset from each face
                    for dx in (0, step_x, pw):
                        for dy in (0, step_y, ph):
                            for dz in (0, step_z, pd):
                                nx, ny, nz = px + dx, py + dy, pz + dz
                                if nx + rw <= W and ny + rh <= H and nz + rd <= D:
                                    cands.add((nx, ny, nz))

                # Sort candidates by preference: lower z, then y, then x
                cands = list(cands)
                cands.sort(key=lambda c: (c[2], c[1], c[0]))

                found = False
                for (cx, cy, cz) in cands:
                    if grid_check_and_mark(grid, cx, cy, cz, rw, rh, rd, mark=False):
                        key = (cz, cy, cx)
                        if best_key is None or key < best_key:
                            best_key = key
                            best = (cx, cy, cz, rw, rh, rd)
                        found = True
                        break

                if found:
                    break

            if best is not None:
                cx, cy, cz, rw, rh, rd = best
                # Mark the grid
                grid_check_and_mark(grid, cx, cy, cz, rw, rh, rd, mark=True)
                placed.append((cx, cy, cz, rw, rh, rd))
                placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                filled_vol += rw * rh * rd
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill any None slots
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

        return placements, filled_vol

    # Define multiple sorting strategies
    def strategy_volume_desc():
        return sorted(range(n), key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    def strategy_max_dim_desc():
        return sorted(range(n), key=lambda i: -max(boxes[i]))

    def strategy_surface_area_desc():
        return sorted(range(n), key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))

    def strategy_volume_asc():
        return sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])

    def strategy_random():
        order = list(range(n))
        random.shuffle(order)
        return order

    # Try multiple strategies and keep the best result
    strategies = [
        strategy_volume_desc,
        strategy_max_dim_desc,
        strategy_surface_area_desc,
    ]

    # Add random strategies if we have time
    best_placements = None
    best_filled = 0

    # First pass: try deterministic strategies
    for strat_func in strategies:
        if time.time() > deadline:
            break
        order = strat_func()
        placements, filled = try_place_all(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placements = placements[:]

    # If we have time, try random restarts
    if time.time() < deadline - 5.0:
        random.seed(42)
        max_random_restarts = 10
        for _ in range(max_random_restarts):
            if time.time() > deadline:
                break
            order = strategy_random()
            placements, filled = try_place_all(order, deadline)
            if filled > best_filled:
                best_filled = filled
                best_placements = placements[:]

    # If no placements found, return all left out
    if best_placements is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return best_placements
# EVOLVE-BLOCK-END
