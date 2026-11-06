"""3D Container Loading - Skyline-based approach with multiple orientations."""
import time
from itertools import product

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 120  # seconds
    
    n = len(boxes)
    
    # Generate all 6 orientations for each box (w, h, d permutations)
    def get_orientations(w, h, d):
        orients = set()
        for perm in product([w, h, d], repeat=3):
            if len(set(perm)) == len(set([w, h, d])) or True:
                pass
        # Actually, just generate all 6 permutations
        orients = []
        for perm in product([w, h, d], repeat=3):
            # Only keep unique permutations to avoid duplicates
            if perm not in orients:
                orients.append(perm)
        return orients
    
    # Skyline approach: maintain a 2D grid of "skyline" points (x, z, height)
    # For each position (x, z) on the floor, track the highest y occupied.
    # This is a simplified 3D skyline - we use a 2D grid where each cell tracks max_y.
    
    # Grid resolution: use unit resolution if container is small, otherwise scale
    # For 60 boxes in reasonable containers, unit resolution should be fine
    # But let's cap grid size to avoid memory issues
    
    # Determine grid dimensions
    max_grid = 500  # cap to avoid huge grids
    
    if W <= max_grid and H <= max_grid and D <= max_grid:
        gw, gh, gd = W, H, D
    else:
        # Scale down
        scale = max_grid / max(W, H, D)
        gw = max(1, int(W * scale))
        gh = max(1, int(H * scale))
        gd = max(1, int(D * scale))
    
    # Use a 2D grid for the floor (x, z) -> max_y occupied
    # This gives us a "skyline" in the y-direction
    # sky[x][z] = highest y that is occupied at that (x,z) column
    sky = [[0] * gd for _ in range(gw)]
    
    placements = [None] * n
    
    # Sort boxes by volume descending (bigger first tends to pack better)
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
    
    placed_count = 0
    filled_volume = 0
    
    for idx in box_indices:
        if time.time() - t0 > DEADLINE:
            break
            
        w, h, d = boxes[idx]
        
        # Try all 6 orientations
        best_pos = None
        best_z = -1  # We want to minimize z (fill from bottom)
        
        orientations = []
        seen = set()
        for ow in [w, h, d]:
            for oh in [w, h, d]:
                for od in [w, h, d]:
                    if (ow, oh, od) != (w, h, d) and (ow, oh, od) in seen:
                        continue
                    # Check it's a valid permutation
                    if sorted([ow, oh, od]) == sorted([w, h, d]):
                        if (ow, oh, od) not in seen:
                            seen.add((ow, oh, od))
                            orientations.append((ow, oh, od))
        
        for ow, oh, od in orientations:
            # Try to find a position (x, y, z) where this orientation fits
            # We scan the 2D floor grid
            found = False
            for x in range(gw - ow + 1):
                if found:
                    break
                for z in range(gd - od + 1):
                    # Check if the box fits at (x, y, z) with orientation (ow, oh, od)
                    # The box occupies x..x+ow, y..y+oh, z..z+od
                    # We need to check that for all (xi, zi) in the box's footprint,
                    # sky[xi][zi] + oh <= H, and also that no already-placed box overlaps
                    
                    # Simplified: check that the max sky value in the footprint + oh <= H
                    max_sky = 0
                    for xi in range(x, x + ow):
                        for zi in range(z, z + od):
                            if sky[xi][zi] > max_sky:
                                max_sky = sky[xi][zi]
                    
                    y = max_sky
                    if y + oh <= H:
                        # This position works
                        if best_pos is None or z < best_z or (z == best_z and y < best_pos[1]):
                            best_pos = (x, y, z, ow, oh, od)
                            best_z = z
                        found = True
                        break
            
            if time.time() - t0 > DEADLINE:
                break
        
        if best_pos is not None:
            x, y, z, ow, oh, od = best_pos
            placements[idx] = (0, x, y, z, ow, oh, od)
            filled_volume += ow * oh * od
            placed_count += 1
            
            # Update skyline
            for xi in range(x, x + ow):
                for zi in range(z, z + od):
                    if y + oh > sky[xi][zi]:
                        sky[xi][zi] = y + oh
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any remaining boxes that weren't processed due to deadline
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
