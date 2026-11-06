"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 115.0  # 5s safety margin before 120s limit
    n = len(boxes)
    
    # Initialize placements with all boxes left out
    # Format: (bin, x, y, z, w, h, d)
    # bin=0 for loaded, bin=-1 for left out
    initial_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_placements = initial_placements
    best_volume = 0
    
    def get_orientations(w, h, d):
        """Return unique orientations of a box (w, h, d)."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def skyline_pack(ordered_indices):
        """
        Pack boxes using a skyline (floor-map) approach.
        We maintain a 2D grid grid[x][z] = minimum y-coordinate available at that (x,z) position.
        Initially all zeros. When we place a box at (x, y, z) with dims (ow, oh, od),
        we update grid[x:x+ow, z:z+od] = max(current, y+oh).
        To find a valid position for a new box with dims (ow, oh, od), we look for a position
        (x, z) such that the max of grid[x:x+ow, z:z+od] + oh <= H.
        We prefer positions with lowest y, then lowest x, then lowest z.
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        # Grid dimensions: W x D
        # grid[x][z] = min y available
        grid = np.zeros((W, D), dtype=np.int64)
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            best_y = None
            best_x = None
            best_z = None
            best_pos = None
            
            for ow, oh, od in orients:
                if ow > W or od > D:
                    continue
                
                # Find the best position for this orientation
                # We scan possible x and z positions
                # For efficiency, we can scan x and z, and compute the required y
                for x in range(W - ow + 1):
                    for z in range(D - od + 1):
                        # Get the max height in the region [x, x+ow) x [z, z+od)
                        region_max = np.max(grid[x:x+ow, z:z+od])
                        y_start = region_max
                        if y_start + oh <= H:
                            # Check if this is better
                            if best_y is None or y_start < best_y or (y_start == best_y and x < best_x) or (y_start == best_y and x == best_x and z < best_z):
                                best_y = y_start
                                best_x = x
                                best_z = z
                                best_pos = (ow, oh, od)
            
            if best_pos:
                ow, oh, od = best_pos
                placements[idx] = (0, best_x, best_y, best_z, ow, oh, od)
                total_volume += ow * oh * od
                # Update grid
                grid[best_x:best_x+ow, best_z:best_z+od] = np.maximum(grid[best_x:best_x+ow, best_z:best_z+od], best_y + oh)
        
        return placements, total_volume
    
    # Generate different sorting strategies
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], # Sum of dims
        lambda i: boxes[i][1], # Height
        lambda i: boxes[i][0], # Width
        lambda i: boxes[i][2], # Depth
    ]
    
    # Run strategies
    for key_func in strategies:
        if time.time() > deadline - 10:
            break
        
        ordered = sorted(indexed_boxes, key=key_func, reverse=True)
        p, v = skyline_pack(ordered)
        
        if v > best_volume:
            best_volume = v
            best_placements = p
    
    return best_placements
# EVOLVE-BLOCK-END
