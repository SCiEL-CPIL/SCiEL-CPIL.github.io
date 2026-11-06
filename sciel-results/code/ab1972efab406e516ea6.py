"""3D Container Loading with 3D occupancy grid and binary search on z-axis."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 115.0  # safety margin before 120s kill
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    box_orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # 3D occupancy grid using a set of occupied unit cells
    # For efficiency, we use a set of (x, y, z) tuples representing occupied unit cubes
    
    def pack_with_grid(ordered_indices):
        """
        Pack boxes using 3D occupancy grid with binary search on z-axis.
        Returns (placements_list, total_volume)
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        # Occupied cells: set of (x, y, z) for each unit cube that is occupied
        # We use a set for O(1) lookup
        occupied = set()
        
        # For each x,y, track the highest z that is occupied (skyline)
        # This allows us to quickly find the lowest available z for a given x,y
        # skyline[x][y] = max z such that (x,y,z) is occupied, or 0 if none
        # We'll use a dict of dicts for sparse storage
        skyline = {}  # (x, y) -> max_z_occupied
        
        def is_cell_occupied(cx, cy, cz):
            return (cx, cy, cz) in occupied
        
        def find_lowest_z(x, y, ow, oh, od, min_z=0):
            """
            Find the lowest z >= min_z such that the box (x, y, z) with dims (ow, oh, od)
            does not overlap with any occupied cells.
            Uses skyline to skip large ranges.
            """
            z = min_z
            while z + od <= D:
                # Check if all cells in the box footprint are free at height z
                free = True
                for dx in range(ow):
                    for dy in range(oh):
                        cx, cy = x + dx, y + dy
                        # Check if any cell from z to z+od-1 is occupied
                        # Use skyline to skip: if skyline[cx,cy] >= z, then there's an occupied cell at or above z
                        key = (cx, cy)
                        if key in skyline:
                            max_occ_z = skyline[key]
                            if max_occ_z >= z:
                                # There is an occupied cell at height >= z
                                # We need to check if it's within [z, z+od)
                                # If max_occ_z < z + od, then there's a conflict
                                # But we need to find the actual conflict
                                # For simplicity, just check cell by cell in the conflicting range
                                # But this could be slow. Let's just mark as not free and move z up
                                free = False
                                break
                        # If skyline doesn't have this key, no occupied cells at all in this column
                    if not free:
                        break
                if free:
                    return z
                # Move z up to just above the highest occupied cell in the footprint
                # Find the max occupied z in the footprint
                max_z_in_footprint = min_z
                for dx in range(ow):
                    for dy in range(oh):
                        key = (x + dx, y + dy)
                        if key in skyline:
                            if skyline[key] >= min_z:
                                max_z_in_footprint = max(max_z_in_footprint, skyline[key])
                # Jump to just above the max occupied z
                if max_z_in_footprint > min_z:
                    z = max_z_in_footprint + 1
                else:
                    z += 1
            return None  # No valid z found
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations
            for ow, oh, od in orients:
                # Try all x, y positions
                for x in range(W - ow + 1):
                    for y in range(H - oh + 1):
                        # Find lowest z using binary search / skyline
                        z = find_lowest_z(x, y, ow, oh, od)
                        if z is None:
                            continue
                        
                        # Score: Prefer lower Z (stacking from bottom), then lower Y, then lower X
                        score = z * 1000000 + y * 1000 + x
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z, ow, oh, od)
            
            if best_pos:
                x, y, z, ow, oh, od = best_pos
                placements[idx] = (0, x, y, z, ow, oh, od)
                total_volume += ow * oh * od
                
                # Mark all cells in the box as occupied
                for dx in range(ow):
                    for dy in range(oh):
                        for dz in range(od):
                            cx, cy, cz = x + dx, y + dy, z + dz
                            occupied.add((cx, cy, cz))
                            key = (cx, cy)
                            if key not in skyline or cz + 1 > skyline[key]:
                                skyline[key] = cz + 1
        
        return placements, total_volume

    # Strategies for sorting
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
        lambda i: boxes[i][1],  # Height
        lambda i: boxes[i][0],  # Width
        lambda i: boxes[i][2],  # Depth
        lambda i: -min(boxes[i][0], boxes[i][1], boxes[i][2]),  # Min dim (ascending)
        lambda i: boxes[i][0] * boxes[i][1],  # Base area
        lambda i: boxes[i][1] * boxes[i][2],  # Side area
        lambda i: boxes[i][0] * boxes[i][2],  # Top area
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    # Run each static sorting strategy
    for sort_key in strategies:
        if time.time() > deadline - 10:
            break
        ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
        p, v = pack_with_grid(ordered)
        if v > best_volume:
            best_placements = p
            best_volume = v

    # If we have time, try some random restarts
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            random.shuffle(indexed_boxes)
            ordered = indexed_boxes[:]
            p, v = pack_with_grid(ordered)
            if v > best_volume:
                best_placements = p
                best_volume = v

    return best_placements
# EVOLVE-BLOCK-END
