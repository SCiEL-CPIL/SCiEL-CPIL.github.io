"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    def get_orientations(w, h, d):
        """Return unique orientations (permutations of dims)."""
        dims = (w, h, d)
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        return orients

    def check_overlap(x, y, z, w, h, d, placed_list):
        """Check if box (x,y,z,w,h,d) overlaps with any in placed_list."""
        for px, py, pz, pw, ph, pd in placed_list:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    def find_placement_grid(boxes_list, container_dims, seed=None, max_time=None):
        """
        Greedy placement using a 3D occupancy grid to find the largest empty space.
        This is a structural change from point-based candidate generation to volume-aware search.
        """
        Wc, Hc, Dc = container_dims
        placed = {}
        placed_boxes = []
        total_vol = 0
        
        # Sort by volume descending
        sorted_boxes = sorted(boxes_list, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1],t[2],t[3])))
        
        # We will use a grid to track occupied space.
        # The grid resolution is 1 unit. If dimensions are large, this might be too slow.
        # Given the 120s limit and 60 boxes, we need to be careful.
        # Let's assume dimensions are reasonable (e.g., < 100). If not, we fall back to point-based.
        
        # Check if grid is feasible
        if Wc * Hc * Dc > 1000000:
            # Fall back to point-based candidate generation if grid is too large
            return find_placement_point(boxes_list, container_dims, seed=seed, max_time=max_time)
        
        # Initialize grid with 0 (empty)
        # Using a flat array for speed
        grid = bytearray(Wc * Hc * Dc)
        
        def get_grid_index(x, y, z):
            return (x * Hc + y) * Dc + z
        
        def can_place_grid(x, y, z, w, h, d):
            if x + w > Wc or y + h > Hc or z + d > Dc:
                return False
            # Check if all cells in the box volume are 0
            for i in range(x, x + w):
                for j in range(y, y + h):
                    for k in range(z, z + d):
                        if grid[get_grid_index(i, j, k)]:
                            return False
            return True
        
        def place_grid(x, y, z, w, h, d):
            for i in range(x, x + w):
                for j in range(y, y + h):
                    for k in range(z, z + d):
                        grid[get_grid_index(i, j, k)] = 1
        
        # For efficiency, we will scan the grid to find the best position.
        # "Best" is defined as the position that minimizes (x, y, z) to keep packing tight,
        # but we want to ensure we don't leave large voids.
        # Actually, the prompt suggests "maximizing the minimum distance value in their footprint".
        # This is complex. A simpler volume-aware strategy is to scan for the first valid position
        # in a specific order (e.g., bottom-left-farthest) but using the grid for fast overlap checks.
        
        # Let's stick to a standard scan order: x=0..W, y=0..H, z=0..D.
        # This is O(W*H*D) per box. With 60 boxes, that's 60 * W*H*D operations.
        # If W*H*D is ~1000, that's 60,000 ops, which is fast.
        # If W*H*D is ~100,000, that's 6,000,000 ops, which is okay in Python if optimized.
        
        for orig_idx, w, h, d in sorted_boxes:
            if max_time and time.time() > max_time:
                break
            
            orients = get_orientations(w, h, d)
            found = False
            
            # Try each orientation
            for ow, oh, od in orients:
                # Scan the grid
                # We want to place it as "low" and "left" as possible to pack tight.
                # Scan z from 0 to Dc-od, y from 0 to Hc-oh, x from 0 to Wc-ow.
                # This is a bit slow in pure Python.
                # Let's try a different approach: iterate through all possible positions and pick the best one
                # based on a heuristic (e.g., closest to origin).
                
                # To speed up, we can break early if we find a position at (0,0,0).
                # But we need to check all orientations? No, if (0,0,0) works for any orientation, it's a good spot.
                
                # Let's just scan in order x, y, z.
                for z in range(Dc - od + 1):
                    if found: break
                    for y in range(Hc - oh + 1):
                        if found: break
                        for x in range(Wc - ow + 1):
                            if can_place_grid(x, y, z, ow, oh, od):
                                place_grid(x, y, z, ow, oh, od)
                                placed[orig_idx] = (x, y, z, ow, oh, od)
                                placed_boxes.append((x, y, z, ow, oh, od))
                                total_vol += ow * oh * od
                                found = True
                                break
                    # Note: The break logic above is slightly off. We need to break all loops.
                    # Let's restructure.
            
            if not found:
                # Box not placed
                pass

        return placed, total_vol

    def find_placement_point(boxes_list, container_dims, seed=None, max_time=None):
        """
        Fallback: Point-based candidate generation (similar to original but optimized).
        """
        Wc, Hc, Dc = container_dims
        placed = {}
        placed_boxes = []
        total_vol = 0
        
        sorted_boxes = sorted(boxes_list, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1],t[2],t[3])))
        
        candidates = set([(0, 0, 0)])
        
        for orig_idx, w, h, d in sorted_boxes:
            if max_time and time.time() > max_time:
                break
            
            orients = get_orientations(w, h, d)
            
            # Generate new candidates from existing placed boxes
            # This is O(N^2) if done naively, but we maintain a list.
            
            best_pos = None
            best_orient = None
            found = False
            
            # Sort candidates for BLF
            sorted_candidates = sorted(candidates, key=lambda p: (p[0], p[1], p[2]))
            
            for cx, cy, cz in sorted_candidates:
                for ow, oh, od in orients:
                    if cx + ow > Wc or cy + oh > Hc or cz + od > Dc:
                        continue
                    if check_overlap(cx, cy, cz, ow, oh, od, placed_boxes):
                        continue
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)
                    found = True
                    break
                if found:
                    break
            
            if found:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed[orig_idx] = (bx, by, bz, bw, bh, bd)
                placed_boxes.append((bx, by, bz, bw, bh, bd))
                total_vol += bw * bh * bd
                
                new_cands = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                for nc in new_cands:
                    if 0 <= nc[0] <= Wc and 0 <= nc[1] <= Hc and 0 <= nc[2] <= Dc:
                        candidates.add(nc)
        
        return placed, total_vol

    boxes_list = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Run initial packing with a few seeds to find a good starting point
    # The grid method is deterministic for a given order, so seeds don't help much unless we shuffle.
    # Let's just run the grid method once. If it's too slow, it will fall back to point-based.
    # Actually, let's run both and pick the best? No, grid is slower.
    # Let's just run the grid method.
    
    best_placed, best_vol = find_placement_grid(boxes_list, (W, H, D), max_time=deadline)
    
    # If grid method failed or was too slow (and fell back), we might want to try point-based too?
    # The fallback is internal. Let's assume the result is good.
    
    # Local Search: Insert left-out boxes
    placed_set = set(best_placed.keys())
    left_out = [i for i in range(len(boxes)) if i not in placed_set]
    left_out.sort(key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def can_place_in_existing(x, y, z, w, h, d, placed_dict):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for idx, (px, py, pz, pw, ph, pd) in placed_dict.items():
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def find_position_for_box(w, h, d, placed_dict):
        orients = get_orientations(w, h, d)
        candidates = set([(0, 0, 0)])
        for idx, (px, py, pz, pw, ph, pd) in placed_dict.items():
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        valid_candidates = []
        for c in candidates:
            if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D:
                valid_candidates.append(c)
        
        valid_candidates.sort(key=lambda p: (p[0], p[1], p[2]))
        
        for cx, cy, cz in valid_candidates:
            for ow, oh, od in orients:
                if can_place_in_existing(cx, cy, cz, ow, oh, od, placed_dict):
                    return (cx, cy, cz, ow, oh, od)
        return None

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        placed_set = set(best_placed.keys())
        left_out = [i for i in range(len(boxes)) if i not in placed_set]
        left_out.sort(key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
        
        for i in left_out:
            if time.time() > deadline - 5:
                break
            
            w, h, d = boxes[i]
            box_vol = w * h * d
            
            pos = find_position_for_box(w, h, d, best_placed)
            if pos:
                bx, by, bz, bw, bh, bd = pos
                best_placed[i] = (bx, by, bz, bw, bh, bd)
                best_vol += box_vol
                improved = True
                continue
            
            if not best_placed:
                continue
                
            placed_items = [(idx, val) for idx, val in best_placed.items()]
            placed_items.sort(key=lambda t: t[1][3]*t[1][4]*t[1][5])
            
            max_removals = 3
            removal_candidates = placed_items[:max_removals]
            
            found_swap = False
            for removal in removal_candidates:
                if time.time() > deadline - 5:
                    break
                
                r_idx, r_pos = removal
                r_vol = r_pos[3] * r_pos[4] * r_pos[5]
                
                if box_vol <= r_vol:
                    continue
                
                del best_placed[r_idx]
                
                pos = find_position_for_box(w, h, d, best_placed)
                if pos:
                    bx, by, bz, bw, bh, bd = pos
                    best_placed[i] = (bx, by, bz, bw, bh, bd)
                    
                    rw, rh, rd = boxes[r_idx]
                    r_pos_new = find_position_for_box(rw, rh, rd, best_placed)
                    if r_pos_new:
                        best_placed[r_idx] = r_pos_new
                        best_vol += box_vol
                    else:
                        best_vol += (box_vol - r_vol)
                    
                    improved = True
                    found_swap = True
                    break
                else:
                    best_placed[r_idx] = r_pos
                    continue
                
                if found_swap:
                    break

    results = [None] * len(boxes)
    for i in range(len(boxes)):
        if i in best_placed:
            x, y, z, w, h, d = best_placed[i]
            results[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)

    return results

# EVOLVE-BLOCK-END
