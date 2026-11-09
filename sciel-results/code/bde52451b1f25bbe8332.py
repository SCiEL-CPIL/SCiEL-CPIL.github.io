"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit

    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)

    # Generate all 6 unique orientations for a box
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique

    # Pre-compute orientations
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))

    # Pre-compute box volumes
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Pre-compute max dimension
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    
    # Pre-compute surface area
    box_surface = [2*(w*h + h*d + w*d) for w, h, d in boxes]

    def run_pack(order, grid=None):
        """
        Pack boxes in the given order.
        If grid is provided, it is a copy of the current occupancy grid (for re-insertion).
        Returns: (results_list, filled_volume, final_grid)
        """
        if grid is None:
            grid = np.zeros((W, H, D), dtype=np.uint8)
        else:
            grid = grid.copy() # Work on a copy
            
        placed_boxes = [] # List of (x, y, z, w, h, d)
        results = [None] * n
        filled_vol = 0

        # We need to know which boxes are already in the grid if we start with one
        # For the initial run, placed_boxes is empty.
        # For re-insertion, we should probably rebuild placed_boxes from grid? 
        # Or just use the grid for collision and a list of "occupied corners" for candidate generation.
        # Rebuilding corners from grid is hard. 
        # So, for re-insertion, we will only use the grid for collision, 
        # and for candidates, we will use the corners of the boxes that are ALREADY placed in this specific run_pack call?
        # No, that's wrong. If we re-insert, the "placed" set is the union of original + new.
        # To keep it simple and fast, we will maintain `placed_boxes` list.
        # If `grid` is passed, we assume `placed_boxes` should correspond to it?
        # Let's change signature: run_pack(order, initial_placed_boxes=None, initial_grid=None)
        
        # Actually, simpler: Just pass the list of already placed boxes and the grid.
        # If initial_placed_boxes is None, it's empty.
        
        # Let's refactor to accept initial state.
        if initial_placed_boxes is None:
            initial_placed_boxes = []
            initial_grid = np.zeros((W, H, D), dtype=np.uint8)
        else:
            initial_placed_boxes = list(initial_placed_boxes)
            initial_grid = initial_grid.copy()
            
        placed_boxes = initial_placed_boxes
        grid = initial_grid
        
        # Calculate current filled volume
        filled_vol = sum(w*h*d for _,_,_,w,h,d in placed_boxes)

        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]

            best_pos = None
            best_orientation = None
            best_score = float('inf')

            # Generate candidates based on current placed boxes
            candidates = set()
            candidates.add((0, 0, 0))
            
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # 8 corners of the placed box
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
                # Also add points slightly offset? No, corners are sufficient for BLB.
            
            # Convert to list for iteration
            candidate_list = list(candidates)

            for (ow, oh, od) in orientations:
                for (cx, cy, cz) in candidate_list:
                    if cx < 0 or cy < 0 or cz < 0:
                        continue
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check collision using NumPy
                    # grid is uint8, so sum is fast
                    if grid[cx:cx+ow, cy:cy+oh, cz:cz+od].sum() == 0:
                        # Score: prefer lower z, then lower y, then lower x
                        score = cz * 10000 + cy * 100 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orientation = (ow, oh, od)

            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                # Update grid
                grid[cx:cx+ow, cy:cy+oh, cz:cz+od] = 1
                # Update placed list
                placed_boxes.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
                filled_vol += ow * oh * od
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)

        return results, filled_vol, placed_boxes, grid

    # Initialize run_pack with default args for the wrapper
    def run_pack_wrapper(order, initial_placed=None, initial_grid=None):
        return run_pack(order, initial_placed, initial_grid)

    # Define the actual run_pack with default args
    # The previous def run_pack didn't have defaults. Let's fix that.
    
    # Redefining run_pack properly
    def run_pack(order, initial_placed=None, initial_grid=None):
        if initial_placed is None:
            placed_boxes = []
            grid = np.zeros((W, H, D), dtype=np.uint8)
        else:
            placed_boxes = list(initial_placed)
            grid = initial_grid.copy()
            
        results = [None] * n
        filled_vol = sum(w*h*d for _,_,_,w,h,d in placed_boxes)

        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]

            best_pos = None
            best_orientation = None
            best_score = float('inf')

            candidates = set()
            candidates.add((0, 0, 0))
            
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            candidate_list = list(candidates)

            for (ow, oh, od) in orientations:
                for (cx, cy, cz) in candidate_list:
                    if cx < 0 or cy < 0 or cz < 0:
                        continue
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    if grid[cx:cx+ow, cy:cy+oh, cz:cz+od].sum() == 0:
                        score = cz * 10000 + cy * 100 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orientation = (ow, oh, od)

            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                grid[cx:cx+ow, cy:cy+oh, cz:cz+od] = 1
                placed_boxes.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
                filled_vol += ow * oh * od
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)

        return results, filled_vol, placed_boxes, grid

    best_results = None
    best_volume = -1
    best_placed = []
    best_grid = None

    # Strategy 1: Volume Descending
    order1 = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    res1, vol1, placed1, grid1 = run_pack(order1)
    if vol1 > best_volume:
        best_volume = vol1
        best_results = res1
        best_placed = placed1
        best_grid = grid1

    if time.time() - start_time > DEADLINE - 10:
        return best_results

    # Strategy 2: Max Dim Descending
    order2 = sorted(range(n), key=lambda i: box_max_dim[i], reverse=True)
    res2, vol2, placed2, grid2 = run_pack(order2)
    if vol2 > best_volume:
        best_volume = vol2
        best_results = res2
        best_placed = placed2
        best_grid = grid2

    if time.time() - start_time > DEADLINE - 10:
        return best_results

    # Strategy 3: Surface Area Descending
    order3 = sorted(range(n), key=lambda i: box_surface[i], reverse=True)
    res3, vol3, placed3, grid3 = run_pack(order3)
    if vol3 > best_volume:
        best_volume = vol3
        best_results = res3
        best_placed = placed3
        best_grid = grid3

    if time.time() - start_time > DEADLINE - 10:
        return best_results

    # Strategy 4: Random Orders
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(n))
        random.shuffle(order)
        res, vol, placed, grid = run_pack(order)
        if vol > best_volume:
            best_volume = vol
            best_results = res
            best_placed = placed
            best_grid = grid

    # Phase 2: Re-insertion of left-out boxes
    # Sort left-out boxes by volume ascending (smaller ones first to fit in gaps)
    if best_results is not None and time.time() - start_time < DEADLINE - 5:
        left_out_indices = [i for i in range(n) if best_results[i][0] == -1]
        # Sort by volume ascending
        left_out_indices.sort(key=lambda i: box_volumes[i])
        
        current_placed = list(best_placed)
        current_grid = best_grid.copy()
        current_results = list(best_results)
        
        for idx in left_out_indices:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Try to place this box
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            candidates = set()
            candidates.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd) in current_placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            candidate_list = list(candidates)

            for (ow, oh, od) in orientations:
                for (cx, cy, cz) in candidate_list:
                    if cx < 0 or cy < 0 or cz < 0:
                        continue
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    if current_grid[cx:cx+ow, cy:cy+oh, cz:cz+od].sum() == 0:
                        score = cz * 10000 + cy * 100 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orientation = (ow, oh, od)

            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                current_grid[cx:cx+ow, cy:cy+oh, cz:cz+od] = 1
                current_placed.append((cx, cy, cz, ow, oh, od))
                current_results[idx] = (0, cx, cy, cz, ow, oh, od)
                best_volume += ow * oh * od
                best_results = current_results
                best_placed = current_placed
                best_grid = current_grid

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
# EVOLVE-BLOCK-END
