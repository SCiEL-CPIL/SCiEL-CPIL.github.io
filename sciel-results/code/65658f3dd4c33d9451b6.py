import time
import random
import itertools
import numpy as np

def load(boxes, container):
    start = time.time()
    deadline = start + 118.0
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Pre-compute unique rotations for each box (as sorted tuples to dedupe)
    box_rots = []
    for w, h, d in boxes:
        rots = list(set(itertools.permutations((w, h, d))))
        box_rots.append(rots)
    
    # Determine if we can use a 3D numpy grid (memory safe)
    use_grid = (W * H * D <= 200_000_000) and (W < 300 and H < 300 and D < 300)
    grid = np.zeros((W, H, D), dtype=np.uint8) if use_grid else None
    
    def blb_solve(ordered_indices, use_grid_local):
        """
        Runs the Bottom-Left-Back heuristic.
        """
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        # Candidate points for placement (Bottom-Left-Back preference)
        # We use a list and check for duplicates by keeping it smallish or using a set if needed.
        # For speed, a list of tuples is fine for 60 boxes.
        candidates = [(0, 0, 0)]
        
        if use_grid_local:
             # Reset grid if reusing
             # (Actually, we'll create a fresh grid or clear it before calling.
             # To avoid large allocation overhead in loops, we clear if possible)
             if grid is None:
                 grid = np.zeros((W, H, D), dtype=np.uint8)
             else:
                 grid[:] = 0
        else:
             placed = []
        
        placed_vol = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                return results, placed_vol, False
            
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            # Iterate candidates. Sorting candidates can help find the "best" BLB spot
            # faster, but since we check all, it just affects which is found first
            # if scores are equal. The score function ensures BLB is preferred.
            
            for cx, cy, cz in candidates:
                for rw, rh, rd in rots:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    if use_grid_local:
                        # Check collision in grid
                        if np.sum(grid[cx:cx+rw, cy:cy+rh, cz:cz+rd]) == 0:
                            score = cz * 1000000 + cy * 1000 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                    else:
                        # Fallback: check against placed list
                        valid = True
                        for px, py, pz, pw, ph, pd in placed:
                            if (cx < px + pw and px < cx + rw and
                                cy < py + ph and py < cy + rh and
                                cz < pz + pd and pz < cz + rd):
                                valid = False
                                break
                        if valid:
                            score = cz * 1000000 + cy * 1000 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                placed_vol += bw * bh * bd
                
                if use_grid_local:
                    grid[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                else:
                    placed.append((bx, by, bz, bw, bh, bd))
                
                # Generate new candidates based on the placed box
                # We add points on the boundaries of the placed box
                new_cands = [
                    (bx, by, bz + bd),       # Top
                    (bx + bw, by, bz),       # Right
                    (bx, by + bh, bz),       # Front
                    (bx + bw, by + bh, bz),  # Right-Front
                    (bx + bw, by, bz + bd),  # Right-Top
                    (bx, by + bh, bz + bd),  # Front-Top
                    (bx + bw, by + bh, bz + bd) # Right-Front-Top
                ]
                
                # Only add if they are within bounds to save space in candidates list
                for nc in new_cands:
                    if nc[0] < W and nc[1] < H and nc[2] < D:
                        candidates.append(nc)
                        
        return results, placed_vol, True

    # Strategies for sorting boxes
    # 1. Volume descending
    # 2. Surface area descending
    # 3. Max dimension descending
    # 4. Min dimension ascending (often good for packing small gaps if large ones fail)
    # 5. Ratio (Volume / MaxDim) - "slenderness"
    
    indices = list(range(n))
    
    sort_strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],
        lambda i: max(boxes[i]),
        lambda i: min(boxes[i]),
        lambda i: (boxes[i][0] * boxes[i][1] * boxes[i][2]) / max(1.0, max(boxes[i])),
        lambda i: (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]) / max(1.0, max(boxes[i])),
    ]
    
    best_results = None
    best_score = 0
    used_grids = False
    
    # Phase 1: Deterministic Sorts
    for s_idx, sort_func in enumerate(sort_strategies):
        if time.time() > deadline - 5.0:
            break
            
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        
        # Alternate grid/no-grid? Grid is faster. Use grid if available.
        use_g = use_grid
        
        res, vol, completed = blb_solve(sorted_indices, use_g)
        
        if completed and vol > best_score:
            best_score = vol
            best_results = [tuple(x) for x in res]
            
    # Phase 2: Random Restarts
    # The directive suggests leveraging speedups (grid) to do more random restarts.
    # If we used the grid, we are significantly faster.
    # If we didn't, we are slower, so fewer restarts.
    
    if use_grid:
        max_randoms = 200
    else:
        max_randoms = 30

    for r in range(max_randoms):
        if time.time() > deadline - 2.0:
            break
            
        random.shuffle(indices)
        
        # Random rotation preference? 
        # Sometimes shuffling the rotation order inside the box helps.
        # But blb_solve uses a fixed order from box_rots.
        # We can pass a flag or just rely on the candidate point order.
        
        res, vol, completed = blb_solve(indices, use_grid)
        
        if completed and vol > best_score:
            best_score = vol
            best_results = [tuple(x) for x in res]
            
    # Phase 3: Local Polish / "Last box" optimization
    # If we have a lot of time left, we can try to optimize the placement of the last few boxes
    # or simply run even more randoms.
    # Given the 120s limit and our current speed, 200 randoms with grid should be fast (< 1s each? maybe 100-500ms).
    # Let's just run as many randoms as fit in the remaining time.
    
    while time.time() < deadline - 1.0:
        random.shuffle(indices)
        res, vol, completed = blb_solve(indices, use_grid)
        if completed and vol > best_score:
            best_score = vol
            best_results = [tuple(x) for x in res]

    if best_results is None:
        # Fallback: just return all as left out or try one more simple place
        # Actually, if no strategy worked, it's very odd. 
        # Try a single box placement?
        # But best_score would be 0. 
        # Initialize to empty placement to avoid None.
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results

# Test / Self-check (not part of load signature requirement)
if __name__ == "__main__":
    # Simple test
    boxes = [(2, 2, 2), (1, 1, 1), (3, 1, 1)]
    container = (5, 5, 5)
    res = load(boxes, container)
    for r in res:
        print(r)
