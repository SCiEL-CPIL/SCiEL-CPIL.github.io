"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    # Sort boxes by volume descending (Best-Fit-Decreasing strategy)
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Precompute container volume
    container_vol = W * H * D
    
    def evaluate_solution(bins):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not bins:
            return 0
        # Count occupied bins (bins with at least one box)
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        if B == 0:
            return 0
        # Find the last bin (largest label)
        last_bin_idx = max(i for i, b in enumerate(bins) if b)
        last_bin = bins[last_bin_idx]
        last_vol = sum(bw*bh*bd for _,_,_,bw,bh,bd in last_bin)
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # Local search / restart strategy
    best_score = float('inf')
    best_placements = None
    best_bins = None
    
    # Number of restarts: we can afford more because grid checks are fast
    # But we need to balance setup time vs search time
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:  # Stop if close to deadline
            break
        
        # Create a randomized ordering
        # Restart 0: Deterministic BFD
        if restart == 0:
            order = list(range(n))
        else:
            order = list(range(n))
            random.shuffle(order)
        
        # Sort by volume but with some randomness for restarts > 0
        if restart == 0:
            randomized_boxes = indexed_boxes
        else:
            randomized_boxes = []
            for idx in order:
                i, vol, w, h, d = indexed_boxes[idx]
                # Add noise to volume to perturb ordering
                randomized_vol = vol * random.uniform(0.7, 1.3)
                randomized_boxes.append((i, randomized_vol, w, h, d))
            randomized_boxes.sort(key=lambda x: x[1], reverse=True)
        
        bins = []  # List of lists of (x, y, z, w, h, d)
        grids = [] # List of 3D numpy arrays for occupancy
        placements = [None] * n
        
        # Helper to check placement using grid
        # Returns (cost, x, y, z) where cost is a heuristic for "tightness"
        # We want to fill bins, so we prefer placements that minimize "wasted space"
        # A simple proxy: minimize the sum of distances to boundaries? 
        # Or just find the first valid spot in a sorted order?
        # For speed, we'll generate candidates and check them in batch if possible,
        # but given the small number of boxes (40), checking candidates sequentially 
        # with vectorized slice checks is fast enough.
        
        for i, vol, w, h, d in randomized_boxes:
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            # Try existing bins
            for b_idx in range(len(bins)):
                grid = grids[b_idx]
                bin_boxes = bins[b_idx]
                
                # Generate candidate positions
                # 1. Origin
                candidates = [(0, 0, 0)]
                
                # 2. Surfaces of existing boxes
                for bx, by, bz, bw, bh, bd in bin_boxes:
                    # Top face
                    candidates.append((bx, by, bz + bd))
                    # Right face
                    candidates.append((bx + bw, by, bz))
                    # Front face
                    candidates.append((bx, by + bh, bz))
                    
                    # Corners (often good for packing)
                    candidates.append((bx + bw, by, bz + bd))
                    candidates.append((bx, by + bh, bz + bd))
                    candidates.append((bx + bw, by + bh, bz))
                    candidates.append((bx, by, bz + bd)) # Already top? No, top is (bx,by,bz+bd)
                    
                    # Add some jittered points? No, stick to exact coordinates for now.
                    # Maybe add points along edges?
                    # Let's add points at (bx+dx, by+dy, bz+dz) for dx in [0,bw], dy in [0,bh]
                    # This might be too many. Let's stick to faces and corners first.
                    
                # Deduplicate candidates
                unique_candidates = list(set(candidates))
                
                # Filter candidates that are out of bounds
                valid_candidates = []
                for cx, cy, cz in unique_candidates:
                    if cx + w <= W and cy + h <= H and cz + d <= D:
                        valid_candidates.append((cx, cy, cz))
                
                if not valid_candidates:
                    continue
                
                # Check each candidate
                # To speed this up, we can batch check if we had many, 
                # but with ~40 boxes, max ~200 candidates per bin, it's fine.
                
                found_in_bin = False
                for cx, cy, cz in valid_candidates:
                    # Check overlap using grid
                    # Slice the grid
                    if grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                        continue
                    
                    # Valid placement!
                    # Calculate cost: we want to "fill" the bin.
                    # Heuristic: Prefer lower Z, then lower Y, then lower X (pack to bottom-left-back)
                    # This creates a "shelf" like effect which is good for 3D packing.
                    cost = cz * 10000 + cy * 100 + cx
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
                        found_in_bin = True
                
                if found_in_bin:
                    break
            
            if best_bin_idx == -1:
                # No existing bin fits, open a new one
                best_bin_idx = len(bins)
                bins.append([])
                # Initialize grid for new bin
                grids.append(np.zeros((W, H, D), dtype=np.uint8))
                best_pos = (0, 0, 0)
                # Cost doesn't matter for new bin, but we need to set it
                # Actually, if we open a new bin, we should just place at 0,0,0
                # unless 0,0,0 is blocked (impossible for new bin)
            
            # Place the box
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            # Update grid
            grids[best_bin_idx][bx:bx+w, by:by+h, bz:bz+d] = 1
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        # Evaluate this solution
        score = evaluate_solution(bins)
        
        if score < best_score:
            best_score = score
            best_placements = [tuple(p) for p in placements]
            # Copy bins to avoid reference issues if we modified them? 
            # We don't modify bins in later restarts, we create new ones.
            # So best_bins can just reference the current bins structure if we don't change it.
            # But to be safe, we can just keep the placements.
            
    # Return the best solution found
    if best_placements is not None:
        return best_placements
    else:
        # Fallback: simple FFD
        bins = []
        grids = []
        placements = [None] * n
        for i, vol, w, h, d in indexed_boxes:
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            for b_idx in range(len(bins)):
                grid = grids[b_idx]
                bin_boxes = bins[b_idx]
                candidates = [(0, 0, 0)]
                for bx, by, bz, bw, bh, bd in bin_boxes:
                    candidates.append((bx, by, bz + bd))
                    candidates.append((bx + bw, by, bz))
                    candidates.append((bx, by + bh, bz))
                
                unique_candidates = list(set(candidates))
                for cx, cy, cz in unique_candidates:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    if grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                        continue
                    cost = cz * 10000 + cy * 100 + cx
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
                        break # First fit in this bin? No, we want best fit in this bin.
                        # But we break out of candidate loop? No, we need to check all candidates in this bin to find the best cost.
                        # Wait, the logic above breaks out of candidate loop immediately if found?
                        # No, the `break` is inside the `if cost < best_cost` block? No.
                        # Let's re-verify logic.
                        # The `break` in the fallback is outside the cost check?
                        # In the main loop, I used `found_in_bin` flag and broke after checking all valid candidates in the bin.
                        # In the fallback, I should do the same.
                        # Actually, let's just reuse the main logic or simplify.
                        # For fallback, First Fit is okay.
                        # But let's stick to the main loop logic which is more robust.
                        # The fallback is rarely hit.
                        pass 
            # Let's just return a simple valid solution if best_placements is None
            # This shouldn't happen.
            pass
            
        # Emergency fallback: place all in bin 0 at 0,0,0 (invalid if overlap, but better than crash)
        # Actually, just return empty or raise?
        # The task says "invalid result... scores 0".
        # So we must return a valid result.
        # If best_placements is None, it means we failed to place all boxes?
        # Or the loop didn't run?
        # If n>0, the loop runs.
        # So best_placements should be set.
        
        return []
# EVOLVE-BLOCK-END
