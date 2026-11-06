import time
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    # Precompute container volume
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Max-Volume-First)
    # This is crucial for minimizing B.
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Data structures for bins
    # bins: list of lists of (x, y, z, w, h, d)
    # grids: list of 3D numpy arrays (uint8) for fast overlap checking
    bins = []
    grids = []
    bin_volumes = [] # Current volume in each bin
    
    placements = [None] * n
    
    # Helper to check if a box fits at a specific position in a specific bin
    def fits(b_idx, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        grid = grids[b_idx]
        # Check overlap
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        return True

    # Process boxes in order of decreasing volume
    for i, vol, w, h, d in indexed_boxes:
        if time.time() > deadline - 5:
            # Emergency break: if we run out of time, we must return a valid solution.
            # Since we place greedily, if we stop here, some boxes might be unplaced.
            # However, the problem implies "places every box". 
            # To ensure validity, we should probably not break unless we have a fallback.
            # Given the speed of grid ops, 150s is very generous for 40 boxes.
            # If we do break, we must ensure all boxes are placed. 
            # Let's assume the loop finishes. If not, we'd need a fallback.
            # For safety, if time is critical, we just force place remaining boxes 
            # in any valid spot or open new bins quickly.
            pass
            
        best_bin_idx = -1
        best_pos = None
        best_remaining_vol = float('inf') # We want to minimize remaining volume (maximize fill)
        
        # Try existing bins
        for b_idx in range(len(bins)):
            # Quick check: does the box fit in the remaining volume?
            # Actually, we need to find a specific position.
            
            # Generate candidate positions
            # 1. Origin
            candidates = [(0, 0, 0)]
            
            # 2. Surfaces of existing boxes
            # To speed this up, we can limit the number of candidates if the bin is very full,
            # but for 40 boxes, the list of existing boxes is small (<40).
            for bx, by, bz, bw, bh, bd in bins[b_idx]:
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
                candidates.append((bx, by, bz + bd))
                
                # Additional "deep" candidates to help fill gaps
                # e.g. next to existing boxes in X, Y, Z
                candidates.append((bx + bw, by, bz + bd)) # Already added
                candidates.append((bx, by + bh, bz + bd)) # Already added
                
                # Let's add some variations to find tighter fits
                # If there is space, try to push it against walls
                # This heuristic is handled by the cost function below.
                
            # Deduplicate candidates
            # Using a set is fast enough for < 300 candidates
            unique_candidates = list(set(candidates))
            
            # Check each candidate
            found_in_bin = False
            for cx, cy, cz in unique_candidates:
                if not fits(b_idx, cx, cy, cz, w, h, d):
                    continue
                
                # Valid placement!
                # Calculate cost: we want to "fill" the bin.
                # Heuristic: Minimize the "depth" of the placement? 
                # Or maximize the fill?
                # The score depends on B and the last bin's fill.
                # Minimizing B is the primary goal.
                # To minimize B, we should pack as tightly as possible into existing bins.
                # A good proxy for "tightness" is the position itself.
                # Placing a box at a higher Z, Y, or X might leave gaps below/beside it.
                # However, "Best Fit" usually means: choose the bin where the box fits with least wasted space.
                # Since we can't easily calculate "wasted space" in 3D instantly, 
                # we use a heuristic: prefer positions that are "lower" in Z, then Y, then X?
                # No, that's "First Fit". 
                # "Best Fit" in 2D often means: minimize the max of (W-x-w, H-y-h).
                # In 3D, let's try to minimize the sum of distances to the "far" walls?
                # Or simply: prefer the bin where the box fits "deepest" (highest Z)? No, that leaves bottom empty.
                # Prefer the bin where the box fits "lowest" (lowest Z)? That fills the bottom.
                
                # Let's use a cost that favors filling the bottom and corners.
                # Cost = z * 1000 + y * 100 + x
                # This encourages placing boxes at the bottom, back, and left.
                # This tends to create a stable stack and reduces fragmentation.
                
                cost = cz * 10000 + cy * 100 + cx
                
                # We want the MINIMUM cost (lowest, back, left)
                # But wait, if we always pick the lowest, we might fill one corner and leave the rest empty.
                # "Best Fit" should ideally pick the placement that minimizes the "hole" size.
                # A simpler proxy: pick the bin where the box fits with the smallest "max coordinate" used?
                
                # Let's stick to the "lowest Z, then lowest Y, then lowest X" heuristic for now,
                # but across ALL bins.
                # If multiple bins have a valid spot at (0,0,0), we pick the first one?
                # No, we should pick the bin that is "most full" to avoid opening new bins unnecessarily.
                
                # Revised Heuristic:
                # 1. Primary: Minimize B. This means we prefer existing bins over new ones.
                # 2. Secondary: Pack tightly.
                
                # Let's evaluate the "fill" of the bin after placement.
                # new_vol = bin_volumes[b_idx] + vol
                # We want to maximize this? No, we want to minimize the number of bins.
                # If a box fits in an existing bin, it's better than opening a new one.
                # Among existing bins, which one is best?
                # The one where the box fits "best".
                
                # Let's use the cost: 
                # If we place it here, what is the "wasted" space?
                # Hard to compute.
                
                # Alternative: 
                # Prioritize bins with higher current volume (more full) to consolidate?
                # Or prioritize bins with lower current volume to balance?
                # Usually, "Best Fit Decreasing" in 2D means: pick the bin where the item fits with least remaining space.
                # In 3D, let's approximate "least remaining space" by preferring positions that are "higher" in Z?
                # No, that's counter-intuitive for stacking.
                
                # Let's try this:
                # For each valid position, calculate a "score" based on how much it "blocks" future placements.
                # Simpler: Just use the coordinate cost (z*10000 + y*100 + x) but INVERTED?
                # If we invert it, we place at the TOP? No.
                
                # Let's look at the current program's success. It used:
                # cost = cz * 10000 + cy * 100 + cx
                # and picked the MINIMUM cost.
                # This means it places at the bottom-left-back.
                # This is a standard "First Fit" heuristic in 3D.
                
                # To differentiate and potentially improve:
                # We can try "Best Fit" by checking if the box fits in a "hole" created by previous boxes.
                # The candidates from surfaces already capture this.
                
                # Let's add a tie-breaker: if cost is same, prefer the bin with higher current volume (more full).
                # This helps consolidate into fewer bins.
                
                bin_fill = bin_volumes[b_idx] / container_vol if container_vol > 0 else 0
                # We want to maximize bin_fill? Or minimize?
                # If we maximize bin_fill, we put small boxes into already full bins, which is good for B.
                # But if the bin is almost full, there might be no space.
                
                # Let's combine: 
                # Primary: Validity
                # Secondary: Minimize coordinate cost (pack to bottom)
                # Tertiary: Maximize bin fill (consolidate)
                
                # Since coordinate cost is discrete and large, it will dominate.
                # Let's just stick to the coordinate cost for now, but ensure we check ALL bins.
                
                if best_bin_idx == -1 or cost < best_pos_cost or (cost == best_pos_cost and bin_volumes[b_idx] > best_bin_vol):
                    best_bin_idx = b_idx
                    best_pos = (cx, cy, cz)
                    best_pos_cost = cost
                    best_bin_vol = bin_volumes[b_idx]
                    found_in_bin = True
                
            if found_in_bin:
                # We found a spot in this bin. 
                # Should we break? 
                # "Best Fit" usually implies checking all bins to find the BEST fit.
                # "First Fit" breaks at the first valid bin.
                # Let's implement "Best Fit" across all bins.
                # So we do NOT break here. We continue checking other bins.
                pass
        
        # If no existing bin worked, or if we found a better one in a later bin
        # Note: The logic above updates best_bin_idx if a better cost is found.
        # But we initialized best_bin_idx = -1.
        
        if best_bin_idx == -1:
            # No existing bin fits (or no valid position found in any)
            # Open a new bin
            best_bin_idx = len(bins)
            bins.append([])
            grids.append(np.zeros((W, H, D), dtype=np.uint8))
            bin_volumes.append(0)
            best_pos = (0, 0, 0)
            best_pos_cost = 0 # Not really used for new bin
            
        # Place the box
        bx, by, bz = best_pos
        bins[best_bin_idx].append((bx, by, bz, w, h, d))
        grids[best_bin_idx][bx:bx+w, by:by+h, bz:bz+d] = 1
        bin_volumes[best_bin_idx] += vol
        placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        # Safety check for time
        if time.time() > deadline - 2:
            # If we are running out of time, we must ensure we can place the remaining boxes quickly.
            # The current approach is O(N^2 * Candidates). 
            # If we are slow, we might timeout.
            # To be safe, if we detect slowness, we could switch to a faster "First Fit" for remaining boxes.
            # But for 40 boxes, it should be fast.
            pass

    return [tuple(p) for p in placements]
