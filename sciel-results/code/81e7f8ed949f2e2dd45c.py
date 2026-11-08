import time
import random
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []

    # Sort indices by volume descending for BFDV
    volumes = [w*h*d for w,h,d in boxes]
    order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    # We will try a few permutations or just the sorted one with random restarts if time permits
    # For 40 boxes, BFDV with sorted order is very strong.
    # We can also try a few random perturbations of the sort order.
    
    best_placements = None
    best_score = float('inf')
    
    # Pre-calculate box dimensions for speed
    box_dims = [(boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def try_order(ord):
        """
        Attempt to pack boxes in the given order using Best-Fit strategy.
        Returns (placements, num_bins) or None if failed (though BFDV rarely fails completely 
        unless box > container, which we assume is not the case or handled by evaluator).
        """
        bins = [] # List of lists of (x, y, z, w, h, d)
        
        for i in ord:
            w, h, d = box_dims[i]
            
            # If box doesn't fit in empty container
            if w > W or h > H or d > D:
                # In a valid packing task, this shouldn't happen. 
                # If it does, we can't pack it. But let's assume valid input.
                # If invalid, we might return a dummy placement or let it fail.
                # The contract says "places every box".
                pass
                
            best_bin_idx = -1
            best_pos = (0, 0, 0)
            best_metric = float('inf')
            
            # Check existing bins
            for b_idx, bin_boxes in enumerate(bins):
                # Try to find a position in this bin
                # Candidate positions: (0,0,0) and top/side of existing boxes
                # For simplicity and speed, we check a limited set of candidates:
                # 1. Origin
                # 2. For each box in bin, try (x+w, y, z), (x, y+h, z), (x, y, z+d)
                
                candidates = [(0, 0, 0)]
                for (ex, ey, ez, ew, eh, ed) in bin_boxes:
                    candidates.append((ex + ew, ey, ez))
                    candidates.append((ex, ey + eh, ez))
                    candidates.append((ex, ey, ez + ed))
                
                # Remove duplicates for speed
                candidates = list(set(candidates))
                
                for (cx, cy, cz) in candidates:
                    # Check bounds
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    # Check overlap with all boxes in this bin
                    valid = True
                    for (ex, ey, ez, ew, eh, ed) in bin_boxes:
                        if not (cx + w <= ex or ex + ew <= cx or
                                cy + h <= ey or ey + eh <= cy or
                                cz + d <= ez or ez + ed <= cz):
                            valid = False
                            break
                    
                    if valid:
                        # Metric: Prefer lower z, then lower y, then lower x (Fits tightly at bottom)
                        # Or simply minimize the "wasted space" heuristically.
                        # Standard Best Fit: minimize the maximum dimension of the "gap"
                        # Here, simple lexicographic (z, y, x) works well for BFDV.
                        metric = (cz, cy, cx)
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_idx = b_idx
                            best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                # Open new bin
                bins.append([])
                best_bin_idx = len(bins) - 1
                best_pos = (0, 0, 0)
            
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
            # Time check
            if time.time() > deadline:
                return None

        # Calculate score for this placement
        num_bins = len(bins)
        
        # Find last bin (largest index)
        # The problem says "largest label defines the last bin".
        # Our bins are 0-indexed and consecutive.
        last_bin_idx = num_bins - 1
        last_bin_boxes = bins[last_bin_idx]
        
        occupied_vol = sum(w*h*d for (x,y,z,w,h,d) in last_bin_boxes)
        total_vol = W * H * D
        fraction = occupied_vol / total_vol if total_vol > 0 else 0
        
        score = num_bins + 0.1 * (1 - fraction)
        
        # Construct placements list in original order
        # We need to map box i to its bin and position
        placements = [None] * n
        
        for b_idx, bin_boxes in enumerate(bins):
            # We need to know which original box index went into which slot in this bin
            # This is tricky because we just appended. 
            # Let's re-trace or store the mapping during the packing process.
            # Actually, the function `try_order` returns placements. 
            # I need to modify `try_order` to return the actual placement tuples.
            
            # Wait, I lost the mapping of which box index went where.
            # Let's refactor `try_order` to track it.
            pass
            
        return score, None # Placeholder

    # Refactored try_order to return full placements
    def solve(ord):
        bins = [] # List of lists of (box_index, x, y, z)
        
        for i in ord:
            w, h, d = box_dims[i]
            
            if w > W or h > H or d > D:
                # Invalid box, return failure
                return None, None

            best_bin_idx = -1
            best_pos = (0, 0, 0)
            best_metric = float('inf')
            
            for b_idx, bin_boxes in enumerate(bins):
                candidates = [(0, 0, 0)]
                for (bi, ex, ey, ez) in bin_boxes:
                    ew, eh, ed = box_dims[bi]
                    candidates.append((ex + ew, ey, ez))
                    candidates.append((ex, ey + eh, ez))
                    candidates.append((ex, ey, ez + ed))
                
                candidates = list(set(candidates))
                
                for (cx, cy, cz) in candidates:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    valid = True
                    for (bi, ex, ey, ez) in bin_boxes:
                        ew, eh, ed = box_dims[bi]
                        if not (cx + w <= ex or ex + ew <= cx or
                                cy + h <= ey or ey + eh <= cy or
                                cz + d <= ez or ez + ed <= cz):
                            valid = False
                            break
                    
                    if valid:
                        metric = (cz, cy, cx)
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_idx = b_idx
                            best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                bins.append([])
                best_bin_idx = len(bins) - 1
                best_pos = (0, 0, 0)
            
            bins[best_bin_idx].append((i, best_pos[0], best_pos[1], best_pos[2]))
            
            if time.time() > deadline:
                return None, None

        num_bins = len(bins)
        last_bin_idx = num_bins - 1
        last_bin_boxes = bins[last_bin_idx]
        
        occupied_vol = 0
        for (bi, x, y, z) in last_bin_boxes:
            w, h, d = box_dims[bi]
            occupied_vol += w * h * d
            
        total_vol = W * H * D
        fraction = occupied_vol / total_vol if total_vol > 0 else 0
        score = num_bins + 0.1 * (1 - fraction)
        
        # Build output
        placements = [None] * n
        for b_idx, bin_boxes in enumerate(bins):
            for (bi, x, y, z) in bin_boxes:
                w, h, d = box_dims[bi]
                placements[bi] = (b_idx, x, y, z, w, h, d)
                
        return score, placements

    # Strategy: 
    # 1. Try the standard BFDV order (sorted by volume desc).
    # 2. If time permits, try a few random shuffles of the top K items to break ties or improve fit.
    
    # Initial run
    score, placements = solve(order)
    if placements is not None:
        best_score = score
        best_placements = placements
    
    # If we have time, try some random restarts with slight perturbations
    # Since n=40, we can afford a few quick runs.
    # Perturbation: swap a few items in the sorted list.
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Create a perturbed order
        # Keep the largest items mostly stable, shuffle the rest
        perm_order = order[:]
        
        # Swap random pairs in the lower half of the sorted list
        # Or just random shuffle the whole list? 
        # Random shuffle usually hurts BFDV. 
        # Better: Randomize the order of items with similar volumes.
        
        # Simple perturbation: Take the sorted list, and for the last 20 items, shuffle them.
        if n > 20:
            cut = max(0, n - 20)
            tail = perm_order[cut:]
            random.shuffle(tail)
            perm_order[cut:] = tail
        else:
            random.shuffle(perm_order)
            
        s, p = solve(perm_order)
        if p is not None:
            if s < best_score:
                best_score = s
                best_placements = p
        
        # If we've tried a few and time is running out, break
        if attempts > 50:
            break

    if best_placements is None:
        # Fallback to shelf if everything failed (should not happen)
        return []
        
    return best_placements
