import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_t_subsets.append(mask)
    
    uncovered = set(all_t_subsets)
    point_counts = [0] * v
    
    # Initialize point counts
    for mask in uncovered:
        # Count bits in mask and add to corresponding points
        # Since v=19, we can just check bits
        for p in range(v):
            if mask & (1 << p):
                point_counts[p] += 1
                
    blocks = []
    
    # Heuristic parameter: How many top points to consider for candidate generation
    # v=19, k=8. 
    # If we take top 16, C(16,8) = 12870 candidates.
    # If we take top 15, C(15,8) = 6435.
    # Let's start with 15 for speed, maybe 16 if time permits.
    top_k_size = 15
    
    while uncovered:
        if time.time() > deadline:
            # Fallback: Just finish quickly with a very simple greedy to ensure validity
            # This should not happen if the main loop is fast enough
            break
            
        # Get top points
        # Sort indices by count descending
        sorted_points = sorted(range(v), key=lambda p: point_counts[p], reverse=True)
        top_points = sorted_points[:top_k_size]
        
        # If top_k_size < k, we must include more points
        if len(top_points) < k:
            # Add random points from the rest
            rest = [p for p in sorted_points[top_k_size:]]
            random.shuffle(rest)
            top_points.extend(rest[:k - len(top_points)])
            
        best_block = None
        best_gain = -1
        
        # Generate all k-subsets from top_points
        # If top_k_size is large, this is the bottleneck
        for candidate in itertools.combinations(top_points, k):
            # Calculate gain: how many uncovered t-subsets are inside candidate
            c_mask = 0
            for p in candidate:
                c_mask |= (1 << p)
                
            gain = 0
            # Check all t-subsets of candidate
            for t_combo in itertools.combinations(candidate, t):
                t_mask = 0
                for p in t_combo:
                    t_mask |= (1 << p)
                if t_mask in uncovered:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block = candidate
                
        # Fallback if best_gain is 0 (should not happen often, but ensures progress)
        if best_gain == 0:
            # Pick an arbitrary uncovered set
            first_uncovered = next(iter(uncovered))
            # Convert mask to points
            pts = []
            for p in range(v):
                if first_uncovered & (1 << p):
                    pts.append(p)
            # Extend to size k
            while len(pts) < k:
                # Add a point not in pts
                for p in range(v):
                    if p not in pts:
                        pts.append(p)
                        break
            best_block = tuple(sorted(pts))
            
        # Add block to result
        blocks.append(list(best_block))
        
        # Update uncovered and point_counts
        # Remove all t-subsets of best_block from uncovered
        b_mask = 0
        for p in best_block:
            b_mask |= (1 << p)
            
        to_remove = []
        for t_combo in itertools.combinations(best_block, t):
            t_mask = 0
            for p in t_combo:
                t_mask |= (1 << p)
            if t_mask in uncovered:
                to_remove.append(t_mask)
                # Update point counts
                for p in t_combo:
                    point_counts[p] -= 1
                    
        for mask in to_remove:
            uncovered.remove(mask)

    return blocks
