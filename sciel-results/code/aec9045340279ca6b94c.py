"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second margin for safety
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Precompute orientations for each box to avoid repeated permutation generation
    # orientations[i] is a list of (w, h, d) tuples for box i
    all_orientations = []
    for w, h, d in boxes:
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        ors = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                ors.append(perm)
        all_orientations.append(ors)

    # Candidate sorting functions to diversify strategies
    def sort_by_volume(idx):
        return boxes[idx][0] * boxes[idx][1] * boxes[idx][2]
    
    def sort_by_max_dim(idx):
        return max(boxes[idx])
        
    def sort_by_min_dim(idx):
        return min(boxes[idx])
        
    def sort_by_aspect(idx):
        w, h, d = boxes[idx]
        max_d = max(w, h, d)
        min_d = min(w, h, d)
        if min_d == 0: return 0
        return max_d / min_d

    strategies = [
        (sort_by_volume, "vol"),
        (sort_by_max_dim, "max"),
        (sort_by_min_dim, "min"),
        (sort_by_aspect, "asp")
    ]

    best_score = -1
    best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # We will run multiple "attempts". Each attempt is a full greedy packing run.
    # We vary the sorting strategy and add randomization to tie-breaks and candidate selection.
    
    # To ensure we make progress, we'll do a few deterministic passes first, then randomized ones.
    # But randomized ones can be very effective if the seed is good.
    
    num_attempts = 0
    random.seed(42) # Fixed seed for reproducibility in this specific run, but different strategies use different logic
    
    # Helper to run one packing attempt
    def run_attempt(box_order, candidate_order_key=None):
        placed = []
        results = [None] * len(boxes)
        placed_vol = 0
        
        # Candidate positions: start with (0,0,0)
        # We will use a list of candidates. To avoid explosion, we limit the number of candidates
        # or prune them. For 60 boxes, the number of corners can grow, but usually stays manageable
        # if we prune dominated positions or limit the list size.
        # A simple "skyline" approach is complex. Let's stick to corner generation but prune duplicates.
        
        candidates = [(0, 0, 0)]
        # To keep candidates sorted for "Best Fit" we need to evaluate all.
        # For "First Fit" we just take the first valid one in a specific order.
        
        # For this strategy, we will use Best Fit Decreasing (BFD) logic within the attempt:
        # For each box, find the position that minimizes some cost function (e.g., distance to origin, or wasted space).
        # Common BFD metric: minimize (x + w/2)^2 + (y + h/2)^2 + (z + d/2)^2 (center of mass)
        # Or simply: minimize x, then y, then z (compactness).
        
        # Let's use a hybrid: 
        # 1. Generate all valid (pos, orient) pairs.
        # 2. Score them.
        # 3. Pick the best.
        
        # To speed up, we can limit the number of candidate positions we check if the list gets too large.
        MAX_CANDIDATES = 500
        if len(candidates) > MAX_CANDIDATES:
            # Keep the "most promising" candidates? 
            # Simple heuristic: keep the ones with smallest x, y, z sums.
            candidates.sort(key=lambda p: p[0] + p[1] + p[2])
            candidates = candidates[:MAX_CANDIDATES]

        for idx in box_order:
            if time.time() > deadline:
                return None # Timeout
            
            w, h, d = boxes[idx]
            best_score_val = float('inf')
            best_placement = None
            best_pos = None
            best_orient = None
            
            # Get unique orientations for this box
            ors = all_orientations[idx]
            
            # Iterate candidates
            # To improve speed, we can break early if we find a "perfect" fit? 
            # Or if the score is already very good?
            
            for cx, cy, cz in candidates:
                if time.time() > deadline:
                    return None
                
                for ow, oh, od in ors:
                    # Bounds check
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Overlap check
                    overlap = False
                    for px, py, pz, pw, ph, pd, _ in placed:
                        if not (cx + ow <= px or px + pw <= cx or 
                                cy + oh <= py or py + ph <= cy or 
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    
                    if overlap:
                        continue
                    
                    # Valid placement found. Calculate score.
                    # Metric: Minimize the "waste" in the bounding box of the placed items?
                    # Or minimize distance to (0,0,0) to keep it compact.
                    # Let's use: (cx + ow/2.0)**2 + (cy + oh/2.0)**2 + (cz + od/2.0)**2
                    score = (cx + ow / 2.0)**2 + (cy + oh / 2.0)**2 + (cz + od / 2.0)**2
                    
                    # Add a small random noise to break ties and diversify attempts
                    # Only add noise if we are in a "randomized" attempt? 
                    # Actually, the box_order is already different. 
                    # But we can also randomize the candidate iteration order slightly?
                    # No, let's keep the scoring deterministic but the order of boxes varied.
                    # However, if we want to explore more, we can randomize the tie-breaking.
                    
                    if score < best_score_val:
                        best_score_val = score
                        best_placement = (cx, cy, cz, ow, oh, od)
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
            
            if best_placement:
                cx, cy, cz, ow, oh, od = best_placement
                placed.append((cx, cy, cz, ow, oh, od, idx))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
                placed_vol += ow * oh * od
                
                # Generate new candidates
                new_candidates = [
                    (cx + ow, cy, cz),
                    (cx, cy + oh, cz),
                    (cx, cy, cz + od)
                ]
                
                # Add to candidates, avoiding duplicates
                # Using a set is O(N) per insert, but N is small. 
                # For speed, just append and dedupe occasionally? 
                # Or use a set for candidates.
                # Let's use a set for candidates to ensure uniqueness.
                # But we need to iterate over them. 
                # Let's maintain a list `candidates` and a set `candidate_set`.
                # Wait, the `candidates` variable is a list. 
                # Let's convert to set logic locally for efficiency if it gets big.
                # For now, simple append + dedup check.
                for nc in new_candidates:
                    if nc not in candidates: # O(N) check, but N is small (<500)
                        candidates.append(nc)
                        
                # Prune if too many
                if len(candidates) > MAX_CANDIDATES:
                    candidates.sort(key=lambda p: p[0] + p[1] + p[2])
                    candidates = candidates[:MAX_CANDIDATES]
            else:
                results[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any None results (shouldn't happen if loop completes, but safe)
        for i in range(len(boxes)):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)
                
        return results, placed_vol

    # Strategy 1: Deterministic Best Fit Decreasing by Volume
    # We will run multiple variations.
    
    # Variation 1: Sort by Volume Desc
    order1 = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    res, vol = run_attempt(order1)
    if res and vol > best_score:
        best_score = vol
        best_results = res

    # Variation 2: Sort by Max Dim Desc
    order2 = sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    res, vol = run_attempt(order2)
    if res and vol > best_score:
        best_score = vol
        best_results = res

    # Variation 3: Sort by Min Dim Desc (often good for filling gaps)
    order3 = sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
    res, vol = run_attempt(order3)
    if res and vol > best_score:
        best_score = vol
        best_results = res

    # Variation 4: Randomized tie-breaking / Shuffling within similar sizes
    # We can create a "bucket" sort: group by volume range, then shuffle within buckets.
    
    # Let's do a few randomized restarts with volume-based sorting but shuffled ties.
    for restart in range(20):
        if time.time() > deadline - 5: # Stop if close to deadline
            break
            
        # Create a slightly perturbed order
        # Group indices by volume
        vol_map = {}
        for i in range(len(boxes)):
            v = boxes[i][0]*boxes[i][1]*boxes[i][2]
            # Round volume to bucketize? Or just sort and shuffle adjacent?
            # Let's just sort by volume, then apply a random swap probability to adjacent items?
            # Simpler: Sort by volume, then for each item, with prob 0.1, swap with next?
            
        # Better: Sort by volume, but add random noise to the sort key.
        noisy_keys = []
        for i in range(len(boxes)):
            v = boxes[i][0]*boxes[i][1]*boxes[i][2]
            noise = random.uniform(0, v * 0.1) # 10% noise
            noisy_keys.append((v + noise, i))
        
        # Sort by noisy volume descending
        noisy_keys.sort(reverse=True)
        order = [i for _, i in noisy_keys]
        
        res, vol = run_attempt(order)
        if res and vol > best_score:
            best_score = vol
            best_results = res

    # If we still have time, try a few more with different sorting metrics
    if time.time() < deadline - 10:
        # Sort by aspect ratio
        order4 = sorted(range(len(boxes)), key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i])>0 else 0, reverse=True)
        res, vol = run_attempt(order4)
        if res and vol > best_score:
            best_score = vol
            best_results = res

        # Sort by sum of dims
        order5 = sorted(range(len(boxes)), key=lambda i: sum(boxes[i]), reverse=True)
        res, vol = run_attempt(order5)
        if res and vol > best_score:
            best_score = vol
            best_results = res

    return best_results
# EVOLVE-BLOCK-END
