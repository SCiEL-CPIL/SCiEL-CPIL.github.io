import time
import math
import numpy as np
import random

def _set_sum_diff_numpy(U):
    """
    Computes |U+U| and |U-U| using NumPy for efficiency.
    U is a numpy array of unique non-negative integers.
    """
    # U+U: All pairwise sums
    # To avoid O(N^2) memory for huge N, we can use the fact that we just need the count of unique sums.
    # However, for N up to a few hundred, N^2 is manageable.
    # Let's use a set for exactness first, but the prompt asks for NumPy boolean indexing strategy.
    # A pure NumPy approach for unique count of pairwise sums is tricky without sorting/flattening.
    # For N=100, 100x100=10k elements. Sorting 10k elements is fast.
    
    U_sum = np.add.outer(U, U)
    U_diff = np.subtract.outer(U, U)
    
    # Count unique elements
    # np.unique returns sorted unique elements.
    n_plus = len(np.unique(U_sum))
    n_minus = len(np.unique(U_diff))
    
    return n_plus, n_minus

def _score_numpy(U):
    """
    Computes the score for a given set U (list or numpy array).
    Score = 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
    """
    if not U or len(U) == 0:
        return 0.0, []
    
    U_arr = np.array(sorted(list(set(U))), dtype=np.int64)
    
    if 0 not in U_arr:
        # The task requires 0 in U. If not present, we should probably ensure it or handle it.
        # The construct function will ensure 0 is there.
        pass
        
    mx = U_arr.max()
    if mx <= 0:
        return 0.0, U_arr.tolist()
        
    n_plus, n_minus = _set_sum_diff_numpy(U_arr)
    
    if n_plus == 0:
        return 0.0, U_arr.tolist()
        
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, U_arr.tolist()
        
    ratio = n_minus / n_plus
    if ratio <= 0:
        return 0.0, U_arr.tolist()
        
    val = 1.0 + math.log(ratio) / log_den
    return val, U_arr.tolist()

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    # Initial set must contain 0.
    # Start with a small good set found by quick search or heuristic.
    # [0, 1, 3] is a decent start.
    # Let's try a few small sets to find a good seed.
    candidates_seed = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
        [0, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 9]
    ]
    
    best_val = -1e9
    best_set = [0, 1, 3]
    
    for c in candidates_seed:
        v, s = _score_numpy(c)
        if v > best_val:
            best_val = v
            best_set = s
            
    # Greedy Construction
    # We iteratively add one element at a time.
    # We maintain the current set U.
    # At each step, we generate a pool of candidate elements to add.
    # We score U + {c} for each c in the pool.
    # We pick the c that maximizes the score.
    # We stop if no candidate improves the score, or if we hit the time limit.
    
    U = list(best_set)
    current_val = best_val
    
    # To speed up scoring, we can cache the sum/diff sets?
    # No, adding an element changes the sum/diff sets significantly.
    # We just recompute.
    
    # We need to be careful about the size of U.
    # If |U| is large, N^2 computation becomes slow.
    # Let's limit the size of U to maybe 50-100 elements?
    # The paper mentions sizes 2003 and 54265, but those are for specific constructions.
    # For a generic greedy search, keeping N small might be better for the ratio if the elements are "independent".
    # However, larger N generally increases |U-U| and |U+U|.
    # The ratio |U-U|/|U+U| tends to be around 2 for "random" sets (since U-U has positive and negative parts, roughly double the size of U+U if no overlaps).
    # Wait, |U-U| is at most 2*max(U)+1. |U+U| is at most 2*max(U).
    # So ratio is at most ~2.
    # log(2) / log(2M+1).
    # To maximize this, we want M to be small?
    # But if M is small, |U-U|/|U+U| might be small due to overlaps.
    # For a set with no additive relations (Sidon-like), |U+U| ~ N^2/2.
    # |U-U| ~ N^2.
    # Ratio ~ 2.
    # Score ~ 1 + log(2)/log(2M+1).
    # To maximize, we want M as small as possible while maintaining the "independent" property.
    # But if M is too small, we can't have many elements.
    # If we have N elements in [0, M], M must be at least ~N^2 for Sidon.
    # So log(2M) ~ 2 log N.
    # Score ~ 1 + log(2) / (2 log N).
    # This decreases as N increases.
    # So maybe small N is better?
    # Let's check N=3, M=3 (0,1,3).
    # U+U = {0,1,2,3,4,6}. Size 6.
    # U-U = {-3,-2,-1,0,1,2,3}. Size 7.
    # Ratio 7/6 = 1.166.
    # Log(1.166) = 0.154.
    # Log(7) = 1.946.
    # Score = 1 + 0.154/1.946 = 1.079.
    
    # Let's check N=2, M=1 (0,1).
    # U+U = {0,1,2}. Size 3.
    # U-U = {-1,0,1}. Size 3.
    # Ratio 1. Score 1.
    
    # Let's check N=4, M=7 (0,1,3,7) - Golomb ruler?
    # U+U:
    # 0,1,3,7
    # 1,2,4,8
    # 3,4,6,10
    # 7,8,10,14
    # Unique: 0,1,2,3,4,6,7,8,10,14. Size 10.
    # U-U:
    # Range -7 to 7.
    # Differences:
    # 1-0=1, 3-0=3, 7-0=7
    # 0-1=-1, 3-1=2, 7-1=6
    # 0-3=-3, 1-3=-2, 7-3=4
    # 0-7=-7, 1-7=-6, 3-7=-4
    # Unique: -7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7. Size 13.
    # Ratio 13/10 = 1.3.
    # Log(1.3) = 0.262.
    # Log(15) = 2.708.
    # Score = 1 + 0.262/2.708 = 1.096.
    # Improvement!
    
    # Let's check N=5, M=11 (0,1,3,7,12)?
    # Or maybe the greedy algorithm will find a better sequence.
    
    # The greedy algorithm will explore adding elements.
    
    max_iterations = 50
    current_U = U
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        mx = max(current_U)
        
        # Generate candidates
        # We want to explore elements that might be "independent".
        # Lacunary sequences grow fast.
        # Candidates:
        # 1. mx + 1 to mx + 50 (small steps)
        # 2. mx * 2, mx * 3, ... (multiples)
        # 3. Random numbers in [mx+1, mx*3]
        # 4. Numbers that complete a Golomb ruler pattern?
        
        candidates = set()
        
        # Small additions
        for d in range(1, 20):
            candidates.add(mx + d)
            
        # Multiples
        for k in range(2, 8):
            candidates.add(mx * k)
            candidates.add(mx * k + 1)
            candidates.add(mx * k - 1)
            
        # Random
        rng = random.Random(int(time.time()*1000) + mx)
        for _ in range(50):
            candidates.add(rng.randint(mx + 1, mx * 3))
            
        # Also try some specific "good" numbers?
        # Powers of 2?
        p = 1
        while p < mx * 2:
            p *= 2
            candidates.add(p)
            candidates.add(p+1)
            
        candidates = list(candidates)
        
        best_candidate = None
        best_candidate_val = current_val
        
        # To speed up, we can pre-filter candidates that are too close to existing elements?
        # Not necessarily bad, but might cause overlaps.
        
        for c in candidates:
            if time.time() > deadline - 5.0:
                break
                
            new_U = current_U + [c]
            v, s = _score_numpy(new_U)
            
            if v > best_candidate_val:
                best_candidate_val = v
                best_candidate = c
                
        if best_candidate is not None:
            current_U.append(best_candidate)
            current_val = best_candidate_val
            if current_val > best_val:
                best_val = current_val
                best_set = sorted(current_U)
        else:
            # No improvement found in this pool.
            # Try a different pool? Or stop?
            # Let's try a wider random pool once.
            rng = random.Random(12345)
            wider_candidates = [rng.randint(mx + 1, mx * 10) for _ in range(20)]
            
            improved = False
            for c in wider_candidates:
                if time.time() > deadline - 5.0:
                    break
                new_U = current_U + [c]
                v, s = _score_numpy(new_U)
                if v > best_candidate_val:
                    best_candidate_val = v
                    best_candidate = c
                    improved = True
                    break
                    
            if improved:
                current_U.append(best_candidate)
                current_val = best_candidate_val
                if current_val > best_val:
                    best_val = current_val
                    best_set = sorted(current_U)
            else:
                # Stagnation. Try removing an element?
                # Or just stop.
                # Let's try removing the largest element and see if a smaller max helps?
                # Unlikely, as log denominator shrinks, but ratio might drop more.
                # Let's just break.
                break
                
    # Final check
    v, s = _score_numpy(best_set)
    if v > best_val:
        best_val = v
        best_set = s
        
    return best_set

# EVOLVE-BLOCK-END
