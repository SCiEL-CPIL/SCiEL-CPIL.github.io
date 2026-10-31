import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _is_golomb(u):
    """Check if set u is a Golomb ruler (all pairwise differences are distinct)."""
    if len(u) < 2:
        return True
    diffs = set()
    n = len(u)
    for i in range(n):
        ui = u[i]
        for j in range(i+1, n):
            d = ui - u[j]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _construct_greedy_golomb(target_size, max_val=100000):
    """
    Construct a set that is a Golomb ruler (maximizes |U-U| relative to size)
    using a greedy approach. This maximizes the numerator of the ratio.
    """
    u = [0]
    # Greedily add the smallest possible next element that keeps the set a Golomb ruler
    # This tends to produce compact rulers, which helps the denominator (max U)
    while len(u) < target_size:
        found = False
        # Start checking from the last element + 1
        # Optimization: We can start from a heuristic lower bound
        last = u[-1]
        # A known heuristic for next element in Golomb ruler is roughly last * 1.3 or similar,
        # but simple incremental search is safer for small n.
        # To speed up, we can skip some values if n is large, but for n<50 it's fine.
        for next_val in range(last + 1, max_val):
            candidate = u + [next_val]
            if _is_golomb(candidate):
                u = candidate
                found = True
                break
        if not found:
            break
    return u

def _optimize_set(u, timeout=10.0):
    """
    Improve a set using local search focused on ratio.
    Moves: Replace an element with a nearby one that improves score.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return [0, 1]
    
    current_score = _score_from_set(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    
    # Precompute current sums/diffs for faster delta evaluation? 
    # For now, full re-evaluation is O(n^2) which is fast for n<100.
    
    n = len(u)
    max_val = max(u)
    
    # We will try to improve by swapping elements or adding/removing.
    # Since we want to maximize log(|D|/|S|) / log(2M+1),
    # increasing |D| (distinct diffs) is good.
    # decreasing |S| (distinct sums) is good.
    # decreasing M is good.
    
    # Strategy: Try to replace each element with a nearby value.
    # We prioritize replacing elements that cause collisions in D or S.
    
    for step in range(5000):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
            break
            
        # Pick a random element to perturb (excluding 0)
        if n < 2:
            break
        idx = random.randint(1, n - 1)
        old_val = u[idx]
        
        # Try a few candidate replacements
        # Range: +/- 10% of current value, or fixed small steps
        search_range = max(1, int(old_val * 0.1))
        
        # Generate candidates
        candidates = []
        for delta in range(-search_range, search_range + 1):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 0:
                continue
            if new_val in u:
                continue
            candidates.append(new_val)
            
        if not candidates:
            continue
            
        # Shuffle to ensure randomness
        random.shuffle(candidates)
        
        improved = False
        for new_val in candidates[:5]: # Limit tries per step
            new_u = u[:]
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            
            if len(new_u) != n:
                continue
                
            s = _score_from_set(new_u)
            if s > current_score:
                u = new_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
                improved = True
                break
                
        if not improved:
            # Try adding an element if size is small
            if n < 40 and random.random() < 0.1:
                # Add a random element in a gap
                if n > 1:
                    idx_add = random.randint(0, n-2)
                    low = u[idx_add]
                    high = u[idx_add+1]
                    if high - low > 1:
                        new_val = random.randint(low+1, high-1)
                        if new_val not in u:
                            test_u = sorted(u + [new_val])
                            s = _score_from_set(test_u)
                            if s > current_score:
                                u = test_u
                                current_score = s
                                if current_score > best_score:
                                    best_score = current_score
                                    best_u = u[:]
                                n += 1
            # Try removing an element if size is large
            elif n > 10 and random.random() < 0.1:
                idx_rm = random.randint(1, n-1)
                test_u = u[:]
                test_u.pop(idx_rm)
                s = _score_from_set(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    n -= 1
                    
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Strategy:
    # 1. Generate Golomb rulers of various sizes. These maximize |U-U|.
    # 2. Optimize them locally to reduce |U+U| and max(U) if possible, 
    #    or to adjust the ratio.
    
    # We know from theory that for a Golomb ruler, |U-U| = n(n-1)+1.
    # |U+U| is at least 2n-1.
    # Ratio is roughly n(2n-1)/ (2n-1) ~ n? No.
    # |U-U| ~ n^2. |U+U| ~ n^2.
    # For Golomb ruler, max U is roughly n^2 * constant.
    # log(2M+1) ~ 2 log n.
    # log(|D|/|S|) is small.
    
    # Let's try sizes 5 to 50.
    for size in range(5, 51):
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Generate a Golomb ruler of this size
        u_golomb = _construct_greedy_golomb(size)
        
        if not u_golomb:
            continue
            
        # Evaluate base score
        s_base = _score_from_set(u_golomb)
        
        # Try to optimize this specific Golomb ruler
        # We might be able to "break" the Golomb property slightly to reduce max(U) significantly
        # or reduce |U+U| by creating sum collisions, if the gain in denominator outweighs loss in numerator.
        
        u_opt, s_opt = _optimize_set(u_golomb, timeout=5.0)
        
        if s_opt > best_s:
            best_s = s_opt
            best_u = u_opt[:]
            
        # Also try a "compressed" version: take the Golomb ruler and scale it down?
        # No, must be integers.
        
        # Alternative: Random sets with structure
        # Generate a set with geometric growth, then optimize
        if time.time() < _DEADLINE - 10.0:
             # Geometric base
             u_geo = [0]
             curr = 1
             growth = random.uniform(1.2, 2.0)
             for i in range(1, size):
                 curr = int(curr * growth)
                 u_geo.append(curr)
             
             u_opt2, s_opt2 = _optimize_set(u_geo, timeout=5.0)
             if s_opt2 > best_s:
                 best_s = s_opt2
                 best_u = u_opt2[:]
                 
    # Final polish on the global best
    if time.time() < _DEADLINE - 2.0:
        u_final, s_final = _optimize_set(best_u, timeout=10.0)
        if s_final > best_s:
            best_s = s_final
            best_u = u_final[:]
            
    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
