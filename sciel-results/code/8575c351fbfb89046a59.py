"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import sys

_N = 20
_MAX_VAL = 10000
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin under 90s kill

def _score(a):
    """Return max(|A+A|, |A*A|) for sorted distinct positive integers a."""
    a = np.asarray(a, dtype=np.int64)
    if len(a) != _N or np.any(a < 1) or np.any(a > _MAX_VAL):
        return 10**9
    if len(set(a.tolist())) != _N:
        return 10**9
    s = a[:, None] + a[None, :]
    p = a[:, None] * a[None, :]
    return max(len(np.unique(s)), len(np.unique(p)))

def _is_valid(a):
    a = list(a)
    return (len(a) == _N and len(set(a)) == _N and
            all(isinstance(x, int) and 1 <= x <= _MAX_VAL for x in a))

def _greedy_construct(seed):
    """
    Construct a set by greedily adding integers that minimize the increase
    in the sumset size. This tends to produce sets with small |A+A|.
    """
    random.seed(seed)
    a = [1]
    # We want 20 numbers. Start with 1.
    # To keep numbers small and distinct, we look in a local range first.
    # A good heuristic for small sumsets is to use numbers that are "additively independent" 
    # or close to a geometric progression but with small ratios.
    # However, pure GP is too rigid. Greedy addition allows deviation.
    
    # Initial pool: small integers
    # We will add numbers one by one.
    # To avoid huge numbers, we restrict the search for the next number to [last+1, last+1000]
    # or similar, but actually, for minimal sumset, we want numbers to be "dense" in a way 
    # that sums collide.
    
    # A better greedy: 
    # 1. Start with 1.
    # 2. For each new element x, calculate the new sumset size if x is added.
    # 3. Pick x that minimizes the increase in |A+A|.
    # 4. Tie-break by minimizing |A*A| increase, then by smallest x.
    
    current_set = {1}
    a_list = [1]
    
    # We need to be careful with the range. If we pick too large numbers, sums don't collide.
    # If we pick too small, we run out of distinct integers.
    # Let's try to stay within [1, 1000] initially, then expand if needed.
    
    for _ in range(_N - 1):
        if time.time() > _DEADLINE - 5:
            break
            
        best_x = None
        best_sum_size = 10**9
        best_prod_size = 10**9
        best_x_val = 10**9
        
        # Search range for next number
        # Start from last element + 1 to last element + 500
        # Also check some small numbers if we haven't used them
        last = a_list[-1]
        
        # Candidate range: 
        # 1. Small numbers not in set (1 to 20)
        # 2. Numbers near the last added number (last+1 to last+200)
        # 3. Random samples from [1, 1000]
        
        candidates = set()
        # Small numbers
        for x in range(1, 30):
            if x not in current_set:
                candidates.add(x)
        # Near last
        for x in range(last + 1, last + 200):
            if x not in current_set and x <= _MAX_VAL:
                candidates.add(x)
        # Random samples
        for _ in range(50):
            x = random.randint(1, 1000)
            if x not in current_set:
                candidates.add(x)
        
        # Filter valid candidates
        candidates = [x for x in candidates if 1 <= x <= _MAX_VAL]
        
        if not candidates:
            # Fallback: just pick next integer
            x = last + 1
            while x in current_set:
                x += 1
            candidates = [x]
        
        for x in candidates:
            # Calculate new sumset size
            new_set = current_set | {x}
            a_arr = np.array(sorted(new_set), dtype=np.int64)
            s = a_arr[:, None] + a_arr[None, :]
            sum_size = len(np.unique(s))
            
            if sum_size > best_sum_size:
                continue
            if sum_size < best_sum_size:
                best_sum_size = sum_size
                best_x = x
                # Calculate product size for tie-breaking
                p = a_arr[:, None] * a_arr[None, :]
                best_prod_size = len(np.unique(p))
                best_x_val = x
            else:
                # Tie in sum size, check product size
                if sum_size == best_sum_size:
                    p = a_arr[:, None] * a_arr[None, :]
                    prod_size = len(np.unique(p))
                    if prod_size < best_prod_size:
                        best_prod_size = prod_size
                        best_x = x
                        best_x_val = x
                    elif prod_size == best_prod_size and x < best_x_val:
                        best_x = x
                        best_x_val = x
        
        if best_x is None:
            # Should not happen
            best_x = last + 1
            while best_x in current_set:
                best_x += 1
        
        current_set.add(best_x)
        a_list.append(best_x)
        
    return sorted(a_list)

def _local_search_product_focus(a, max_iter=500):
    """
    Local search that specifically targets reducing |A*A| while keeping |A+A| small.
    We swap elements or perturb them.
    """
    a = np.array(a, dtype=np.int64)
    best_score = _score(a)
    best_a = a.copy()
    
    # Pre-compute current sum and prod sizes
    s_curr = best_a[:, None] + best_a[None, :]
    p_curr = best_a[:, None] * best_a[None, :]
    sum_size_curr = len(np.unique(s_curr))
    prod_size_curr = len(np.unique(p_curr))
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        # Try single element perturbation
        i = random.randint(0, _N - 1)
        old_val = a[i]
        
        # Try a few deltas
        deltas = random.sample([-10, -5, -2, -1, 1, 2, 5, 10], 4)
        
        for delta in deltas:
            new_val = old_val + delta
            if new_val < 1 or new_val > _MAX_VAL:
                continue
            if new_val in set(a.tolist()) - {old_val}:
                continue
            
            new_a = a.copy()
            new_a[i] = new_val
            new_a.sort()
            
            # Quick check: sumset size
            s_new = new_a[:, None] + new_a[None, :]
            sum_size_new = len(np.unique(s_new))
            
            # If sumset size increases significantly, skip
            if sum_size_new > sum_size_curr + 5:
                continue
            
            # Check product set size
            p_new = new_a[:, None] * new_a[None, :]
            prod_size_new = len(np.unique(p_new))
            
            new_score = max(sum_size_new, prod_size_new)
            
            if new_score < best_score:
                a = new_a
                best_score = new_score
                best_a = a.copy()
                sum_size_curr = sum_size_new
                prod_size_curr = prod_size_new
                improved = True
                break
            elif new_score == best_score and prod_size_new < prod_size_curr:
                # Prefer lower product size even if max is same
                a = new_a
                best_a = a.copy()
                sum_size_curr = sum_size_new
                prod_size_curr = prod_size_new
                improved = True
                break
        
        if not improved:
            # Try a swap
            j = random.randint(0, _N - 1)
            if i == j:
                continue
            # Swap a[i] and a[j]
            a[i], a[j] = a[j], a[i]
            a.sort()
            
            s_new = a[:, None] + a[None, :]
            sum_size_new = len(np.unique(s_new))
            if sum_size_new > sum_size_curr + 5:
                # Revert
                a = best_a.copy()
                continue
            
            p_new = a[:, None] * a[None, :]
            prod_size_new = len(np.unique(p_new))
            new_score = max(sum_size_new, prod_size_new)
            
            if new_score < best_score:
                best_score = new_score
                best_a = a.copy()
                sum_size_curr = sum_size_new
                prod_size_curr = prod_size_new
                improved = True
            elif new_score == best_score and prod_size_new < prod_size_curr:
                best_a = a.copy()
                sum_size_curr = sum_size_new
                prod_size_curr = prod_size_new
                improved = True
            else:
                # Revert
                a = best_a.copy()
                
        if not improved:
            # Small random jump to escape local minima
            if random.random() < 0.1:
                k = random.randint(0, _N - 1)
                a[k] += random.randint(-50, 50)
                a = np.clip(a, 1, _MAX_VAL)
                # Ensure distinct
                for idx in range(_N):
                    while a[idx] in set(a.tolist()) - {a[idx]}:
                        a[idx] += 1
                a.sort()
                # Update stats
                s_curr = a[:, None] + a[None, :]
                p_curr = a[:, None] * a[None, :]
                sum_size_curr = len(np.unique(s_curr))
                prod_size_curr = len(np.unique(p_curr))
                best_score = max(sum_size_curr, prod_size_curr)
                best_a = a.copy()
            else:
                # Accept no improvement for a few steps then break? 
                # Or just continue.
                pass
                
    return best_a.tolist(), best_score

def construct():
    start_time = time.time()
    best_set = None
    best_score = 10**9
    
    # 1. Generate diverse candidates using Greedy Construction
    # Different seeds yield different greedy paths
    seeds = [42, 123, 999, 7, 314, 271, 161, 8, 13, 21]
    
    for seed in seeds:
        if time.time() > _DEADLINE - 10:
            break
        try:
            candidate = _greedy_construct(seed)
            if len(candidate) != _N:
                continue
            if not _is_valid(candidate):
                continue
            sc = _score(candidate)
            if sc < best_score:
                best_score = sc
                best_set = candidate
        except Exception:
            continue
    
    # 2. Also try some simple geometric progressions as baselines
    for base in [1.1, 1.2, 1.3, 1.5, 1.7, 2.0]:
        if time.time() > _DEADLINE - 10:
            break
        vals = [int(1 * (base ** i)) for i in range(_N)]
        # Ensure distinct and within range
        vals = sorted(set(v for v in vals if 1 <= v <= _MAX_VAL))
        if len(vals) < _N:
            # Pad with small numbers if needed
            for x in range(1, _MAX_VAL+1):
                if x not in vals:
                    vals.append(x)
                if len(vals) == _N:
                    break
        if len(vals) != _N:
            continue
        vals = vals[:_N]
        sc = _score(vals)
        if sc < best_score:
            best_score = sc
            best_set = vals
    
    # 3. Local search polish on the best candidate found
    if best_set is not None:
        polished, sc = _local_search_product_focus(best_set, max_iter=500)
        if sc < best_score:
            best_score = sc
            best_set = polished
    
    # 4. If we have time, try to polish a second best candidate
    # (We didn't track second best, so we just re-run greedy with a different seed and polish)
    if time.time() < _DEADLINE - 20 and best_set is not None:
        alt_seed = 555
        try:
            alt_candidate = _greedy_construct(alt_seed)
            if _is_valid(alt_candidate):
                alt_polished, alt_sc = _local_search_product_focus(alt_candidate, max_iter=300)
                if alt_sc < best_score:
                    best_score = alt_sc
                    best_set = alt_polished
        except Exception:
            pass

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set
# EVOLVE-BLOCK-END
