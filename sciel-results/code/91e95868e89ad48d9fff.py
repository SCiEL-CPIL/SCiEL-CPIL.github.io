import time

# EVOLVE-BLOCK-START

def _gen_smooth(limit):
    """Generate all 2,3,5-smooth numbers up to limit."""
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_SMOOTH_SET = set(_SMOOTH)

# Precompute divisibility compatibility masks
# _INCOMPAT_MASK[i] has bits set for all j such that _SMOOTH[i] and _SMOOTH[j] are INCOMPATIBLE
# (one divides the other, or they are the same index)
_INCOMPAT_MASK = [0] * _N
for i in range(_N):
    vi = _SMOOTH[i]
    mask = 1 << i  # Incompatible with itself (distinctness)
    for j in range(_N):
        if i == j:
            continue
        vj = _SMOOTH[j]
        if vi > vj:
            if vi % vj == 0:
                mask |= (1 << j)
        else:
            if vj % vi == 0:
                mask |= (1 << j)
    _INCOMPAT_MASK[i] = mask

# Helper to get list of indices from mask, sorted by value (which is index order)
def _get_avail_indices(mask):
    indices = []
    m = mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
    return indices

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    results = [None] * 129
    total_terms = 0

    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Fallback for remaining targets if time is up
            results[n] = _quick_fallback(n)
            total_terms += len(results[n])
            continue

        found = None
        # Try k from 1 to 30
        for k in range(1, 31):
            if k > n:
                break
            
            # Feasibility check: Can we sum k distinct positive integers to n?
            # Minimum sum is k(k+1)/2
            if n < k * (k + 1) // 2:
                break 

            found = _find_minimal_rep(n, k)
            if found is not None:
                break
        
        if found is None:
            # Should not happen for n<=128, but handle gracefully
            found = _quick_fallback(n)
        
        results[n] = found
        total_terms += len(found)

    return results[1:129]

def _find_minimal_rep(target, k):
    """
    Find k distinct smooth numbers summing to target, pairwise non-dividing.
    Returns a list of values or None.
    """
    if k == 1:
        if target in _SMOOTH_SET:
            return [target]
        return None

    # DFS with pruning
    # State: remaining_k, remaining_sum, available_mask, current_path
    
    def dfs(remaining_k, remaining_sum, available_mask, current_path):
        if remaining_k == 0:
            return remaining_sum == 0
        
        if remaining_k > 0 and remaining_sum <= 0:
            return False
            
        avail_indices = _get_avail_indices(available_mask)
        
        if len(avail_indices) < remaining_k:
            return False
            
        # Pruning bounds
        # avail_indices is sorted by value ascending (since _SMOOTH is sorted)
        
        # Min sum: sum of smallest remaining_k available
        min_sum = sum(_SMOOTH[i] for i in avail_indices[:remaining_k])
        if remaining_sum < min_sum:
            return False
            
        # Max sum: sum of largest remaining_k available
        max_sum = sum(_SMOOTH[i] for i in avail_indices[-remaining_k:])
        if remaining_sum > max_sum:
            return False
        
        # Iterate candidates
        # Heuristic: Try larger values first?
        # If we pick a large value, we reduce remaining_sum quickly.
        # If we pick a small value, we keep options open for the sum but block many multiples.
        # Let's try iterating in reverse order (largest to smallest) to see if it finds solutions faster.
        # Often, larger terms are more "efficient" in reaching the target sum with fewer terms.
        
        for i in reversed(avail_indices):
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue
                
            current_path.append(i)
            
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            
            if dfs(remaining_k - 1, remaining_sum - v, new_mask, current_path):
                return True
                
            current_path.pop()
            
        return False

    # Initial mask: all bits set
    full_mask = (1 << _N) - 1
    path = []
    
    if dfs(k, target, full_mask, path):
        return [_SMOOTH[i] for i in path]
    else:
        return None

def _quick_fallback(n):
    # If the main search fails (shouldn't happen), return a valid representation.
    if n in _SMOOTH_SET:
        return [n]
    # Try 2 terms
    for a in _SMOOTH:
        if a >= n: break
        b = n - a
        if b in _SMOOTH_SET and a != b and a % b != 0 and b % a != 0:
            return [a, b]
    # Try 3 terms
    for a in _SMOOTH:
        if a >= n: break
        for b in _SMOOTH:
            if b <= a: continue
            if a + b >= n: break
            c = n - a - b
            if c in _SMOOTH_SET and c > b:
                # Check divisibility
                if a % b != 0 and b % a != 0 and a % c != 0 and c % a != 0 and b % c != 0 and c % b != 0:
                    return [a, b, c]
    # Last resort
    return [1] if n == 1 else [n]

# EVOLVE-BLOCK-END
