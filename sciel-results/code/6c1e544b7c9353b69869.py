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

# Precompute prefix sums for min/max pruning
_PREFIX_SUM = [0] * (_N + 1)
for i in range(_N):
    _PREFIX_SUM[i + 1] = _PREFIX_SUM[i] + _SMOOTH[i]

# Cache for DFS: (remaining_k, remaining_sum, mask) -> bool (False if failed)
_DFS_CACHE = {}

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

    # Clear cache for this specific (target, k) pair to avoid stale entries
    # Actually, we use a global cache keyed by (remaining_k, remaining_sum, mask)
    # This is safe because the state is fully determined by these three values

    # DFS with pruning and memoization
    # State: remaining_k, remaining_sum, available_mask, current_path
    
    def dfs(remaining_k, remaining_sum, available_mask, current_path):
        if remaining_k == 0:
            return remaining_sum == 0
        
        if remaining_k > 0 and remaining_sum <= 0:
            return False
            
        # Memoization check
        cache_key = (remaining_k, remaining_sum, available_mask)
        if cache_key in _DFS_CACHE:
            return _DFS_CACHE[cache_key]
        
        # Count available indices
        avail_count = bin(available_mask).count('1')
        if avail_count < remaining_k:
            _DFS_CACHE[cache_key] = False
            return False
            
        # Pruning bounds using prefix sums
        # Get the indices of available elements
        avail_indices = []
        m = available_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            avail_indices.append(idx)
            m ^= lsb
        
        # Min sum: sum of smallest remaining_k available
        min_sum = sum(_SMOOTH[i] for i in avail_indices[:remaining_k])
        if remaining_sum < min_sum:
            _DFS_CACHE[cache_key] = False
            return False
            
        # Max sum: sum of largest remaining_k available
        max_sum = sum(_SMOOTH[i] for i in avail_indices[-remaining_k:])
        if remaining_sum > max_sum:
            _DFS_CACHE[cache_key] = False
            return False
        
        # Additional pruning: each remaining number must be <= remaining_sum
        # And we need remaining_k numbers from available that are <= remaining_sum
        valid_indices = [i for i in avail_indices if _SMOOTH[i] <= remaining_sum]
        if len(valid_indices) < remaining_k:
            _DFS_CACHE[cache_key] = False
            return False
        
        # Iterate candidates - try larger values first for efficiency
        for i in reversed(valid_indices):
            v = _SMOOTH[i]
            current_path.append(i)
            
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            
            if dfs(remaining_k - 1, remaining_sum - v, new_mask, current_path):
                _DFS_CACHE[cache_key] = True
                return True
                
            current_path.pop()
            
        _DFS_CACHE[cache_key] = False
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
    # Last resort - use DFS with relaxed constraints as fallback
    # Try up to 5 terms with DFS
    for k in range(2, 6):
        if n < k * (k + 1) // 2:
            break
        _DFS_CACHE.clear()
        result = _find_minimal_rep(n, k)
        if result is not None:
            return result
    # Absolute last resort
    return [1] if n == 1 else [n]

# EVOLVE-BLOCK-END
