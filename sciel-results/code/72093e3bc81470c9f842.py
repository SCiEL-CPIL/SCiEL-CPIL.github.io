"""Optimized program: fast paths for small k, backtracking for larger k."""
# EVOLVE-BLOCK-START
import time

def _gen_smooth(limit):
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

# Build global smooth list up to 128 once
_SMOOTH = _gen_smooth(128)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}
_SMOOTH_SET = set(_SMOOTH)
_N_SMOOTH = len(_SMOOTH)

# Precompute incompatibility masks globally
_INC_MASK = [0] * _N_SMOOTH
for _i in range(_N_SMOOTH):
    _vi = _SMOOTH[_i]
    for _j in range(_i + 1, _N_SMOOTH):
        _vj = _SMOOTH[_j]
        if _vi % _vj == 0 or _vj % _vi == 0:
            _INC_MASK[_i] |= (1 << _j)
            _INC_MASK[_j] |= (1 << _i)

# Precompute smooth values up to each n for fast candidate filtering
# _SMOOTH_UP_TO[n] = list of smooth values <= n, sorted
_SMOOTH_UP_TO = [[] for _ in range(129)]
for _n in range(1, 129):
    _SMOOTH_UP_TO[_n] = [v for v in _SMOOTH if v <= _n]

def construct():
    start = time.time()
    DEADLINE = 85.0
    
    answers = [None] * 129
    smooth_vals = _SMOOTH
    inc_mask = _INC_MASK
    val_index = _VAL_INDEX
    
    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break
            
        found = None
        
        # k=1
        if n in val_index:
            found = [n]
        else:
            # k=2
            found = _find_k2(n, smooth_vals, inc_mask, val_index)
            if not found:
                # k=3
                found = _find_k3(n, smooth_vals, inc_mask, val_index)
                if not found:
                    # k >= 4: use backtracking
                    for k in range(4, min(31, n + 1)):
                        found = _find_sum_fast(n, k, smooth_vals, inc_mask, val_index)
                        if found: break
                        
        answers[n] = found

    # Fallback for any missing (should be rare with optimized search)
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _greedy_fill(n, smooth_vals, inc_mask, val_index)

    return answers[1:129]

def _find_k2(n, smooth_vals, inc_mask, val_index):
    """Find 2 distinct smooth numbers summing to n, compatible."""
    # Only need to check a < n/2 to ensure a < b
    half_n = n // 2
    for a in smooth_vals:
        if a > half_n:
            break
        b = n - a
        if b in _SMOOTH_SET:
            idx_a = val_index[a]
            idx_b = val_index[b]
            if not (inc_mask[idx_a] & (1 << idx_b)):
                return [a, b]
    return None

def _find_k3(n, smooth_vals, inc_mask, val_index):
    """Find 3 distinct smooth numbers summing to n, compatible."""
    # a < b < c, a + b + c = n
    # a < n/3, b < (n-a)/2
    
    for i_a, a in enumerate(smooth_vals):
        if a * 3 >= n:
            break
        
        # b must be > a and a + b < n (so c > 0)
        # Also b < (n - a) / 2 to ensure c > b
        max_b = (n - a) // 2
        
        for i_b in range(i_a + 1, len(smooth_vals)):
            b = smooth_vals[i_b]
            if b > max_b:
                break
            c = n - a - b
            if c <= b:
                continue
            
            if c in _SMOOTH_SET:
                idx_c = val_index[c]
                # Check compatibility
                if inc_mask[i_a] & (1 << i_b):
                    continue
                if inc_mask[i_a] & (1 << idx_c):
                    continue
                if inc_mask[i_b] & (1 << idx_c):
                    continue
                
                return [a, b, c]
    return None

def _find_sum_fast(n, k, smooth_vals, inc_mask, val_index):
    """Find k distinct smooth values summing to n, pairwise div-free."""
    # Get candidates <= n
    candidates = _SMOOTH_UP_TO[n]
    if len(candidates) < k:
        return None
    
    # Map to indices
    cand_indices = [val_index[v] for v in candidates]
    # Sort descending for better pruning
    cand_indices.sort(key=lambda i: smooth_vals[i], reverse=True)
    
    # Precompute suffix sums for tighter bounds
    # suffix_sum[i] = sum of the largest min(k, len-candidates) items from position i
    # Actually, for pruning we want: max possible sum with remaining_k items starting from pos
    # This is the sum of the largest remaining_k items from pos onwards
    m = len(cand_indices)
    # Precompute prefix sums of the sorted candidate values
    cand_vals = [smooth_vals[i] for i in cand_indices]
    # suffix_max[i][r] = max sum of r items from positions i to end
    # But this is expensive. Instead, use a simpler bound:
    # The max sum of r items from pos onwards is the sum of the r largest items from pos onwards
    # Since they're sorted descending, it's just the sum of cand_vals[pos:pos+r]
    
    best = None
    chosen = []
    chosen_mask = 0
    
    def backtrack(start, remaining_k, remaining_sum):
        nonlocal best, chosen, chosen_mask
        if best is not None:
            return True
        if remaining_k == 0:
            if remaining_sum == 0:
                best = [smooth_vals[i] for i in chosen]
                return True
            return False
        if remaining_k > remaining_sum:
            return False
        if start + remaining_k > m:
            return False
        
        # Tighter pruning: max possible sum with remaining_k items from start
        # is the sum of the first remaining_k items from start (since sorted desc)
        if start + remaining_k <= m:
            max_possible = sum(cand_vals[start:start + remaining_k])
            if remaining_sum > max_possible:
                return False
        
        # Min possible sum with remaining_k items from start
        # is the sum of the last remaining_k items from start to end
        if m - start >= remaining_k:
            min_possible = sum(cand_vals[m - remaining_k:m])
            if remaining_sum < min_possible:
                return False
            
        for pos in range(start, m):
            idx = cand_indices[pos]
            v = cand_vals[pos]
            if v > remaining_sum:
                continue
            
            if chosen_mask & inc_mask[idx]:
                continue
            
            chosen.append(idx)
            chosen_mask |= (1 << idx)
            if backtrack(pos + 1, remaining_k - 1, remaining_sum - v):
                return True
            chosen.pop()
            chosen_mask ^= (1 << idx)
        
        return False
    
    backtrack(0, k, n)
    return best

def _greedy_fill(n, smooth_vals, inc_mask, val_index):
    """Fallback fill."""
    if n in val_index:
        return [n]
    for k in range(2, min(10, n + 1)):
        found = _find_sum_fast(n, k, smooth_vals, inc_mask, val_index)
        if found:
            return found
    return [1] if n >= 1 else [1]
# EVOLVE-BLOCK-END
