"""Rewritten program: direct, term-minimal smooth-antichain search with deadline guard."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1

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

def _div_free(a, b):
    # True if neither divides the other
    return (a % b != 0) and (b % a != 0)

# Build global smooth list up to 128 once
_SMOOTH = _gen_smooth(128)
# Precompute, for each smooth value, which other smooth values are incompatible (one divides the other)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}

def construct():
    start = time.time()
    DEADLINE = 85.0   # safety margin before 90s kill

    answers = [None] * 129   # answers[n] for n in 1..128

    # Precompute smooth values up to 128
    smooth_vals = _SMOOTH
    n_smooth = len(smooth_vals)
    
    # Precompute divisibility incompatibility as bitmasks
    # inc_mask[i] has bit j set if smooth_vals[i] and smooth_vals[j] are incompatible (one divides the other)
    inc_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_vals[i]
        for j in range(i + 1, n_smooth):
            vj = smooth_vals[j]
            if vi % vj == 0 or vj % vi == 0:
                inc_mask[i] |= (1 << j)
                inc_mask[j] |= (1 << i)
    
    # For each target, search k = 1..30 for a div-free smooth sum
    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break
        found = None
        for k in range(1, 31):
            if time.time() - start > DEADLINE:
                break
            if k == 1:
                if n in _VAL_INDEX:
                    found = [n]
                    break
                continue
            if k > n:
                break
            # Find candidates that are <= n
            candidates = [v for v in smooth_vals if v <= n]
            found = _find_sum_fast(candidates, n, k, smooth_vals, inc_mask, _VAL_INDEX)
            if found is not None:
                break
        answers[n] = found

    # If deadline hit and some missing, do a greedy fill for the rest (keeps validity)
    if any(answers[n] is None for n in range(1, 129)):
        for n in range(1, 129):
            if answers[n] is None:
                answers[n] = _greedy_fill(n, smooth_vals, inc_mask, _VAL_INDEX)

    return answers[1:129]

def _find_sum_fast(candidates, n, k, smooth_vals, inc_mask, val_index):
    """Find k distinct smooth values from candidates summing to n, pairwise div-free.
    Uses precomputed incompatibility masks and exponent-vector sorting for early pruning."""
    if not candidates or k <= 0:
        return None
    
    # Map candidate values to their indices in smooth_vals
    cand_indices = []
    for v in candidates:
        idx = val_index.get(v)
        if idx is not None:
            cand_indices.append(idx)
    
    if len(cand_indices) < k:
        return None
    
    # Sort by exponent vector (a,b,c) ascending, then by value ascending
    # This allows immediate dominance rejection: if we pick a vector that dominates
    # a previously picked one, we can skip it early
    def exp_key(idx):
        v = smooth_vals[idx]
        a = b = c = 0
        while v % 2 == 0:
            v //= 2; a += 1
        while v % 3 == 0:
            v //= 3; b += 1
        while v % 5 == 0:
            v //= 5; c += 1
        return (a, b, c, smooth_vals[idx])
    
    cand_indices.sort(key=exp_key)
    cand_vals = [smooth_vals[i] for i in cand_indices]
    
    # Precompute suffix min and max sums for pruning
    # suffix_min[i] = sum of smallest (k-1) values from cand_indices[i:]
    # suffix_max[i] = sum of largest (k-1) values from cand_indices[i:]
    m = len(cand_indices)
    suffix_min = [0] * (m + 1)
    suffix_max = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1]
        suffix_max[i] = suffix_max[i + 1]
        # We need to track the k-1 smallest/largest from position i onward
        # Simplified: use cumulative sums
    # Better: precompute for each position, the sum of the next (k-1) smallest and largest
    # But that's complex. Let's use a simpler bound:
    # At position pos with remaining_k, the min possible sum is sum of next remaining_k smallest
    # and max is sum of next remaining_k largest
    
    best = None
    chosen = []
    chosen_mask = 0  # bitmask of chosen indices
    
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
        
        # Compute min and max possible sum from positions start..m-1 choosing remaining_k items
        # Min: take the remaining_k smallest values from cand_vals[start:]
        # Max: take the remaining_k largest values from cand_vals[start:]
        avail = cand_vals[start:]
        if len(avail) < remaining_k:
            return False
        # For efficiency, since cand_vals is sorted by exponent vector (not value),
        # we need to compute min/max of choosing remaining_k from avail
        # This is expensive, so use a looser bound:
        min_possible = sum(sorted(avail)[:remaining_k])
        max_possible = sum(sorted(avail, reverse=True)[:remaining_k])
        if remaining_sum < min_possible or remaining_sum > max_possible:
            return False
        
        for pos in range(start, m):
            idx = cand_indices[pos]
            v = cand_vals[pos]
            if v > remaining_sum:
                continue
            
            # Check compatibility with all chosen using bitmask
            if chosen_mask & inc_mask[idx]:
                continue
            
            # Additional pruning: after picking this, can we still reach remaining_sum - v?
            new_remaining = remaining_sum - v
            new_k = remaining_k - 1
            if new_k > 0:
                if pos + 1 + new_k > m:
                    continue
                # Quick check: min possible from remaining
                if new_remaining > 0:
                    # At least need new_k values each >= 1, so new_remaining >= new_k
                    if new_remaining < new_k:
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
    """Fallback fill for targets not solved by main search."""
    if n in val_index:
        return [n]
    
    # Try k=2 to k=6
    for k in range(2, 7):
        candidates = [v for v in smooth_vals if v <= n]
        found = _find_sum_fast(candidates, n, k, smooth_vals, inc_mask, val_index)
        if found is not None:
            return found
    
    # Last resort: return [n] if smooth, else try harder
    if n in val_index:
        return [n]
    
    # Try all pairs
    for i in range(len(smooth_vals)):
        a = smooth_vals[i]
        if a > n:
            break
        b = n - a
        if b <= 0:
            continue
        if b in val_index:
            j = val_index[b]
            if not (inc_mask[i] & (1 << j)):
                return [a, b]
    
    return [n] if n in val_index else [1]
# EVOLVE-BLOCK-END
