"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

# Precompute smooth numbers up to 128
def _generate_smooth(max_val):
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

SMOOTH = _generate_smooth(128)
SMOOTH_SET = set(SMOOTH)
SMOOTH_LEN = len(SMOOTH)

# Precompute: for each smooth number, which other smooth numbers are "incompatible" (one divides the other)
# compat[i][j] = True if SMOOTH[i] and SMOOTH[j] are compatible (neither divides the other)
COMPAT = [[False] * SMOOTH_LEN for _ in range(SMOOTH_LEN)]
for i in range(SMOOTH_LEN):
    for j in range(i + 1, SMOOTH_LEN):
        a, b = SMOOTH[i], SMOOTH[j]
        if b % a != 0:
            COMPAT[i][j] = True
            COMPAT[j][i] = True

# Precompute suffix sums for pruning: min_sum_from[idx][k] = sum of k smallest smooth numbers from index idx onward
# This helps prune DFS branches where remaining < minimum possible sum
MAX_K = 30
min_sum_from = [[0] * (MAX_K + 1) for _ in range(SMOOTH_LEN + 1)]
for idx in range(SMOOTH_LEN - 1, -1, -1):
    for k in range(1, MAX_K + 1):
        if idx + k <= SMOOTH_LEN:
            min_sum_from[idx][k] = SMOOTH[idx] + min_sum_from[idx + 1][k - 1]
        else:
            min_sum_from[idx][k] = float('inf')

# Also: max possible sum with k terms starting from idx (using largest available)
max_sum_from = [[0] * (MAX_K + 1) for _ in range(SMOOTH_LEN + 1)]
for idx in range(SMOOTH_LEN - 1, -1, -1):
    for k in range(1, MAX_K + 1):
        if idx + k <= SMOOTH_LEN:
            max_sum_from[idx][k] = SMOOTH[idx] + max_sum_from[idx + 1][k - 1]
        else:
            max_sum_from[idx][k] = 0

def _find_min_antichain(n, max_terms=30):
    """Find representation of n as sum of distinct smooth numbers forming an antichain,
    minimizing the number of terms."""
    
    # Try 1 term first
    if n in SMOOTH_SET:
        return [n]
    
    # Try 2 terms
    for i in range(SMOOTH_LEN):
        a = SMOOTH[i]
        if a >= n:
            break
        b = n - a
        if b in SMOOTH_SET and b > a and b % a != 0:
            return [a, b]
    
    # Try 3 terms using optimized nested loops
    for i in range(SMOOTH_LEN):
        a = SMOOTH[i]
        if a >= n:
            break
        remaining = n - a
        if remaining < 2:  # need at least 2 more positive terms
            break
        for j in range(i + 1, SMOOTH_LEN):
            b = SMOOTH[j]
            if b >= remaining:
                break
            c = remaining - b
            if c <= b:
                continue
            if c not in SMOOTH_SET:
                continue
            if c > n:
                continue
            # Check antichain using precomputed compatibility
            # Find indices: a=SMOOTH[i], b=SMOOTH[j], c=?
            # c must be > b, so its index > j
            # Check: COMPAT[i][j] and COMPAT[i][c_idx] and COMPAT[j][c_idx]
            # We need to find c's index
            # Since c > b and both in SMOOTH, c_idx > j
            # Quick check: b % a != 0, c % a != 0, c % b != 0
            if b % a == 0 or c % a == 0 or c % b == 0:
                continue
            return [a, b, c]
    
    # Try 4+ terms using optimized DFS with pruning
    # Use index-based tracking for fast compatibility checks
    smooth_le_n = [x for x in SMOOTH if x <= n]
    slen = len(smooth_le_n)
    
    # Map value to index in smooth_le_n for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth_le_n)}
    
    # Precompute compatibility within smooth_le_n
    compat_local = [[False] * slen for _ in range(slen)]
    for i in range(slen):
        for j in range(i + 1, slen):
            a, b = smooth_le_n[i], smooth_le_n[j]
            if b % a != 0:
                compat_local[i][j] = True
                compat_local[j][i] = True
    
    # Precompute min_sum_local: min sum of k terms from position idx onward in smooth_le_n
    min_sum_local = [[0] * (MAX_K + 1) for _ in range(slen + 1)]
    for idx in range(slen - 1, -1, -1):
        for k in range(1, MAX_K + 1):
            if idx + k <= slen:
                min_sum_local[idx][k] = smooth_le_n[idx] + min_sum_local[idx + 1][k - 1]
            else:
                min_sum_local[idx][k] = float('inf')
    
    def dfs(remaining, start_idx, chosen, chosen_idx, depth):
        """Find terms to sum to remaining using smooth_le_n[start_idx:], 
        with 'depth' more terms needed. chosen_idx tracks indices in smooth_le_n for compat checks."""
        if depth == 0:
            if remaining == 0:
                return list(chosen)
            return None
        if depth == 1:
            if remaining in val_to_idx:
                c_idx = val_to_idx[remaining]
                if c_idx >= start_idx:
                    # Check compatibility with all chosen
                    ok = True
                    for ci in chosen_idx:
                        if not compat_local[ci][c_idx]:
                            ok = False
                            break
                    if ok:
                        return chosen + [remaining]
            return None
        
        # Pruning: minimum sum of 'depth' terms from start_idx onward
        if min_sum_local[start_idx][depth] > remaining:
            return None
        # Pruning: maximum sum check (remaining must be achievable)
        # The max sum of 'depth' terms from start_idx is sum of last 'depth' terms
        if start_idx + depth <= slen:
            max_possible = sum(smooth_le_n[slen - depth : slen])
            if remaining > max_possible:
                return None
        
        for idx in range(start_idx, slen):
            val = smooth_le_n[idx]
            if val > remaining:
                break
            # Check compatibility with all chosen
            ok = True
            for ci in chosen_idx:
                if not compat_local[ci][idx]:
                    ok = False
                    break
            if not ok:
                continue
            # Pruning: after picking this, can we still achieve remaining - val with depth-1 terms?
            new_remaining = remaining - val
            if depth - 1 > 0 and min_sum_local[idx + 1][depth - 1] > new_remaining:
                break  # All subsequent values are larger, so this won't improve
            
            chosen.append(val)
            chosen_idx.append(idx)
            result = dfs(new_remaining, idx + 1, chosen, chosen_idx, depth - 1)
            if result is not None:
                return result
            chosen.pop()
            chosen_idx.pop()
        return None
    
    # Try 4 terms
    result = dfs(n, 0, [], [], 4)
    if result is not None:
        return result
    
    # Try 5 terms
    result = dfs(n, 0, [], [], 5)
    if result is not None:
        return result
    
    # Try 6+ terms
    max_k = min(30, n)
    for k in range(6, max_k + 1):
        result = dfs(n, 0, [], [], k)
        if result is not None:
            return result
    
    raise ValueError(f"No antichain representation found for n={n}")

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 second deadline for safety
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Return what we have so far with None for remaining
            while len(answers) < 128:
                answers.append(None)
            return answers
        
        result = _find_min_antichain(n)
        answers.append(result)
    
    return answers
# EVOLVE-BLOCK-END
