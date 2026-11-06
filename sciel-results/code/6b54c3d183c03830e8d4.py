"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_set = set(smooth_numbers)
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility: div[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    div_matrix = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_numbers[i], smooth_numbers[j]
                if b % a == 0 or a % b == 0:
                    div_matrix[i][j] = True
    
    # Map value -> index in smooth_numbers
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Precompute all antichain subsets of size 1-5 and their sums
    # antichain_sums[k] = dict mapping sum -> list of terms (first found, i.e., minimal in some order)
    antichain_sums = {k: {} for k in range(1, 6)}
    
    # Size 1
    for i in range(n_smooth):
        s = smooth_numbers[i]
        if s <= 128 and s not in antichain_sums[1]:
            antichain_sums[1][s] = [smooth_numbers[i]]
    
    # Size 2
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if not div_matrix[i][j] and s not in antichain_sums[2]:
                antichain_sums[2][s] = [a, b]
    
    # Size 3
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s2 = a + b
            if s2 > 128:
                break
            if div_matrix[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s3 = s2 + c
                if s3 > 128:
                    break
                if div_matrix[i][k] or div_matrix[j][k]:
                    continue
                if s3 not in antichain_sums[3]:
                    antichain_sums[3][s3] = [a, b, c]
    
    # Size 4
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s2 = a + b
            if s2 > 128:
                break
            if div_matrix[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s3 = s2 + c
                if s3 > 128:
                    break
                if div_matrix[i][k] or div_matrix[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    d = smooth_numbers[l]
                    s4 = s3 + d
                    if s4 > 128:
                        break
                    if div_matrix[i][l] or div_matrix[j][l] or div_matrix[k][l]:
                        continue
                    if s4 not in antichain_sums[4]:
                        antichain_sums[4][s4] = [a, b, c, d]
    
    # Size 5
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s2 = a + b
            if s2 > 128:
                break
            if div_matrix[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s3 = s2 + c
                if s3 > 128:
                    break
                if div_matrix[i][k] or div_matrix[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    d = smooth_numbers[l]
                    s4 = s3 + d
                    if s4 > 128:
                        break
                    if div_matrix[i][l] or div_matrix[j][l] or div_matrix[k][l]:
                        continue
                    for m in range(l + 1, n_smooth):
                        e = smooth_numbers[m]
                        s5 = s4 + e
                        if s5 > 128:
                            break
                        if (div_matrix[i][m] or div_matrix[j][m] or 
                            div_matrix[k][m] or div_matrix[l][m]):
                            continue
                        if s5 not in antichain_sums[5]:
                            antichain_sums[5][s5] = [a, b, c, d, e]
    
    # Now for each target 1..128, find the minimal k antichain
    answers = []
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            break
        
        found = None
        for k in range(1, 6):
            if n in antichain_sums[k]:
                found = antichain_sums[k][n]
                break
        
        if found is None:
            # Fallback: DFS for k >= 6
            found = _dfs_antichain(n, smooth_numbers, div_matrix, val_to_idx, smooth_set)
        
        answers.append(found)
    
    # If we timed out, fill remaining with fallback
    while len(answers) < 128:
        n = len(answers) + 1
        found = None
        for k in range(1, 6):
            if n in antichain_sums[k]:
                found = antichain_sums[k][n]
                break
        if found is None:
            found = _dfs_antichain(n, smooth_numbers, div_matrix, val_to_idx, smooth_set)
        answers.append(found)
    
    return answers

def _dfs_antichain(n, smooth_numbers, div_matrix, val_to_idx, smooth_set):
    """DFS fallback for targets not covered by precomputed antichains of size <= 5."""
    sys.setrecursionlimit(10000)
    
    candidates = [x for x in smooth_numbers if x <= n]
    candidates_sorted = sorted(candidates, reverse=True)
    best = None
    
    def dfs(remaining, chosen, idx, max_terms):
        nonlocal best
        if best is not None and len(chosen) >= len(best):
            return
        if remaining == 0:
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if len(chosen) >= max_terms:
            return
        if idx >= len(candidates_sorted):
            return
        if remaining < 0:
            return
        
        # Pruning: even if we take all remaining candidates, can we reach 'remaining'?
        # (loose bound)
        
        val = candidates_sorted[idx]
        if val <= remaining:
            # Check antichain
            valid = True
            for c in chosen:
                if div_matrix[val_to_idx[val]][val_to_idx[c]]:
                    valid = False
                    break
            if valid:
                chosen.append(val)
                dfs(remaining - val, chosen, idx + 1, max_terms)
                chosen.pop()
        
        # Try skipping
        dfs(remaining, chosen, idx + 1, max_terms)
    
    dfs(n, [], 0, 30)
    
    if best is not None:
        return best
    
    # Absolute last resort
    if n in smooth_set:
        return [n]
    else:
        return [1, n - 1]
# EVOLVE-BLOCK-END
