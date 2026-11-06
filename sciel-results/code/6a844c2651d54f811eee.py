"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    n_smooth = len(smooth_numbers)
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Precompute incompatibility: incomp[i][j] = True if smooth_numbers[i] and smooth_numbers[j] cannot both be in antichain
    incomp = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_numbers[i], smooth_numbers[j]
                if b % a == 0 or a % b == 0:
                    incomp[i][j] = True
    
    # Precompute antichains of size 1, 2, 3
    antichain_by_sum = {k: {} for k in range(1, 31)}
    
    # Size 1
    for i in range(n_smooth):
        s = smooth_numbers[i]
        if s <= 128 and s not in antichain_by_sum[1]:
            antichain_by_sum[1][s] = [smooth_numbers[i]]
    
    # Size 2
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if not incomp[i][j] and s not in antichain_by_sum[2]:
                antichain_by_sum[2][s] = [a, b]
    
    # Size 3
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s2 = a + b
            if s2 > 128:
                break
            if incomp[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s3 = s2 + c
                if s3 > 128:
                    break
                if incomp[i][k] or incomp[j][k]:
                    continue
                if s3 not in antichain_by_sum[3]:
                    antichain_by_sum[3][s3] = [a, b, c]
    
    # For each target, find minimum antichain size
    answers = []
    for target in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback
            if target in val_to_idx:
                answers.append([target])
            elif target > 1:
                answers.append([1, target - 1])
            else:
                answers.append([1])
            continue
        
        found = None
        # Try sizes 1, 2, 3
        for k in range(1, 4):
            if target in antichain_by_sum[k]:
                found = antichain_by_sum[k][target]
                break
        
        if found is None:
            # DFS for size >= 4
            found = _find_antichain(target, smooth_numbers, incomp, val_to_idx, deadline)
        
        answers.append(found)
    
    return answers

def _find_antichain(target, smooth_numbers, incomp, val_to_idx, deadline):
    """Find minimum-size antichain (size >= 4) summing to target using DFS with pruning."""
    sys.setrecursionlimit(10000)
    
    # Candidates: smooth numbers <= target, sorted descending
    candidates = [x for x in smooth_numbers if x <= target]
    candidates.sort(reverse=True)
    n_cand = len(candidates)
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build incompatibility matrix for candidates
    cand_incomp = [[False] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        gi = val_to_idx[candidates[i]]
        for j in range(n_cand):
            if i != j:
                gj = val_to_idx[candidates[j]]
                cand_incomp[i][j] = incomp[gi][gj]
    
    # Precompute suffix sums for pruning
    suffix_sum = [0] * (n_cand + 1)
    for i in range(n_cand - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + candidates[i]
    
    best = None
    best_size = 31  # Max allowed is 30
    
    def dfs(remaining, chosen, start_idx, current_size):
        nonlocal best, best_size
        
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if current_size < best_size:
                best_size = current_size
                best = list(chosen)
            return
        
        if current_size >= best_size:
            return
        
        if start_idx >= n_cand:
            return
        
        if remaining < 0:
            return
        
        # Pruning: not enough sum available from remaining candidates
        if suffix_sum[start_idx] < remaining:
            return
        
        # Try each candidate from start_idx
        for idx in range(start_idx, n_cand):
            if time.time() > deadline:
                return
            
            val = candidates[idx]
            if val > remaining:
                continue
            
            # Check compatibility with all chosen
            valid = True
            for c in chosen:
                ci = cand_to_idx[c]
                if cand_incomp[ci][idx]:
                    valid = False
                    break
            if not valid:
                continue
            
            chosen.append(val)
            dfs(remaining - val, chosen, idx + 1, current_size + 1)
            chosen.pop()
            
            # Early termination: if we found a 4-term solution, that's optimal (since we already checked 1-3)
            if best_size == 4:
                return
    
    dfs(target, [], 0, 0)
    
    if best is not None:
        return best
    
    # Fallback: greedy approach
    remaining = target
    chosen = []
    for val in candidates:
        if val > remaining:
            continue
        valid = True
        for c in chosen:
            ci = val_to_idx[c]
            vi = val_to_idx[val]
            if incomp[ci][vi]:
                valid = False
                break
        if valid:
            chosen.append(val)
            remaining -= val
            if remaining == 0:
                break
    
    if remaining == 0 and chosen:
        return chosen
    else:
        # Last resort
        if target in val_to_idx:
            return [target]
        else:
            return [1, target - 1] if target > 1 else [1]
# EVOLVE-BLOCK-END
