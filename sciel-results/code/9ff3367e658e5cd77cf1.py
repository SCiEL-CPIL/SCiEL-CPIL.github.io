"""Divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val, sorted."""
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

def _build_divisibility_masks(smooth_list):
    """For each index i, build a bitmask of indices j where smooth_list[j] divides smooth_list[i]."""
    n = len(smooth_list)
    # divisors_mask[i] is a set of indices j such that smooth_list[j] divides smooth_list[i]
    divisors_mask = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if smooth_list[j] % smooth_list[i] == 0:
                divisors_mask[i].add(j)
    return divisors_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    max_target = 128
    smooth = _generate_smooth(max_target)
    n_smooth = len(smooth)
    
    # Precompute: for each smooth number, which other smooth numbers divide it
    # div_mask[i] = set of indices j != i such that smooth[j] divides smooth[i]
    div_mask = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div_mask[i].add(j)
    
    # For each target, find minimum number of terms
    # Try k=1,2,3,... up to some limit
    answers = [None] * max_target
    
    # Precompute which smooth numbers can be used for each target
    # For target n, only smooth numbers <= n are candidates
    candidates_for_target = []
    for n in range(1, max_target + 1):
        cands = [i for i in range(n_smooth) if smooth[i] <= n]
        candidates_for_target.append(cands)
    
    # For efficiency, precompute for each target and each k, 
    # we'll do DFS with pruning
    
    for n in range(1, max_target + 1):
        if time.time() > deadline:
            break
        if answers[n - 1] is not None:
            continue
            
        cands = candidates_for_target[n - 1]
        cand_vals = [smooth[i] for i in cands]
        cand_indices = cands
        
        # Try k from 1 upward
        found = False
        for k in range(1, 31):
            if time.time() > deadline:
                break
            if found:
                break
                
            # DFS to find k distinct smooth numbers from cands that sum to n
            # and form an antichain
            result = None
            
            def dfs(pos, remaining, chosen, chosen_set):
                nonlocal result
                if time.time() > deadline:
                    return
                if result is not None:
                    return
                if len(chosen) == k:
                    if remaining == 0:
                        result = list(chosen)
                    return
                if pos >= len(cand_vals):
                    return
                # Pruning: even if we take all remaining, can we reach the target?
                # Max possible sum from remaining elements
                if len(chosen) + (len(cand_vals) - pos) < k:
                    return
                # Min possible: if we skip too many
                # Try taking cand_vals[pos]
                val = cand_vals[pos]
                idx = cand_indices[pos]
                
                # Check antichain condition: new term must not divide or be divided by any chosen term
                valid = True
                for cidx in chosen_set:
                    if smooth[cidx] % val == 0 or val % smooth[cidx] == 0:
                        valid = False
                        break
                
                if valid and val <= remaining:
                    chosen.append(val)
                    chosen_set.add(idx)
                    dfs(pos + 1, remaining - val, chosen, chosen_set)
                    chosen.pop()
                    chosen_set.discard(idx)
                    if result is not None:
                        return
                
                # Try skipping cand_vals[pos]
                dfs(pos + 1, remaining, chosen, chosen_set)
            
            dfs(0, n, [], set())
            if result is not None:
                answers[n - 1] = result
                found = True
    
    # For any targets not yet found, use a fallback (shouldn't happen for 1..128)
    for n in range(1, max_target + 1):
        if answers[n - 1] is None:
            # Fallback: use the seed approach
            smooth_small = [x for x in smooth if x <= n]
            from itertools import combinations
            answer = None
            for count in (3, 2, 1):
                for terms in combinations(smooth_small, count):
                    if sum(terms) == n and all(b % a for a, b in combinations(terms, 2)):
                        answer = list(terms)
                        break
                if answer is not None:
                    break
            answers[n - 1] = answer
    
    return answers
# EVOLVE-BLOCK-END
