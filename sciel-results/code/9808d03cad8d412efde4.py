"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
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

def _is_antichain(terms):
    """Check if no term divides another in the list."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _find_antichain_for_target(n, smooth_up_to_n, max_terms=30, time_limit=5.0):
    """
    Find a representation of n as a sum of distinct 2,3,5-smooth numbers
    forming a divisibility antichain, minimizing the number of terms.
    """
    start_time = time.time()
    
    # Pre-filter: smooth numbers <= n
    candidates = [x for x in smooth_up_to_n if x <= n]
    
    # Try increasing number of terms from 1 up to max_terms
    for k in range(1, max_terms + 1):
        if time.time() - start_time > time_limit:
            break
        
        # For k=1: just check if n itself is smooth
        if k == 1:
            if n in smooth_up_to_n:
                return [n]
            continue
        
        # For k=2: check pairs
        if k == 2:
            for i in range(len(candidates)):
                a = candidates[i]
                rem = n - a
                if rem <= a or rem > n:
                    continue
                if rem in smooth_up_to_n and rem != a:
                    # Check antichain: neither divides the other
                    if a % rem != 0 and rem % a != 0:
                        return [a, rem]
            continue
        
        # For k >= 3: use DFS with pruning
        # Sort candidates in descending order for better pruning
        sorted_cands = sorted(candidates, reverse=True)
        
        # Precompute divisibility relationships for pruning
        # For a given set of chosen terms, a new term must not be divisible by any
        # chosen term, and must not divide any chosen term
        
        best_result = None
        
        def dfs(idx, remaining, chosen, depth):
            nonlocal best_result
            if time.time() - start_time > time_limit:
                return
            if remaining == 0:
                if depth > 0 and _is_antichain(chosen):
                    best_result = list(chosen)
                return
            if remaining < 0:
                return
            if depth == k:
                return
            if idx >= len(sorted_cands):
                return
            
            # Pruning: if even taking the largest remaining candidates
            # can't sum to 'remaining', stop
            # Simple bound: sum of largest (k-depth) candidates from idx onwards
            # But this is expensive to compute each time; use a simpler bound
            if remaining > 0:
                # Maximum possible sum from remaining candidates
                # Just check if remaining is achievable
                pass
            
            for i in range(idx, len(sorted_cands)):
                cand = sorted_cands[i]
                if cand > remaining:
                    continue
                if time.time() - start_time > time_limit:
                    return
                
                # Check antichain condition with already chosen terms
                valid = True
                for chosen_term in chosen:
                    if cand % chosen_term == 0 or chosen_term % cand == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                chosen.append(cand)
                dfs(i + 1, remaining - cand, chosen, depth + 1)
                chosen.pop()
                
                if best_result is not None:
                    return
                
                # If we've used too many terms, break
                # Actually depth is tracked, so this is handled
        
        dfs(0, n, [], 0)
        
        if best_result is not None:
            return best_result
    
    # Fallback: if no antichain found with limited search, try a different approach
    # For small n, we can be more exhaustive
    if n <= 50:
        for k in range(1, min(n, 15) + 1):
            if k == 1:
                if n in smooth_up_to_n:
                    return [n]
                continue
            # Use combinations for small k
            if k <= 4:
                for combo in combinations(candidates, k):
                    if sum(combo) == n and _is_antichain(combo):
                        return list(combo)
    
    # Last resort: return a valid but possibly non-minimal representation
    # Greedy approach
    result = []
    rem = n
    for cand in sorted(candidates, reverse=True):
        if cand > rem:
            continue
        # Check antichain
        valid = True
        for chosen_term in result:
            if cand % chosen_term == 0 or chosen_term % cand == 0:
                valid = False
                break
        if valid:
            result.append(cand)
            rem -= cand
            if rem == 0:
                break
    
    if rem == 0 and result:
        return result
    
    # If all else fails, return [n] if n is smooth, else raise
    if n in smooth_up_to_n:
        return [n]
    
    # This should not happen for 1..128
    raise ValueError(f"No representation found for {n}")

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    smooth_all = _generate_smooth(128)
    smooth_set = set(smooth_all)
    
    answers = []
    for n in range(1, 129):
        if time.time() - start_time > deadline:
            # Emergency: use a simple fallback for remaining targets
            for m in range(n, 129):
                if m in smooth_set:
                    answers.append([m])
                else:
                    # Greedy fallback
                    result = []
                    rem = m
                    for cand in sorted([x for x in smooth_all if x <= m], reverse=True):
                        if cand > rem:
                            continue
                        valid = True
                        for ct in result:
                            if cand % ct == 0 or ct % cand == 0:
                                valid = False
                                break
                        if valid:
                            result.append(cand)
                            rem -= cand
                            if rem == 0:
                                break
                    if rem == 0 and result:
                        answers.append(result)
                    else:
                        # Last resort: just use [m] even if not smooth (shouldn't happen)
                        answers.append([m])
            break
        
        ans = _find_antichain_for_target(n, smooth_all, max_terms=30, time_limit=3.0)
        answers.append(ans)
    
    return answers
# EVOLVE-BLOCK-END
