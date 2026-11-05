"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val."""
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

def construct():
    """
    Construct 128 rows for targets 1..128.
    Each row contains 1..30 distinct positive 2^a*3^b*5^c terms summing to target,
    forming a divisibility antichain.
    Minimize total number of summands over all targets.
    """
    max_target = 128
    smooth = _generate_smooth(max_target)
    smooth_set = set(smooth)
    m = len(smooth)
    
    # answers[i] will hold the representation for target i
    # We use 1-based indexing for targets, so answers[1] to answers[128]
    answers = [None] * (max_target + 1)
    
    # Precompute for k=1 to k=5 (or more if needed)
    # We iterate k from 1 upwards.
    # For each k, we find all antichains of size k and assign them to their sums
    # if the sum hasn't been assigned yet (which implies minimal k).
    
    max_k_to_try = 5 # Start with 5, expand if needed
    
    for k in range(1, max_k_to_try + 1):
        # Check if all targets are solved
        if all(answers[i] is not None for i in range(1, max_target + 1)):
            break
            
        # Generate all combinations of k distinct smooth numbers
        # We can prune combinations that sum to > max_target
        # itertools.combinations doesn't support pruning easily, 
        # but m is small (38).
        
        for comb in combinations(smooth, k):
            s = sum(comb)
            if s > max_target:
                continue
            # If this sum is already solved, skip
            if answers[s] is not None:
                continue
            # Check antichain
            if _is_antichain(comb):
                answers[s] = list(comb)
                
    # If there are still unsolved targets, use DFS for them
    for n in range(1, max_target + 1):
        if answers[n] is None:
            # Try k = max_k_to_try + 1, ...
            for k in range(max_k_to_try + 1, 31):
                found = _find_antichain_dfs(n, k, smooth, smooth_set)
                if found is not None:
                    answers[n] = found
                    break
            if answers[n] is None:
                # Fallback (should not happen)
                answers[n] = [n] if n in smooth_set else [1]
                
    # Return rows 1..128
    return answers[1:]

def _find_antichain_dfs(n, k, smooth, smooth_set):
    """Find k distinct smooth numbers summing to n forming an antichain."""
    if k == 1:
        if n in smooth_set:
            return [n]
        return None
    
    candidates = [x for x in smooth if x <= n]
    if len(candidates) < k:
        return None
        
    # Sort candidates
    candidates.sort()
    
    result = [None]
    
    def dfs(remaining_sum, remaining_k, start_idx, current):
        if remaining_k == 0:
            if remaining_sum == 0:
                result[0] = current[:]
                return True
            return False
        if remaining_k == 1:
            # The last number must be remaining_sum
            if remaining_sum >= candidates[start_idx] and remaining_sum in smooth_set:
                # Check index to ensure distinctness and order? 
                # Actually, we just need it to be in the list of candidates 
                # and not in current. Since we pick from sorted list and 
                # enforce increasing order, if it's >= last picked, it's distinct.
                # But we need to ensure it's actually in the smooth list.
                # And check antichain with current.
                s = remaining_sum
                valid = True
                for c in current:
                    if c > s:
                        c, s = s, c
                    if s % c == 0:
                        valid = False
                        break
                if valid:
                    current.append(s)
                    result[0] = current[:]
                    current.pop()
                    return True
            return False
            
        # Pruning: 
        # Minimum sum of remaining_k numbers starting from start_idx
        min_possible = 0
        count = 0
        for i in range(start_idx, len(candidates)):
            min_possible += candidates[i]
            count += 1
            if count == remaining_k:
                break
        if count < remaining_k or min_possible > remaining_sum:
            return False
            
        # Maximum sum check? 
        # Max sum of remaining_k numbers from end?
        # We need to pick numbers <= remaining_sum.
        
        # Iterate
        for i in range(start_idx, len(candidates)):
            a = candidates[i]
            if a > remaining_sum:
                break
            # Check antichain with current
            valid = True
            for c in current:
                if c > a:
                    c, a = a, c
                if a % c == 0:
                    valid = False
                    break
            if not valid:
                continue
                
            current.append(a)
            if dfs(remaining_sum - a, remaining_k - 1, i + 1, current):
                return True
            current.pop()
            
        return False
        
    dfs(n, k, 0, [])
    return result[0]

# EVOLVE-BLOCK-END
