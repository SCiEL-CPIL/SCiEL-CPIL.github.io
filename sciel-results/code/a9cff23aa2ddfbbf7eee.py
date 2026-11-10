"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin
    
    n = 5040
    target = 4979
    divs = _get_divisors(n)
    # divs is sorted descending
    
    # Greedy fallback
    remainder = target
    greedy_chosen = []
    for d in divs:
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
    if remainder != 0:
        return greedy_chosen
    
    best_solution = greedy_chosen
    best_k = len(greedy_chosen)
    
    # Try to find a solution with fewer divisors using meet-in-the-middle
    # For small k, enumerate all combinations of size k/2 and k-k/2
    # But for this specific problem, let's try a smarter search
    
    # Precompute suffix sums for pruning
    # divs is sorted descending
    suffix_sum = [0] * (len(divs) + 1)
    for i in range(len(divs) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divs[i]
    
    # For each k from 1 to best_k - 1, try to find a combination
    for k in range(1, best_k):
        if time.time() > deadline:
            break
        
        # Use backtracking with pruning
        # We want to select k distinct divisors that sum to target
        # divs is sorted descending, so we can prune when:
        # 1. Current sum + suffix_sum[i] < target (even taking all remaining won't reach target)
        # 2. Current sum > target
        # 3. Not enough elements left to pick (k - len(chosen) > len(divs) - i)
        
        found = False
        chosen = []
        
        def backtrack(i, current_sum, remaining_k):
            nonlocal found
            if time.time() > deadline:
                return
            if remaining_k == 0:
                if current_sum == target:
                    found = True
                    return
                return
            
            # Pruning: if we can't reach target even with max possible
            # Max possible sum from remaining elements
            if i + remaining_k > len(divs):
                return
            
            # If current_sum + sum of next remaining_k largest elements < target, prune
            # The next remaining_k largest elements starting from i are divs[i:i+remaining_k]
            # But we need to be careful - we want the maximum sum we can get
            # Since divs is sorted descending, the max sum is sum(divs[i:i+remaining_k])
            max_possible = sum(divs[i:i + remaining_k])
            if current_sum + max_possible < target:
                return
            
            # If current_sum > target, prune
            if current_sum > target:
                return
            
            # If we have exactly remaining_k elements left, check if they sum to target
            if remaining_k == len(divs) - i:
                if current_sum + suffix_sum[i] == target:
                    chosen.extend(divs[i:])
                    found = True
                return
            
            for j in range(i, len(divs) - remaining_k + 1):
                if time.time() > deadline:
                    return
                new_sum = current_sum + divs[j]
                if new_sum > target:
                    continue  # Since divs is descending, all subsequent will be smaller, 
                             # but we still need to check if we can reach target with fewer larger elements
                    # Actually, since we're going in order, and divs is descending,
                    # if divs[j] makes sum > target, then divs[j+1] < divs[j], so it might work
                    # So we can't break, we continue
                chosen.append(divs[j])
                backtrack(j + 1, new_sum, remaining_k - 1)
                chosen.pop()
                if found:
                    return
        
        backtrack(0, 0, k)
        
        if found:
            best_solution = list(chosen)
            best_k = k
            break
    
    return best_solution
# EVOLVE-BLOCK-END
