"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # 1. Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    def get_sum(s):
        return sum(s)
    
    # 2. Initial Solution: Greedy by value
    # We start with 1. Then we try to add numbers from N down to 2.
    # A number can be added if it is compatible with all currently selected numbers.
    # However, simple greedy might miss better combinations.
    # Let's use a smarter initialization: Start with all primes? No, primes conflict with composites.
    # Actually, the set of all primes <= N is a valid solution (pairwise coprime).
    # Sum of primes <= 256 is quite large.
    # Let's start with the set of all primes <= N plus 1.
    
    primes = [i for i in range(2, N + 1) if prime_factors[i] == {i}]
    initial_set = [1] + primes
    
    # Ensure distinct and sorted for consistency
    current_set = list(set(initial_set))
    current_sum = get_sum(current_set)
    best_set = list(current_set)
    best_sum = current_sum
    
    # 3. Local Search: 1-for-1 Swaps and Insertions
    # We try to improve the sum by:
    # a) Adding a number if it fits (increases sum)
    # b) Replacing a number with a larger one if it fits and increases sum
    
    # To speed up, we can iterate through candidates multiple times.
    # Candidates: 1..N
    # Since 1 is always included, we focus on 2..N.
    
    # Optimization: Precompute compatibility matrix?
    # N=256, 256x256 matrix is small.
    # compat[i][j] = True if i and j are coprime.
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            compat[i][j] = is_compatible(i, j)
            compat[j][i] = compat[i][j]
        compat[i][i] = False # A number is not compatible with itself in terms of adding it again
    
    # Helper to check if a candidate can be added to a set
    def can_add(candidate, selected_set):
        for s in selected_set:
            if s == candidate:
                return False
            if not compat[s][candidate]:
                return False
        return True
    
    # Helper to check if a candidate can replace an element
    def can_replace(candidate, remove_elem, selected_set):
        if candidate == remove_elem:
            return False
        # Check if candidate is compatible with all other elements in selected_set except remove_elem
        for s in selected_set:
            if s == remove_elem:
                continue
            if not compat[s][candidate]:
                return False
        return True

    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try Insertions: Add a number not in the set
        # We want to add numbers that increase the sum. Since all numbers are positive, any valid addition increases sum.
        # We should try to add the largest possible numbers first?
        # Or just iterate through all numbers 2..N not in current_set.
        
        # To be efficient, we can just check all numbers 1..N.
        # If a number is not in the set and can be added, we add it.
        # Since we want to maximize sum, adding any valid number is good.
        # However, adding a small number might block adding a large number later?
        # Greedy insertion usually works well if we do multiple passes.
        
        # Strategy: Iterate through numbers in descending order.
        # If we can add it, we add it.
        
        # But wait, if we add a small prime, it blocks its multiples.
        # If we add a large composite, it blocks its prime factors.
        # Generally, larger numbers have fewer "conflicts" in terms of count?
        # Actually, a prime p blocks p, 2p, 3p...
        # A composite like 16 (2^4) only blocks 2.
        # So composites are "cheaper" in terms of blocking primes?
        # No, 16 blocks 2. 2 blocks 2, 4, 6, ...
        # So 16 is a better choice than 2 if we want to keep 2's slot for something else?
        # No, if we pick 16, we use prime 2. If we pick 2, we use prime 2.
        # So they are equivalent in terms of prime usage.
        # But 16 > 2. So 16 is always better than 2 if both are compatible with the rest.
        
        # Therefore, a good heuristic is to prefer larger numbers.
        
        # Let's do a pass:
        # For each number in descending order (N down to 2):
        #   If number is not in current_set and can_add(number, current_set):
        #       Add it.
        #       improved = True
        
        # This might be slow if done naively O(N^2) per pass.
        # N=256, N^2 = 65536. This is very fast.
        
        for num in range(N, 1, -1):
            if num in current_set:
                continue
            if can_add(num, current_set):
                current_set.append(num)
                current_sum += num
                improved = True
                # Update best
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = list(current_set)
        
        # Now try 1-for-1 swaps to improve sum further.
        # Replace a smaller number in the set with a larger number not in the set.
        # For each element in current_set (sorted ascending to try replacing small ones first?):
        #   For each candidate not in current_set (sorted descending):
        #       If candidate > element and can_replace(candidate, element, current_set):
        #           Replace element with candidate.
        #           improved = True
        #           break inner loop (restart outer loop? or continue?)
        
        # To avoid O(N^3), we can be smart.
        # Let's just iterate a few times.
        
        # Sort current_set to make iteration easier
        current_set.sort()
        
        for i in range(len(current_set)):
            if time.time() > deadline:
                break
            elem = current_set[i]
            # Try to replace elem with a larger number
            # Candidates are numbers not in current_set
            # We want the largest possible candidate that is compatible.
            
            # Optimization: Instead of checking all candidates, we can check from N down to elem+1.
            for cand in range(N, elem, -1):
                if cand in current_set:
                    continue
                if can_replace(cand, elem, current_set):
                    # Perform swap
                    current_sum -= elem
                    current_set[i] = cand
                    current_sum += cand
                    improved = True
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = list(current_set)
                    # Re-sort to maintain order for next iterations?
                    # Or just break and restart the outer loop?
                    # Breaking and restarting is safer for correctness of indices.
                    current_set.sort()
                    break # Break inner loop, restart outer loop via 'improved' flag logic?
                    # Actually, if we break inner, we continue outer.
                    # But indices might be off if we don't restart.
                    # Let's just break out of both loops and restart the while loop.
                    break
            
            if improved:
                # We made a change. Let's restart the local search loop to see if more swaps are possible.
                # This is a simple hill-climbing approach.
                break
        
    # Final check: Ensure distinct integers
    # The logic above ensures distinctness by checking `in current_set`.
    # But let's be safe.
    final_set = list(set(best_set))
    
    # Ensure types are int
    final_set = [int(x) for x in final_set]
    
    return final_set

# EVOLVE-BLOCK-END
