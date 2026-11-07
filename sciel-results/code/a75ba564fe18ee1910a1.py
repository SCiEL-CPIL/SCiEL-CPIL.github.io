import time
import random
from math import gcd

def get_smallest_prime_factor(n):
    if n % 2 == 0:
        return 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for safety
    
    candidates = list(range(2, 257))
    
    # Precompute SPF for all candidates
    spf_map = {x: get_smallest_prime_factor(x) for x in candidates}
    
    # Fast can_add check
    def can_add(x, S):
        """Check if x can be added to set S without violating the triple constraint."""
        n = len(S)
        if n < 2:
            return True
        
        # For all distinct pairs (b,c) in S, check:
        # 1. x does not divide b*c
        # 2. b does not divide x*c  
        # 3. c does not divide x*b
        for i in range(n):
            b = S[i]
            for j in range(i + 1, n):
                c = S[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True
    
    # Local search: try to improve set by 1-for-2 swaps
    def local_search(S, all_candidates):
        """Try to improve S by removing one element and adding two others."""
        S_set = set(S)
        remaining = [x for x in all_candidates if x not in S_set]
        improved = True
        max_iterations = 30
        iteration = 0
        
        while improved and iteration < max_iterations:
            iteration += 1
            if time.time() > deadline:
                break
            improved = False
            
            # Try 1-for-2 swaps: remove one, add two
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                S_no_i = [S[j] for j in range(len(S)) if j != i]
                S_no_i_set = set(S_no_i)
                
                # Try all pairs (x, y) from remaining that can both be added
                for xi in range(len(remaining)):
                    if time.time() > deadline:
                        break
                    x = remaining[xi]
                    if x in S_no_i_set:
                        continue
                    if not can_add(x, S_no_i):
                        continue
                    
                    S_with_x = S_no_i + [x]
                    S_with_x_set = S_no_i_set | {x}
                    
                    for yi in range(len(remaining)):
                        if yi == xi:
                            continue
                        y = remaining[yi]
                        if y in S_with_x_set:
                            continue
                        if can_add(y, S_with_x):
                            # Found improvement
                            S = S_with_x + [y]
                            S_set = S_with_x_set | {y}
                            remaining = [z for z in all_candidates if z not in S_set]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return S
    
    best_set = []
    
    # Order 1: Value Descending
    order1 = sorted(candidates, reverse=True)
    S1 = []
    for x in order1:
        if time.time() > deadline:
            break
        if can_add(x, S1):
            S1.append(x)
    if len(S1) > len(best_set):
        best_set = S1
    
    # Order 2: SPF Descending (Primes and large-factor numbers first)
    order2 = sorted(candidates, key=lambda x: (-spf_map[x], -x))
    S2 = []
    for x in order2:
        if time.time() > deadline:
            break
        if can_add(x, S2):
            S2.append(x)
    if len(S2) > len(best_set):
        best_set = S2
    
    # Order 3: Smallest first
    order3 = sorted(candidates)
    S3 = []
    for x in order3:
        if time.time() > deadline:
            break
        if can_add(x, S3):
            S3.append(x)
    if len(S3) > len(best_set):
        best_set = S3
    
    # Apply local search to the best set found so far
    if time.time() < deadline - 5:
        improved_set = local_search(list(best_set), candidates)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Randomized restarts with local search
    attempts = 0
    while time.time() < deadline - 10 and attempts < 30:
        random.shuffle(candidates)
        Sr = []
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, Sr):
                Sr.append(x)
        
        # Apply local search to random result
        if time.time() < deadline - 5:
            Sr_improved = local_search(Sr, candidates)
            if len(Sr_improved) > len(Sr):
                Sr = Sr_improved
        
        if len(Sr) > len(best_set):
            best_set = Sr
        
        attempts += 1
    
    # Final local search on best set
    if time.time() < deadline - 2:
        final_improved = local_search(list(best_set), candidates)
        if len(final_improved) > len(best_set):
            best_set = final_improved
    
    # Ensure we return a list of Python ints
    return list(best_set)
