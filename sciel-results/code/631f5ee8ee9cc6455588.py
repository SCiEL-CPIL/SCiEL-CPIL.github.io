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
    deadline = start_time + 85  # Leave 5s margin for safety
    
    candidates = list(range(2, 257))
    cand_set = set(candidates)
    
    # Precompute SPF for all candidates
    spf_map = {x: get_smallest_prime_factor(x) for x in candidates}
    
    # Precompute all pair products for fast divisibility checks
    # For a given x, we need to check: does x divide b*c for any pair (b,c) in S?
    # And: does b divide x*c? And: does c divide x*b?
    
    def can_add(x, S):
        """Check if x can be added to set S without violating the triple constraint."""
        n = len(S)
        if n < 2:
            return True
        
        # Check: x does not divide b*c for any distinct pair (b,c) in S
        # Check: b does not divide x*c for any distinct pair (b,c) in S  
        # Check: c does not divide x*b for any distinct pair (b,c) in S
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
    
    def can_add_fast(x, S, S_set):
        """Optimized can_add with early exit on most likely conflicts first."""
        n = len(S)
        if n < 2:
            return True
        
        # Sort S by likelihood of causing conflict with x
        # Numbers that share factors with x are more likely to cause issues
        # Check pairs where b or c shares a factor with x first
        spf_x = spf_map[x]
        
        # First pass: check pairs involving numbers that share spf with x
        shared = [v for v in S if spf_map[v] == spf_x or gcd(v, x) > 1]
        other = [v for v in S if v not in shared]
        
        # Check pairs within shared first (most likely to conflict)
        for i in range(len(shared)):
            b = shared[i]
            for j in range(i + 1, len(shared)):
                c = shared[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        
        # Check pairs between shared and other
        for b in shared:
            for c in other:
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        
        # Check pairs within other
        for i in range(len(other)):
            b = other[i]
            for j in range(i + 1, len(other)):
                c = other[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        
        return True
    
    def greedy_build(order, time_limit):
        """Build a set greedily following the given order."""
        S = []
        S_set = set()
        for x in order:
            if time.time() > time_limit:
                break
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
        return S
    
    def remove_and_refill(S, all_candidates, time_limit):
        """Remove one element from S, then greedily add back the best candidates."""
        best = S
        best_len = len(S)
        
        for i in range(len(S)):
            if time.time() > time_limit:
                break
            S_no_i = [S[j] for j in range(len(S)) if j != i]
            S_no_i_set = set(S_no_i)
            
            # Try to refill with candidates not in S_no_i, in order of value descending
            candidates_not_in = [x for x in all_candidates if x not in S_no_i_set]
            candidates_not_in.sort(reverse=True)
            
            S_new = list(S_no_i)
            S_new_set = set(S_no_i)
            
            for x in candidates_not_in:
                if time.time() > time_limit:
                    break
                if len(S_new) >= best_len + 2:
                    break
                if can_add(x, S_new):
                    S_new.append(x)
                    S_new_set.add(x)
            
            if len(S_new) > best_len:
                best = S_new
                best_len = len(S_new)
        
        return best
    
    def local_search(S, all_candidates, time_limit):
        """Try to improve S by various swap operations."""
        S = list(S)
        improved = True
        max_iterations = 20
        iteration = 0
        
        while improved and iteration < max_iterations:
            iteration += 1
            if time.time() > time_limit:
                break
            improved = False
            
            # Try 1-for-2 swaps: remove one, add two
            for i in range(len(S)):
                if time.time() > time_limit:
                    break
                S_no_i = [S[j] for j in range(len(S)) if j != i]
                S_no_i_set = set(S_no_i)
                
                remaining = [x for x in all_candidates if x not in S_no_i_set]
                remaining.sort(reverse=True)
                
                for x in remaining:
                    if time.time() > time_limit:
                        break
                    if not can_add(x, S_no_i):
                        continue
                    
                    S_with_x = S_no_i + [x]
                    S_with_x_set = S_no_i_set | {x}
                    
                    remaining2 = [y for y in remaining if y != x and y not in S_with_x_set]
                    
                    for y in remaining2:
                        if time.time() > time_limit:
                            break
                        if can_add(y, S_with_x):
                            S = S_with_x + [y]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Try remove-and-refill
            if not improved and time.time() < time_limit - 2:
                S2 = remove_and_refill(S, all_candidates, time_limit)
                if len(S2) > len(S):
                    S = S2
                    improved = True
        
        return S
    
    best_set = []
    
    # Multiple greedy orderings
    # Order 1: Value Descending
    S1 = greedy_build(sorted(candidates, reverse=True), deadline - 60)
    if len(S1) > len(best_set):
        best_set = list(S1)
    
    # Order 2: SPF Descending (Primes and large-factor numbers first)
    S2 = greedy_build(sorted(candidates, key=lambda x: (-spf_map[x], -x)), deadline - 55)
    if len(S2) > len(best_set):
        best_set = list(S2)
    
    # Order 3: Smallest first
    S3 = greedy_build(sorted(candidates), deadline - 50)
    if len(S3) > len(best_set):
        best_set = list(S3)
    
    # Order 4: Primes first, then composites by SPF
    primes = [x for x in candidates if spf_map[x] == x]
    composites = [x for x in candidates if spf_map[x] != x]
    order4 = sorted(primes, reverse=True) + sorted(composites, key=lambda x: (-spf_map[x], -x))
    S4 = greedy_build(order4, deadline - 45)
    if len(S4) > len(best_set):
        best_set = list(S4)
    
    # Apply local search to the best set found so far
    if time.time() < deadline - 15:
        improved_set = local_search(list(best_set), candidates, deadline - 10)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Randomized restarts with local search
    attempts = 0
    while time.time() < deadline - 12 and attempts < 50:
        random.shuffle(candidates)
        Sr = greedy_build(candidates, deadline - 12)
        
        # Apply local search to random result
        if time.time() < deadline - 8:
            Sr_improved = local_search(Sr, candidates, deadline - 5)
            if len(Sr_improved) > len(Sr):
                Sr = Sr_improved
        
        if len(Sr) > len(best_set):
            best_set = Sr
        
        attempts += 1
    
    # Final local search on best set
    if time.time() < deadline - 3:
        final_improved = local_search(list(best_set), candidates, deadline - 1)
        if len(final_improved) > len(best_set):
            best_set = final_improved
    
    # Ensure we return a list of Python ints
    return list(best_set)
