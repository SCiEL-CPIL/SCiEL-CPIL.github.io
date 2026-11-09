import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(2, n + 1):
        for j in range(2, n + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Precompute: for each a and b, d = a // gcd(a, b)
    # And for each d, the list of multiples of d in [2, 256]
    multiples_of = {}
    for d in range(1, n + 1):
        multiples_of[d] = [k for k in range(d, n + 1, d) if k >= 2]
    
    # Precompute is_prime
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    def is_valid(S):
        S_list = sorted(S)
        S_set = set(S_list)
        m = len(S_list)
        if m < 3:
            return True
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                d = a // gcd_table[a][b]
                if d == 1:
                    # Any c distinct from a, b works. Since m >= 3, there is such c.
                    return False
                # Need c in S_list, c != a, c != b, such that d | c.
                found = False
                for k in multiples_of[d]:
                    if k in S_set and k != a and k != b:
                        found = True
                        break
                if found:
                    return False
        return True
    
    def try_add(c, S, S_set):
        # Check if S U {c} is valid.
        # Only need to check triples involving c.
        S_list = list(S)
        m = len(S_list)
        if m < 2:
            return True
        
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                # Check a | c*b
                d1 = a // gcd_table[a][c]
                if d1 == 1:
                    return False
                if b % d1 == 0:
                    return False
                # Check b | c*a
                d2 = b // gcd_table[b][c]
                if d2 == 1:
                    return False
                if a % d2 == 0:
                    return False
                # Check c | a*b
                d3 = c // gcd_table[c][a]
                if d3 == 1:
                    return False
                if b % d3 == 0:
                    return False
                # Also check c | a*b with the other factor
                d4 = c // gcd_table[c][b]
                if d4 == 1:
                    return False
                if a % d4 == 0:
                    return False
        return True
    
    def try_swap_remove_add(c, a_to_remove, S, S_set):
        # Check if (S - {a_to_remove}) U {c} is valid
        new_S_set = S_set - {a_to_remove}
        new_S_set.add(c)
        new_S_list = [x for x in S if x != a_to_remove]
        new_S_list.append(c)
        return is_valid(new_S_list)
    
    best_S = None
    best_size = 0
    
    # Multiple restarts with different initial sets
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            # Start with all primes
            S = [i for i in range(2, n + 1) if is_prime[i]]
        elif restart == 1:
            # Start with all numbers > 128 (large numbers are less likely to divide products)
            S = [i for i in range(129, n + 1)]
        elif restart == 2:
            # Start with primes and some large composites
            S = [i for i in range(2, n + 1) if is_prime[i] or i > 200]
        elif restart == 3:
            # Start with squares and primes
            S = [i for i in range(2, n + 1) if is_prime[i] or int(math.isqrt(i))**2 == i]
        else:
            # Random initial set
            S = random.sample(candidates, min(30, len(candidates)))
        
        S_set = set(S)
        current_size = len(S)
        
        if current_size > best_size and is_valid(S):
            best_size = current_size
            best_S = S[:]
        
        # Local search: try to add elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding each non-member
            for c in candidates:
                if c in S_set:
                    continue
                if time.time() > deadline - 5:
                    break
                
                if try_add(c, S, S_set):
                    S.append(c)
                    S_set.add(c)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S[:]
                    improved = True
                    continue
                
                # Try swaps: remove a, add c
                # Heuristic: try removing elements that share factors with c
                tried = set()
                
                # First try removing prime factors of c
                temp_c = c
                factors = []
                for p in range(2, int(temp_c**0.5) + 1):
                    if temp_c % p == 0:
                        factors.append(p)
                        while temp_c % p == 0:
                            temp_c //= p
                if temp_c > 1:
                    factors.append(temp_c)
                
                for a in factors:
                    if a in S_set and a not in tried:
                        tried.add(a)
                        if try_swap_remove_add(c, a, S, S_set):
                            S.remove(a)
                            S_set.remove(a)
                            S.append(c)
                            S_set.add(c)
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S[:]
                            improved = True
                            break
                
                # If no factor swap worked, try removing random elements
                if not improved and len(S) > 0:
                    # Try a few random removals
                    num_tries = min(5, len(S))
                    for _ in range(num_tries):
                        if time.time() > deadline - 5:
                            break
                        a = random.choice(S)
                        if a not in tried:
                            tried.add(a)
                            if try_swap_remove_add(c, a, S, S_set):
                                S.remove(a)
                                S_set.remove(a)
                                S.append(c)
                                S_set.add(c)
                                if current_size > best_size:
                                    best_size = current_size
                                    best_S = S[:]
                                improved = True
                                break
        
        # After adding, try to improve by swapping pairs
        if time.time() < deadline - 5:
            swap_improved = True
            while swap_improved and time.time() < deadline - 5:
                swap_improved = False
                S_list = list(S)
                for i in range(len(S_list)):
                    if time.time() > deadline - 5:
                        break
                    a = S_list[i]
                    for j in range(i + 1, len(S_list)):
                        if time.time() > deadline - 5:
                            break
                        b = S_list[j]
                        # Try to find c not in S such that (S - {a, b}) U {c1, c2} is valid
                        # This is complex; skip for now
                        pass
    
    # Final validation and return
    if best_S is not None and is_valid(best_S):
        return list(best_S)
    
    # Fallback: return primes
    S = [i for i in range(2, n + 1) if is_prime[i]]
    return list(S)

# EVOLVE-BLOCK-END
