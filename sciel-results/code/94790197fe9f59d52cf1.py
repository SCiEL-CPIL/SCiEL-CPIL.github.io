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
    deadline = start_time + 85  # 5s margin
    
    candidates = list(range(2, 257))
    
    # Precompute SPF for all candidates
    spf_map = {x: get_smallest_prime_factor(x) for x in candidates}
    
    best_set = []
    
    # Define the check function
    def can_add(x, S):
        # S is a list of ints
        # Check 1: x does not divide b*c for any distinct b,c in S
        n = len(S)
        for i in range(n):
            b = S[i]
            for j in range(i + 1, n):
                c = S[j]
                if (b * c) % x == 0:
                    return False
        
        # Check 2: No existing b divides x*c for any c in S \ {b}
        # The triple is (b, x, c) or (b, c, x).
        # If a=b, condition is b | x*c.
        # If a=x, condition is x | b*c (checked above).
        # If a=c, condition is c | b*x.
        # So we need to check if any b in S divides x*c for any c in S (c != b).
        # And if any c in S divides b*x for any b in S (b != c).
        # Due to symmetry, checking b | x*c for all distinct b,c in S covers both.
        
        for b in S:
            for c in S:
                if b == c:
                    continue
                if (x * c) % b == 0:
                    return False
        return True

    # Helper to run greedy with a specific order
    def run_greedy(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        return S

    # Order 1: Value Descending
    order1 = sorted(candidates, reverse=True)
    set1 = run_greedy(order1)
    if len(set1) > len(best_set):
        best_set = set1

    # Order 2: SPF Descending (Primes and large-factor numbers first)
    order2 = sorted(candidates, key=lambda x: (-spf_map[x], -x))
    set2 = run_greedy(order2)
    if len(set2) > len(best_set):
        best_set = set2

    # Order 3: Randomized restarts
    # If we have time, try a few random permutations
    attempts = 0
    while time.time() < deadline - 5:
        random.shuffle(candidates)
        set_r = run_greedy(candidates)
        if len(set_r) > len(best_set):
            best_set = set_r
        attempts += 1
        if attempts > 50: # Cap random attempts to ensure we don't loop forever if can_add is slow
            break

    return best_set
