import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Try LCM of 5..20 first
    L = 1
    for i in range(5, 21):
        L = lcm(L, i)
        
    # Get divisors of L that are >= 5
    divisors = []
    for d in range(5, int(math.isqrt(L)) + 1):
        if L % d == 0:
            divisors.append(d)
            if d * d != L:
                divisors.append(L // d)
    divisors = [d for d in divisors if 5 <= d <= 10000000]
    
    # DP: 0/1 Knapsack
    # dp[sum] = min_terms
    dp = {0: 0}
    parent = {} # sum -> (prev_sum, d)
    
    for d in divisors:
        w = L // d
        # Iterate over a copy of dp to avoid using d multiple times
        items_to_update = []
        for s, terms in dp.items():
            new_s = s + w
            if new_s > L:
                continue
            new_terms = terms + 1
            if new_s not in dp or new_terms < dp[new_s]:
                items_to_update.append((new_s, new_terms, s, d))
        
        for new_s, new_terms, prev_s, used_d in items_to_update:
            dp[new_s] = new_terms
            parent[new_s] = (prev_s, used_d)
            
        if L in dp:
            # Reconstruct
            path = []
            cur_s = L
            while cur_s != 0:
                prev_s, used_d = parent[cur_s]
                path.append(used_d)
                cur_s = prev_s
            # Verify
            if sum(Fraction(1, d) for d in path) == 1:
                return path

    # If not found, try larger L
    L2 = 1
    for i in range(5, 31):
        L2 = lcm(L2, i)
    divisors2 = []
    for d in range(5, int(math.isqrt(L2)) + 1):
        if L2 % d == 0:
            divisors2.append(d)
            if d * d != L2:
                divisors2.append(L2 // d)
    divisors2 = [d for d in divisors2 if 5 <= d <= 10000000]
    
    dp2 = {0: 0}
    parent2 = {}
    for d in divisors2:
        w = L2 // d
        items_to_update = []
        for s, terms in dp2.items():
            new_s = s + w
            if new_s > L2:
                continue
            new_terms = terms + 1
            if new_s not in dp2 or new_terms < dp2[new_s]:
                items_to_update.append((new_s, new_terms, s, d))
        
        for new_s, new_terms, prev_s, used_d in items_to_update:
            dp2[new_s] = new_terms
            parent2[new_s] = (prev_s, used_d)
            
        if L2 in dp2:
            path = []
            cur_s = L2
            while cur_s != 0:
                prev_s, used_d = parent2[cur_s]
                path.append(used_d)
                cur_s = prev_s
            if sum(Fraction(1, d) for d in path) == 1:
                return path

    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
