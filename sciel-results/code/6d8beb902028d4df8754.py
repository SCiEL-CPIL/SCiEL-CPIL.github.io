"""Rewritten program for bounded LCM clique problem."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    LIMIT = 512
    
    # Precompute LCM table using bit-level optimization
    # lcm(a,b) = a * b // gcd(a,b)
    # We need lcm(a,b) <= LIMIT for all pairs in our set
    
    n = LIMIT
    
    # Build compatibility: compat[x] = set of y where lcm(x,y) <= LIMIT
    # Use efficient precomputation
    compat = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            g = gcd(a, b)
            if a * b // g <= n:
                compat[a].add(b)
                if a != b:
                    compat[b].add(a)
    
    # For fast checking, also build a boolean matrix as list of bytearrays
    # Actually, let's use the sets directly but with a fast membership check
    
    def lcm_leq(a, b):
        """Check if lcm(a,b) <= LIMIT"""
        if a == b:
            return True
        return b in compat[a]
    
    def is_compatible_with_all(x, S):
        """Check if x is compatible with all elements in S (S is a set for O(1) lookup)"""
        for y in S:
            if not lcm_leq(x, y):
                return False
        return True
    
    def verify_clique(S):
        """Verify S is a valid clique"""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if not lcm_leq(S_list[i], S_list[j]):
                    return False
        return True
    
    best_S = set()
    
    # Strategy 1: Start with known good seeds and greedily expand
    # Key insight: divisors of highly composite numbers form good starting points
    
    # A number x is "universal" if it's compatible with many others
    # Numbers that divide 5040, 7560, etc. tend to be good hubs
    
    # Let's try: all numbers that divide some highly composite number <= 512
    # Actually, let's just try multiple seeds
    
    def greedy_expand(S, order):
        """Greedily add elements from order to S if compatible"""
        S = set(S)
        for x in order:
            if x not in S and is_compatible_with_all(x, S):
                S.add(x)
        return S
    
    def local_search(S, max_rounds=50):
        """Local search: try to add elements, or swap to improve"""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 10:
            improved = False
            rounds += 1
            
            # Phase 1: Try to add any compatible element
            for x in range(1, n + 1):
                if x not in S and is_compatible_with_all(x, S):
                    S.add(x)
                    improved = True
            
            # Phase 2: Try 1-for-2 swaps (remove one, add two)
            if not improved:
                S_list = list(S)
                for i in range(len(S_list)):
                    if time.time() > deadline - 10:
                        break
                    S_minus = S - {S_list[i]}
                    # Find all elements compatible with S_minus
                    candidates = [x for x in range(1, n + 1) 
                                  if x not in S_minus and is_compatible_with_all(x, S_minus)]
                    # Try pairs of candidates
                    for idx in range(len(candidates)):
                        for jdx in range(idx + 1, len(candidates)):
                            x1, x2 = candidates[idx], candidates[jdx]
                            if lcm_leq(x1, x2):
                                new_S = S_minus | {x1, x2}
                                if verify_clique(new_S) and len(new_S) > len(S):
                                    S = new_S
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # Phase 3: Try 2-for-3 swaps (remove two, add three)
            if not improved and len(S) > 4:
                S_list = list(S)
                for i in range(len(S_list)):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i + 1, len(S_list)):
                        S_minus = S - {S_list[i], S_list[j]}
                        candidates = [x for x in range(1, n + 1) 
                                      if x not in S_minus and is_compatible_with_all(x, S_minus)]
                        # Try triples of candidates
                        found = False
                        for idx in range(len(candidates)):
                            for jdx in range(idx + 1, len(candidates)):
                                for kdx in range(jdx + 1, len(candidates)):
                                    x1, x2, x3 = candidates[idx], candidates[jdx], candidates[kdx]
                                    if lcm_leq(x1, x2) and lcm_leq(x1, x3) and lcm_leq(x2, x3):
                                        new_S = S_minus | {x1, x2, x3}
                                        if verify_clique(new_S) and len(new_S) > len(S):
                                            S = new_S
                                            improved = True
                                            found = True
                                            break
                                if found:
                                    break
                            if found:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return S
    
    # Multiple seed strategies
    seeds = [
        # Small numbers
        list(range(1, 23)),
        # Numbers up to 30
        list(range(1, 31)),
        # Divisors of 5040 within range
        [x for x in range(1, 513) if 5040 % x == 0],
        # Divisors of 7560 within range  
        [x for x in range(1, 513) if 7560 % x == 0],
        # Divisors of 10080 within range
        [x for x in range(1, 513) if 10080 % x == 0],
        # Powers of 2
        [2**k for k in range(0, 10)],
        # Multiples of small numbers
        list(range(1, 17)) + [x for x in range(17, 513) if any(x % d == 0 for d in range(2, 11))],
    ]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        S = set(seed)
        # Filter seed to only keep valid clique
        S_list = list(S)
        valid_seed = set()
        for x in S_list:
            if is_compatible_with_all(x, valid_seed):
                valid_seed.add(x)
        S = valid_seed
        
        # Greedy expand in different orders
        S = greedy_expand(S, range(1, n + 1))
        S = local_search(S, max_rounds=30)
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Try more aggressive local search on best_S
    if time.time() < deadline - 15:
        best_S = local_search(best_S, max_rounds=50)
    
    # Try random restarts
    import random
    random.seed(123)
    if time.time() < deadline - 20:
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            
            # Random seed: pick random compatible subset
            seed_size = random.randint(8, 20)
            # Pick from numbers that are "central" (compatible with many)
            centrality = [len(compat[x]) for x in range(1, n + 1)]
            # Weighted random selection
            weights = [c**2 for c in centrality]
            seed = random.sample(range(1, n + 1), min(seed_size, n), counts=weights)
            
            S = set(seed)
            # Filter to valid clique
            S_list = list(S)
            valid_seed = set()
            for x in S_list:
                if is_compatible_with_all(x, valid_seed):
                    valid_seed.add(x)
            S = valid_seed
            
            S = greedy_expand(S, range(1, n + 1))
            S = local_search(S, max_rounds=20)
            
            if len(S) > len(best_S):
                best_S = set(S)
    
    # Final: try to add any remaining compatible elements
    for x in range(1, n + 1):
        if x not in best_S and is_compatible_with_all(x, best_S):
            best_S.add(x)
    
    # One more local search pass
    if time.time() < deadline - 5:
        best_S = local_search(best_S, max_rounds=20)
    
    # Final verification
    S_list = list(best_S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            if not lcm_leq(S_list[i], S_list[j]):
                # Remove the problematic element
                best_S.discard(S_list[j])
                break
    
    return list(best_S)
# EVOLVE-BLOCK-END
