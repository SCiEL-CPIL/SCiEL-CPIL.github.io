"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    LIMIT = 512
    
    # Precompute lcm for all pairs
    # lcm(a,b) = a * b // gcd(a,b)
    # We need lcm(a,b) <= LIMIT for all pairs in our set
    
    # Precompute compatibility matrix
    # compat[i][j] = True if lcm(i+1, j+1) <= LIMIT
    n = LIMIT
    # Use a list of sets for efficiency
    # For each number x in 1..512, store the set of numbers y in 1..512 such that lcm(x,y) <= 512
    
    # Actually, let's think about this differently.
    # Two numbers a, b have lcm(a,b) <= 512 if and only if...
    # lcm(a,b) = a*b/gcd(a,b) <= 512
    
    # Key insight: if we have a number a in our set, then for any b in our set,
    # a*b/gcd(a,b) <= 512, which means b <= 512 * gcd(a,b) / a
    
    # Precompute for each x, the list of y such that lcm(x,y) <= 512
    # This is O(n^2) = 512^2 = 262144, which is fast
    
    # compat[x] = set of y (1-indexed, so x from 1 to 512)
    compat = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a * b // gcd(a, b) <= n:
                compat[a].add(b)
    
    # Now we need to find the largest subset S of {1,...,512} such that for all a,b in S, b in compat[a]
    # This is equivalent to finding a clique in the graph where edge (a,b) exists iff lcm(a,b) <= 512
    # But that's NP-hard in general. However, n=512 is small enough for heuristic approaches.
    
    # Strategy: Multi-strategy search with local search
    # 1. Start with different seeds
    # 2. Greedy add
    # 3. Local search (1-for-1 and 2-for-3 swaps)
    
    def is_compatible_with_all(x, S):
        """Check if x is compatible with all elements in S."""
        for y in S:
            if x * y // gcd(x, y) > n:
                return False
        return True
    
    def greedy_add(S, candidates):
        """Try to add candidates to S in order, adding if compatible."""
        S = list(S)
        for x in candidates:
            if x not in S and is_compatible_with_all(x, S):
                S.append(x)
        return S
    
    def local_search_1for1(S, max_iters=1000):
        """Try to improve S by removing one element and adding another."""
        S = list(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(S)):
                # Try removing S[i]
                S_without = S[:i] + S[i+1:]
                # Try adding each element not in S_without
                for x in range(1, n + 1):
                    if x not in S_without and is_compatible_with_all(x, S_without):
                        # Check if this improves (it always does since we're adding one)
                        # But we need to make sure the new set is valid
                        new_S = S_without + [x]
                        # Verify validity
                        valid = True
                        for a in new_S:
                            for b in new_S:
                                if a != b and a * b // gcd(a, b) > n:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            S = new_S
                            improved = True
                            break
                if improved:
                    break
        return S
    
    def local_search_2for3(S, max_iters=500):
        """Try to improve S by removing two elements and adding three."""
        S = list(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    S_without = S[:i] + S[i+1:j] + S[j+1:]
                    # Try adding pairs of elements
                    for x in range(1, n + 1):
                        if x in S_without:
                            continue
                        if not is_compatible_with_all(x, S_without):
                            continue
                        S_with_x = S_without + [x]
                        for y in range(1, n + 1):
                            if y in S_with_x or y == x:
                                continue
                            if not is_compatible_with_all(y, S_with_x):
                                continue
                            new_S = S_with_x + [y]
                            # Verify validity
                            valid = True
                            for a in new_S:
                                for b in new_S:
                                    if a != b and a * b // gcd(a, b) > n:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                S = new_S
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return S
    
    # Strategy 1: Start with small numbers (1 to k) and greedily add
    best_S = []
    
    # Try multiple seed strategies
    seeds = [
        list(range(1, 23)),  # 1 to 22
        list(range(1, 16)),  # 1 to 15
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256, 288, 320, 384, 448],  # powers and multiples
    ]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        S = list(seed)
        
        # Greedy add remaining numbers
        for x in range(1, n + 1):
            if time.time() > deadline - 10:
                break
            if x not in S and is_compatible_with_all(x, S):
                S.append(x)
        
        # Local search 1-for-1
        S = local_search_1for1(S)
        
        # Local search 2-for-3
        if time.time() < deadline - 10:
            S = local_search_2for3(S)
        
        if len(S) > len(best_S):
            best_S = S
    
    # Try random restarts with time remaining
    if time.time() < deadline - 15:
        import random
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random seed: pick random subset of small numbers
            seed_size = random.randint(5, 15)
            seed = random.sample(range(1, 50), seed_size)
            S = list(seed)
            
            # Greedy add
            for x in range(1, n + 1):
                if time.time() > deadline - 10:
                    break
                if x not in S and is_compatible_with_all(x, S):
                    S.append(x)
            
            # Local search
            S = local_search_1for1(S)
            
            if time.time() < deadline - 10:
                S = local_search_2for3(S)
            
            if len(S) > len(best_S):
                best_S = S
    
    # Final verification
    for a in best_S:
        for b in best_S:
            if a != b and a * b // gcd(a, b) > n:
                # Should not happen, but just in case
                pass
    
    return best_S
# EVOLVE-BLOCK-END
