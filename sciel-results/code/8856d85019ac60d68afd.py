"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)

    # Strategy:
    # 1. Initial best: Multiples of 210 (score 4: {2,3,5,7})
    best = [210 * i for i in range(1, 13)]
    best_score = evaluate(best)
    
    # 2. Graph Clique Search for smaller prime sets P
    # We look for a clique of size 12 in the graph G_P where edge (u,v) exists
    # if u+v is P-smooth.
    
    # Candidate prime sets to try, sorted by size
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 5, 13},
        {2, 3, 7, 11},
    ]

    for P in prime_sets_to_try:
        if len(P) >= best_score:
            continue
            
        if time.time() > deadline:
            break

        # Precompute valid sums for this P
        # A sum S is valid if all prime factors of S are in P
        valid_sums = set()
        for s in range(2, MAX_VAL + 1):
            # Check if s is P-smooth
            temp = s
            is_smooth = True
            while temp > 1:
                p = spf[temp]
                if p not in P:
                    is_smooth = False
                    break
                while temp % p == 0:
                    temp //= p
            if is_smooth:
                valid_sums.add(s)
        
        # Heuristic search for a clique of size 12
        # We restrict the search space to numbers 1..5000 to keep it fast, 
        # but we can expand if needed. 
        # Actually, if we use multiples of the product of P, sums are automatically P-smooth?
        # No, if x = k*M, y = l*M, x+y = (k+l)M. If M is product of P, then x+y is divisible by M, 
        # so its prime factors are a subset of P. 
        # So if we pick 12 multiples of M = prod(P), the score is <= |P|.
        # This is a trivial solution. 
        # M = 2*3*5*7 = 210. 12 multiples: 210, 420, ..., 2520. All <= 10000.
        # So for P={2,3,5,7}, we always have a solution with score 4.
        # Can we do better? 
        # If P={2,3,5}, M=30. 12 multiples: 30..360. Sums are multiples of 30. 
        # Prime factors of sums are subset of {2,3,5}. Score 3.
        # If P={2,3}, M=6. 12 multiples: 6..72. Sums are multiples of 6. 
        # Prime factors of sums are subset of {2,3}. Score 2.
        # If P={2}, M=2. 12 multiples: 2..24. Sums are even. 
        # Prime factors of sums are subset of {2}. Score 1.
        # Wait, is this true?
        # If x=2, y=4, x+y=6=2*3. 3 is not in {2}. 
        # So multiples of 2 do NOT guarantee P-smoothness for P={2}.
        # The sum of two even numbers is even, but may have other prime factors.
        # So the trivial "multiples of M" only works if we ensure the quotient (k+l) is also P-smooth?
        # No, x+y = M(k+l). The prime factors of x+y are PrimeFactors(M) U PrimeFactors(k+l).
        # So we need PrimeFactors(k+l) subset of P.
        # If we choose k, l such that k+l is P-smooth, then x+y is P-smooth.
        # This reduces to finding 12 integers k_1..k_12 such that all k_i+k_j are P-smooth.
        # Then x_i = M * k_i.
        # This is the same problem on smaller numbers!
        
        # So the graph clique approach is the correct generalization.
        
        # Let's run the clique search on a limited range to find if a better P exists.
        # If we find a clique of size 12 for P, we return it.
        
        SEARCH_RANGE = 200
        # Build adjacency list
        adj = [set() for _ in range(SEARCH_RANGE + 1)]
        for x in range(1, SEARCH_RANGE + 1):
            for y in range(x+1, SEARCH_RANGE + 1):
                if (x + y) in valid_sums:
                    adj[x].add(y)
                    adj[y].add(x)
        
        # Find max clique of size 12
        # Greedy with backtracking
        def find_clique():
            # Order vertices by degree
            vertices = list(range(1, SEARCH_RANGE + 1))
            vertices.sort(key=lambda v: len(adj[v]), reverse=True)
            
            # Try to build a clique starting from each vertex
            for start_v in vertices:
                if len(adj[start_v]) < 11:
                    break
                clique = [start_v]
                candidates = adj[start_v]
                
                def backtrack(clique, candidates):
                    if len(clique) == 12:
                        return list(clique)
                    if time.time() > deadline:
                        return None
                    
                    # Filter candidates to those connected to all in clique
                    # Since we intersect at each step, candidates are already connected to all in clique
                    # We just need to pick one
                    if not candidates:
                        return None
                    
                    # Heuristic: pick candidate with highest degree within candidates
                    # deg_in_cands(v) = len(adj[v] & candidates)
                    # This is expensive to compute for all.
                    # Just iterate in sorted order?
                    
                    # To speed up, we can just iterate through candidates
                    # and try to extend.
                    for v in list(candidates):
                        new_candidates = candidates & adj[v]
                        new_candidates.discard(v)
                        
                        if len(new_candidates) < 12 - (len(clique) + 1):
                            continue
                        
                        res = backtrack(clique + [v], new_candidates)
                        if res:
                            return res
                    return None
                
                res = backtrack(clique, candidates)
                if res:
                    return res
            return None
        
        clique = find_clique()
        if clique:
            # Verify
            score = evaluate(clique)
            if score < best_score:
                best = clique
                best_score = score
                if best_score == 1:
                    return best

    return best

# EVOLVE-BLOCK-END
