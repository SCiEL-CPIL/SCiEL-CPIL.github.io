import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s

    n = 12
    max_val = 10000
    max_sum = 2 * max_val

    # Precompute smallest prime factor (SPF) table up to 2*max_val = 20000
    spf = list(range(max_sum + 1))
    for i in range(2, int(max_sum**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute prime factor sets for all sums 2..20000
    pf_cache = [None] * (max_sum + 1)
    pf_cache[0] = frozenset()
    pf_cache[1] = frozenset()
    for x in range(2, max_sum + 1):
        factors = set()
        y = x
        while y > 1:
            p = spf[y]
            factors.add(p)
            while y % p == 0:
                y //= p
        pf_cache[x] = frozenset(factors)

    def score_set(arr):
        """Count distinct prime factors across all pairwise sums."""
        all_primes = set()
        for i in range(n):
            ai = arr[i]
            for j in range(i+1, n):
                all_primes.update(pf_cache[ai + arr[j]])
        return len(all_primes)

    # Small prime sets to try as "allowed" primes
    # Strategy: find 12 distinct ints <= 10000 such that all pairwise sums
    # have prime factors only from a small set S.
    # If such a set exists, the answer is |S| (or a subset thereof).
    
    # Candidate small prime sets (ordered by size, then by "usefulness")
    candidate_prime_sets = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {2, 11},
        {2, 3, 5},
        {2, 3, 7},
        {2, 3, 11},
        {2, 5, 7},
        {2, 5, 11},
        {2, 7, 11},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 7, 11},
        {2, 3, 5, 13},
        {2, 3, 7, 13},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 13},
        {2, 3, 5, 7, 17},
        {2, 3, 5, 7, 19},
        {2, 3, 5, 7, 23},
        {2, 3, 5, 7, 29},
        {2, 3, 5, 7, 31},
        {2, 3, 5, 7, 37},
        {2, 3, 5, 7, 41},
        {2, 3, 5, 7, 43},
    ]

    best_score = float('inf')
    best_arr = None

    random.seed(42)

    # For each candidate prime set S, greedily select 12 distinct integers
    # such that for every pair (a, b) in the selection, all prime factors of a+b are in S.
    # This is a "conflict graph" where edge (a,b) exists if a+b has a prime factor outside S.
    # We want an independent set of size 12.
    
    for S in candidate_prime_sets:
        if time.time() > deadline:
            break
        
        # Precompute: for each value v in 1..max_val, which values u (1..max_val) are "compatible"
        # i.e., all prime factors of v+u are in S.
        # Instead of full matrix, we'll do greedy selection.
        
        # Greedy: start with a pool of candidate values, iteratively pick one that is compatible
        # with all already-picked values.
        
        # To make this tractable, we'll try multiple random orderings and greedy picks.
        # Also try structured initializations.
        
        # First, try structured: multiples of lcm of S
        # If S = {2,3}, then sums should only have factors 2,3. 
        # If all numbers are multiples of 6, sums are multiples of 6, so factors are subset of {2,3}.
        # Actually 6a+6b = 6(a+b), and a+b can have any prime factor. So that doesn't work.
        
        # Better: if all numbers are of the form k*p where p is chosen so that sums stay in S.
        # For S={2}: all numbers must be powers of 2 times... no, sum of two powers of 2 is not necessarily power of 2.
        # 2^a + 2^b = 2^min(a,b)*(1+2^|a-b|), which has odd factor unless |a-b|=0 (not distinct) or...
        # Actually 2+4=6=2*3, introduces 3. So S={2} is hard.
        
        # For S={2,3}: numbers of form 2^a * 3^b. Sums of two such may introduce other primes.
        # 2+3=5, introduces 5. So even {2,3} is tricky.
        
        # Let's just do the greedy conflict-graph approach.
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Create a shuffled list of candidate values
            # Focus on smaller values first (more likely to have "nice" sums)
            # Mix: some small values, some random
            if trial < 20:
                # Small values: 1..500
                pool = list(range(1, 501))
            elif trial < 40:
                # Multiples of small numbers
                base = random.choice([2, 3, 4, 5, 6, 10, 12, 15, 20, 30])
                pool = list(range(base, max_val+1, base))
            else:
                pool = list(range(1, max_val+1))
            
            random.shuffle(pool)
            
            # Greedy selection
            selected = []
            for v in pool:
                if len(selected) == n:
                    break
                # Check if v is compatible with all selected
                compatible = True
                for s in selected:
                    if v == s:
                        continue
                    sum_val = v + s
                    if sum_val > max_sum:
                        compatible = False
                        break
                    factors = pf_cache[sum_val]
                    if not factors.issubset(S):
                        compatible = False
                        break
                if compatible:
                    selected.append(v)
            
            if len(selected) == n:
                s = score_set(selected)
                if s < best_score:
                    best_score = s
                    best_arr = list(selected)
                if best_score == len(S):
                    # Can't do better than |S| for this S, and smaller S would have been tried
                    break
        
        if best_score <= len(S):
            # Found a solution with score <= len(S), no need to try larger S
            break

    # If we found something, do local search to potentially improve
    if best_arr is not None:
        arr = list(best_arr)
        current_score = best_score
        
        # Local search: try replacing elements to reduce prime count
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                old_val = arr[i]
                # Try a few random replacements
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    new_val = random.randint(1, max_val)
                    if new_val in arr and new_val != old_val:
                        continue
                    arr[i] = new_val
                    new_score = score_set(arr)
                    if new_score < current_score:
                        current_score = new_score
                        no_improve = 0
                        improved = True
                    else:
                        arr[i] = old_val
                        no_improve += 1
                if time.time() > deadline - 5:
                    break
            
            if not improved and no_improve >= max_no_improve:
                break
        
        best_arr = arr
        best_score = current_score
    else:
        # Fallback: use multiples of 6
        best_arr = [6 * i for i in range(1, n+1)]
        best_score = score_set(best_arr)

    return sorted(best_arr)
