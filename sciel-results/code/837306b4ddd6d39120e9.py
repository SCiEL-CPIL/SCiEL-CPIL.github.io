import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0 # Safety margin
    
    MAX_VAL = 10000
    N = 12
    LIMIT = 2 * MAX_VAL
    
    # 1. SPF Table
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def is_p_smooth(n, prime_set):
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p not in prime_set:
                return False
            while n % p == 0:
                n //= p
        return True

    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    best_result = None
    best_score = float('inf')
    
    # Try increasing sets of primes
    for k in range(2, len(primes) + 1):
        if time.time() > deadline:
            break
            
        current_primes = set(primes[:k])
        
        # Precompute neighbors for this prime set
        # neighbors[x] = set of y such that x+y is smooth
        # This is O(MAX_VAL * avg_smooth_count)
        # Smooth count for {2,3} is low, for {2,3,5} is higher.
        
        # Optimization: Only compute neighbors for numbers we might visit?
        # No, we need global view for greedy intersection.
        # But 10000 * 10000 is too big.
        # We compute neighbors on the fly or store as lists.
        
        # Let's store as lists of valid y's.
        neighbors = [None] * (MAX_VAL + 1)
        for x in range(1, MAX_VAL + 1):
            # y must be in [1, MAX_VAL]
            # x+y in [x+1, x+MAX_VAL]
            valid_ys = []
            # Iterate through possible sums? No, iterate through y?
            # Better: Iterate through smooth numbers s in [x+1, x+MAX_VAL]
            # y = s - x
            
            # We need a list of smooth numbers in range.
            # Let's precompute a boolean array is_smooth[s]
            
        # This approach is O(MAX_VAL^2) if done naively.
        # Let's precompute is_smooth for the current prime set.
        is_smooth = [False] * (LIMIT + 1)
        for s in range(2, LIMIT + 1):
            if is_p_smooth(s, current_primes):
                is_smooth[s] = True
        
        # Now build neighbors
        # For each x, neighbors are y such that is_smooth[x+y] is True
        # This is still O(MAX_VAL^2) if we loop y for each x.
        # But we can loop s (smooth numbers) and add to neighbors?
        # No, s = x + y. For a fixed s, pairs (x, s-x).
        
        # Let's use a different strategy for Clique Finding:
        # Randomized Greedy with Backtracking?
        # Or just Randomized Greedy with many restarts.
        
        # To make it fast, we can restrict the candidate pool.
        # Numbers that are "smooth" themselves? No, sums must be smooth.
        # However, if x is small, x+y is small, more likely to be smooth?
        
        # Let's try a specific heuristic:
        # 1. Generate a pool of candidates.
        #    Candidates should be numbers such that they form smooth sums with many other candidates.
        #    A good proxy: Numbers that are themselves "smooth" or close to smooth?
        #    Actually, if we pick all $a_i$ to be multiples of $M = \prod_{p \in S} p$,
        #    then $a_i + a_j$ is a multiple of $M$.
        #    If $M$ is small, $a_i + a_j$ has $S$ as factors, but might have others.
        #    Example: S={2,3}. M=6.
        #    a_i = 6*i. Sum = 6(i+j).
        #    Prime factors of 6(i+j) are {2,3} U factors(i+j).
        #    To ensure only {2,3}, i+j must be {2,3}-smooth.
        #    So the problem reduces to finding 12 distinct integers $i_1...i_{12}$
        #    such that $i_k + i_l$ is {2,3}-smooth.
        #    And $6*i_k \le 10000 \implies i_k \le 1666$.
        #    This is a smaller instance!
        
        # Let's apply this reduction.
        # For a prime set S, let M = product(S).
        # We seek $a_i = M * b_i$.
        # Condition: $M(b_i + b_j)$ is S-smooth.
        # Since M is S-smooth, $b_i + b_j$ MUST be S-smooth.
        # So we need a clique of size 12 in the graph of $b \in [1, \lfloor 10000/M \rfloor]$
        # where edge exists if $b_u + b_v$ is S-smooth.
        
        # This reduces the search space significantly for small S.
        
        M = 1
        for p in current_primes:
            M *= p
            
        max_b = MAX_VAL // M
        if max_b < N:
            continue # Can't fit N distinct multiples
            
        # Build graph for b in [1, max_b]
        # is_smooth_b_sum[s] for s in [2, 2*max_b]
        is_smooth_small = [False] * (2 * max_b + 1)
        for s in range(2, 2 * max_b + 1):
            if is_p_smooth(s, current_primes):
                is_smooth_small[s] = True
                
        # Now find clique of size N in [1..max_b]
        # max_b is small if M is large.
        # For S={2,3}, M=6, max_b=1666.
        # For S={2,3,5}, M=30, max_b=333.
        
        # Greedy Clique on [1..max_b]
        found_clique = None
        random.seed(42)
        
        # Precompute neighbors for b
        # neighbors_b[b] = list of c != b such that b+c is smooth
        neighbors_b = [set() for _ in range(max_b + 1)]
        for b in range(1, max_b + 1):
            for c in range(1, max_b + 1):
                if b == c:
                    continue
                if b + c <= 2 * max_b and is_smooth_small[b + c]:
                    neighbors_b[b].add(c)
                    
        # Try to find clique
        # Use a simple randomized greedy
        for _ in range(1000): # Restart limit
            if time.time() > deadline:
                break
                
            # Pick random start
            start = random.randint(1, max_b)
            clique = [start]
            # Candidates: neighbors of start
            candidates = set(neighbors_b[start])
            
            for _ in range(N - 1):
                if not candidates:
                    break
                    
                # Pick a node from candidates
                # Prefer nodes with high degree within candidates?
                # For simplicity, random pick
                next_node = random.choice(list(candidates))
                clique.append(next_node)
                
                # Update candidates: must be neighbor to all in clique
                # Specifically, must be in neighbors of next_node AND current candidates
                candidates &= neighbors_b[next_node]
                
                # Also remove nodes already in clique?
                # They are not in neighbors (no self-loops), so they are removed.
                
            if len(clique) == N:
                found_clique = clique
                break
                
        if found_clique:
            # Convert back to a_i
            result = [M * b for b in found_clique]
            # Verify
            valid = True
            primes_used = set()
            for i in range(N):
                for j in range(i+1, N):
                    s = result[i] + result[j]
                    pf = get_prime_factors(s)
                    if not pf.issubset(current_primes):
                        valid = False
                        break
                    primes_used |= pf
                if not valid:
                    break
            
            if valid:
                score = len(primes_used)
                if score < best_score:
                    best_score = score
                    best_result = result
                # We found a solution for this S. 
                # Can we do better? Only with smaller S, which we already tried.
                # So we can return immediately?
                # Yes, because we iterate S in increasing size.
                # Any solution with larger S will have score >= current score (likely).
                # Wait, if S={2,3} fails, we try {2,3,5}.
                # If {2,3,5} succeeds, score is <= 3.
                # If {2,3} had succeeded, score would be <= 2.
                # So returning the first found is optimal.
                return result
                
    # If no clique found via reduction, fallback to local search
    # (This case is unlikely for small N and MAX_VAL, but good for safety)
    
    # Fallback: Random search + Local Search
    if best_result is None:
        # Initialize with a simple set
        best_result = list(range(1, 13))
        
        def count_primes(arr):
            union = set()
            for i in range(N):
                for j in range(i+1, N):
                    union |= get_prime_factors(arr[i] + arr[j])
            return len(union)
            
        current = list(best_result)
        current_score = count_primes(current)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                old = current[i]
                # Try swapping with a random number
                for _ in range(50):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in current:
                        continue
                    current[i] = new_val
                    new_score = count_primes(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    current[i] = old
                if improved:
                    break
        best_result = current
        
    return best_result
