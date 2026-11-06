"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Helper to get prime factors
    def prime_factors(n):
        factors = set()
        if n <= 1:
            return factors
        if n % 2 == 0:
            factors.add(2)
            while n % 2 == 0:
                n //= 2
        p = 3
        while p * p <= n:
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
            p += 2
        if n > 1:
            factors.add(n)
        return factors

    # Strategy 1: Find a clique of size 12 in the graph of 2,3-smooth numbers scaled by 6.
    # x_i = 6 * k_i. Sum = 6 * (k_i + k_j).
    # Primes of sum are {2, 3} U Primes(k_i + k_j).
    # We want Primes(k_i + k_j) subset {2, 3}.
    # So k_i + k_j must be 2^a * 3^b.
    
    max_k = 10000 // 6
    S = []
    s_set = set()
    
    # Generate 2^a * 3^b <= max_k
    for a in range(0, 15):
        pow2 = (1 << a)
        if pow2 > max_k:
            break
        for b in range(0, 15):
            val = pow2 * (3 ** b)
            if val > max_k:
                break
            if val not in s_set:
                s_set.add(val)
                S.append(val)
                
    S.sort()
    n_s = len(S)
    
    # Build adjacency
    adj = [set() for _ in range(n_s)]
    for i in range(n_s):
        for j in range(i + 1, n_s):
            if S[i] + S[j] in s_set:
                adj[i].add(j)
                adj[j].add(i)
                
    # Find clique of size 12
    clique = []
    found_clique = False
    
    def backtrack(candidates):
        nonlocal found_clique, clique
        if time.time() > deadline:
            return
        if len(clique) == 12:
            found_clique = True
            return
        if len(clique) + len(candidates) < 12:
            return
        if not candidates:
            return
        
        for idx, v in enumerate(candidates):
            clique.append(v)
            
            new_cands = []
            for u in candidates[idx+1:]:
                if u in adj[v]:
                    new_cands.append(u)
                    
            backtrack(new_cands)
            
            if found_clique:
                return
            clique.pop()
            
    # Filter candidates that have degree >= 11
    initial_cands = [i for i in range(n_s) if len(adj[i]) >= 11]
    
    if initial_cands:
        backtrack(initial_cands)
        
    if found_clique:
        result = [6 * S[k] for k in clique]
        if len(set(result)) == 12 and all(x <= 10000 for x in result):
            return result

    # Strategy 2: Fallback Local Search if clique not found or timeout during clique search
    # Try to minimize prime count. Seed has 8.
    
    def evaluate_set(nums):
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(prime_factors(nums[i] + nums[j]))
        return len(all_primes)
        
    best_score = float('inf')
    best_set = list(range(1, 13))
    best_score = evaluate_set(best_set)
    
    # Randomized Local Search
    for _ in range(200):
        if time.time() > deadline:
            break
            
        # Start with random multiples of 6 to bias towards {2,3}
        if random.random() < 0.5:
            # Random 2,3-smooth numbers scaled by 6
            base = random.sample(S, min(12, len(S)))
            if len(base) < 12:
                # Pad with random
                while len(base) < 12:
                    base.append(random.randint(1, 10000))
            curr_set = [6 * x for x in base[:12]]
            # Ensure distinct
            if len(set(curr_set)) < 12:
                continue
        else:
            curr_set = random.sample(range(1, 10001), 12)
            
        curr_score = evaluate_set(curr_set)
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
            if best_score <= 2:
                return best_set
                
        # Local Search: Swap optimization
        improved = True
        while improved and time.time() < deadline:
            improved = False
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate candidates
            candidates = [old_val + random.randint(-10, 10), random.randint(1, 10000)]
            # Bias towards 6 * 2^a * 3^b
            candidates.extend([6 * random.choice(S) for _ in range(5)])
            
            for new_val in candidates:
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in curr_set:
                    continue
                
                new_set = curr_set.copy()
                new_set[idx] = new_val
                new_score = evaluate_set(new_set)
                
                if new_score < curr_score:
                    curr_score = new_score
                    curr_set = new_set
                    improved = True
                    if curr_score <= 2:
                        return curr_set
                    if curr_score < best_score:
                        best_score = curr_score
                        best_set = curr_set.copy()
                        
    return best_set
# EVOLVE-BLOCK-END
