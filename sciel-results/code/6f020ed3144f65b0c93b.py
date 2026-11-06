"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Helper to check if n is P-smooth
    def is_p_smooth(n, primes):
        if n <= 0:
            return False
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    # Helper to get prime factors
    def prime_factors(n):
        factors = set()
        if n <= 1:
            return factors
        d = 2
        while d * d <= n:
            if n % d:
                d += 1
            else:
                while n % d == 0:
                    n //= d
                factors.add(d)
        if n > 1:
            factors.add(n)
        return factors

    # Strategy 1: Find a clique of size 12 in the graph of 5-smooth numbers scaled by 30.
    # x_i = 30 * k_i. Sum = 30 * (k_i + k_j).
    # Primes of sum are {2, 3, 5} U Primes(k_i + k_j).
    # We want Primes(k_i + k_j) subset {2, 3, 5}.
    # So k_i + k_j must be 5-smooth.
    
    P = [2, 3, 5]
    max_k = 10000 // 30
    
    # Generate 5-smooth numbers <= max_k
    K = []
    k_set = set()
    
    def gen_smooth(max_val, primes):
        if not primes:
            if max_val >= 1:
                yield 1
            return
        p = primes[0]
        rest = primes[1:]
        pow_p = 1
        while pow_p <= max_val:
            for val in gen_smooth(max_val // pow_p, rest):
                yield val * pow_p
            pow_p *= p
            
    for k in gen_smooth(max_k, P):
        if k not in k_set:
            k_set.add(k)
            K.append(k)
            
    K.sort()
    n_k = len(K)
    
    # Build adjacency
    adj = [set() for _ in range(n_k)]
    for i in range(n_k):
        for j in range(i + 1, n_k):
            if is_p_smooth(K[i] + K[j], P):
                adj[i].add(j)
                adj[j].add(i)
                
    # Find clique of size 12 using Bron-Kerbosch with pivoting
    clique = []
    found_clique = False
    
    def bron_kerbosch(R, P_set, X):
        nonlocal found_clique, clique
        if time.time() > deadline:
            return
        if len(R) == 12:
            found_clique = True
            clique = R.copy()
            return
        if not P_set and not X:
            return
            
        # Pivot selection
        if P_set or X:
            U = list(P_set) + list(X)
            u = max(U, key=lambda v: len(adj[v] & (P_set | X)))
            candidates = [v for v in P_set if v not in adj[u]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch(R + [v], P_set & adj[v], X & adj[v])
            if found_clique:
                return
            P_set.remove(v)
            X.add(v)
            
    # Filter initial candidates
    initial_P = {i for i in range(n_k) if len(adj[i]) >= 11}
    if initial_P:
        bron_kerbosch([], initial_P, set())
        
    if found_clique:
        result = [30 * K[k] for k in clique]
        if len(set(result)) == 12 and all(x <= 10000 for x in result):
            return result

    # Strategy 2: Fallback Local Search
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
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Start with random multiples of 30 to bias towards {2,3,5}
        if random.random() < 0.7:
            # Random 5-smooth numbers scaled by 30
            base = random.sample(K, min(12, len(K)))
            if len(base) < 12:
                while len(base) < 12:
                    base.append(random.randint(1, max_k))
            curr_set = [30 * x for x in base[:12]]
            if len(set(curr_set)) < 12:
                continue
        else:
            curr_set = random.sample(range(1, 10001), 12)
            
        curr_score = evaluate_set(curr_set)
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
            if best_score <= 3:
                return best_set
                
        # Local Search: Swap optimization
        improved = True
        while improved and time.time() < deadline:
            improved = False
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate candidates
            candidates = [old_val + random.randint(-5, 5), random.randint(1, 10000)]
            # Bias towards 30 * 5-smooth
            candidates.extend([30 * random.choice(K) for _ in range(5)])
            
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
                    if curr_score <= 3:
                        return curr_set
                    if curr_score < best_score:
                        best_score = curr_score
                        best_set = curr_set.copy()
                        
    return best_set
# EVOLVE-BLOCK-END
