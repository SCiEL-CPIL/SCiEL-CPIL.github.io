"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL  # 20000
    
    # Precompute Smallest Prime Factor (SPF) up to MAX_SUM
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Return set of distinct prime factors of n using SPF."""
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)
    
    def evaluate_set(nums):
        """Calculate the number of distinct prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(all_primes)
    
    # Precompute prime factors for all sums 2..20000
    sum_factors = [frozenset() for _ in range(MAX_SUM + 1)]
    for s in range(2, MAX_SUM + 1):
        sum_factors[s] = get_prime_factors(s)
    
    def evaluate_set_fast(nums):
        """Fast evaluation using precomputed sum_factors."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(sum_factors[nums[i] + nums[j]])
        return len(all_primes)
    
    # Generate candidate prime sets to try, in order of size
    # We want to find the smallest P such that there exist 12 numbers
    # whose pairwise sums are all P-smooth
    
    # First, generate all primes up to 20000
    primes_up_to_20000 = []
    for p in range(2, MAX_SUM + 1):
        if spf[p] == p:
            primes_up_to_20000.append(p)
    
    # Strategy: Try prime sets of increasing size
    # For each prime set P, build graph where edge (u,v) exists if u+v is P-smooth
    # Then find clique of size 12
    
    def is_p_smooth(n, prime_set):
        """Check if n is smooth with respect to prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1
    
    # For efficiency, precompute which sums are P-smooth for a given P
    def build_smooth_sums(prime_set):
        """Return a boolean array where arr[s] = True if s is P-smooth."""
        smooth = [False] * (MAX_SUM + 1)
        for s in range(2, MAX_SUM + 1):
            if is_p_smooth(s, prime_set):
                smooth[s] = True
        return smooth
    
    # Greedy clique finder on a graph defined by adjacency matrix
    def find_clique_size_12(adj_matrix, n_vertices, max_iter=500):
        """
        Find a clique of size 12 using greedy + backtracking heuristic.
        adj_matrix[i][j] = True if edge between i and j.
        Returns list of vertex indices or None.
        """
        # Sort vertices by degree (descending) for better greedy
        degrees = [sum(1 for j in range(n_vertices) if j != i and adj_matrix[i][j]) 
                   for i in range(n_vertices)]
        order = sorted(range(n_vertices), key=lambda x: -degrees[x])
        
        # Multiple random restarts with greedy
        for restart in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Pick a random starting vertex (biased toward high-degree)
            if restart < 10:
                start = order[random.randint(0, min(100, n_vertices-1))]
            else:
                start = order[random.randint(0, n_vertices-1)]
            
            clique = [start]
            # Greedily add vertices
            for _ in range(11):
                # Find candidates adjacent to all in clique
                candidates = []
                for v in range(n_vertices):
                    if v in clique:
                        continue
                    ok = True
                    for c in clique:
                        if not adj_matrix[v][c]:
                            ok = False
                            break
                    if ok:
                        candidates.append(v)
                
                if not candidates:
                    break
                
                # Pick the one with highest degree within remaining candidates
                # or random for diversity
                if random.random() < 0.3:
                    next_v = random.choice(candidates)
                else:
                    # Pick candidate with highest degree
                    best_v = max(candidates, key=lambda v: degrees[v])
                    next_v = best_v
                
                clique.append(next_v)
            
            if len(clique) == 12:
                return clique
        
        return None
    
    # Try prime sets of increasing size
    # Start with size 2, then 3, etc.
    
    # For size 2: try all pairs of small primes
    small_primes = primes_up_to_20000[:50]  # First 50 primes
    
    best_result = None
    best_score = float('inf')
    
    # First, try to find a solution with 2 primes (optimal)
    found = False
    
    for p1_idx in range(len(small_primes)):
        if time.time() > deadline:
            break
        p1 = small_primes[p1_idx]
        for p2_idx in range(p1_idx + 1, len(small_primes)):
            if time.time() > deadline:
                break
            p2 = small_primes[p2_idx]
            prime_set = frozenset([p1, p2])
            
            # Build smooth sums
            smooth = build_smooth_sums(prime_set)
            
            # Build graph: vertices are numbers 1..10000
            # Edge (u,v) if smooth[u+v] is True
            # This is a 10000x10000 graph - too large for explicit matrix
            # Instead, use implicit representation and greedy search
            
            # For greedy clique, we need to check adjacency quickly
            # Let's limit to a subset of candidate numbers
            # Numbers that appear in many smooth sums are better candidates
            
            # Count for each number u, how many v exist such that u+v is smooth
            # This is expensive for 10000 numbers, so sample
            
            # Alternative: just try random numbers and check
            # Use a different approach: generate candidate numbers that are "good"
            
            # A number u is good if there are many v such that u+v is P-smooth
            # Let's estimate this by checking a sample of v's
            
            # Actually, let's just do a direct search:
            # Try to build a clique greedily from random starts
            
            n_found = 0
            for attempt in range(200):
                if time.time() > deadline:
                    break
                
                # Start with a random number
                clique = [random.randint(1, MAX_VAL)]
                
                for _ in range(11):
                    # Find a number not in clique that is adjacent to all in clique
                    found_next = False
                    for _try in range(50):  # Try 50 random candidates
                        v = random.randint(1, MAX_VAL)
                        if v in clique:
                            continue
                        ok = True
                        for c in clique:
                            if not smooth[c + v]:
                                ok = False
                                break
                        if ok:
                            clique.append(v)
                            found_next = True
                            break
                    
                    if not found_next:
                        break
                
                if len(clique) == 12:
                    # Verify
                    if len(set(clique)) == 12 and all(1 <= x <= MAX_VAL for x in clique):
                        score = evaluate_set_fast(clique)
                        if score <= 2:
                            return clique
                        elif score < best_score:
                            best_score = score
                            best_result = clique.copy()
            
            if found:
                break
    
    # If we didn't find a 2-prime solution, try 3 primes
    if best_score > 2 and time.time() < deadline:
        # Try triples of small primes
        triple_count = 0
        for p1_idx in range(min(20, len(small_primes))):
            if time.time() > deadline:
                break
            p1 = small_primes[p1_idx]
            for p2_idx in range(p1_idx + 1, min(30, len(small_primes))):
                if time.time() > deadline:
                    break
                p2 = small_primes[p2_idx]
                for p3_idx in range(p2_idx + 1, min(40, len(small_primes))):
                    if time.time() > deadline:
                        break
                    if triple_count > 100:
                        break
                    p3 = small_primes[p3_idx]
                    prime_set = frozenset([p1, p2, p3])
                    triple_count += 1
                    
                    smooth = build_smooth_sums(prime_set)
                    
                    for attempt in range(100):
                        if time.time() > deadline:
                            break
                        
                        clique = [random.randint(1, MAX_VAL)]
                        
                        for _ in range(11):
                            found_next = False
                            for _try in range(50):
                                v = random.randint(1, MAX_VAL)
                                if v in clique:
                                    continue
                                ok = True
                                for c in clique:
                                    if not smooth[c + v]:
                                        ok = False
                                        break
                                if ok:
                                    clique.append(v)
                                    found_next = True
                                    break
                            
                            if not found_next:
                                break
                        
                        if len(clique) == 12 and len(set(clique)) == 12:
                            score = evaluate_set_fast(clique)
                            if score < best_score:
                                best_score = score
                                best_result = clique.copy()
                                if best_score <= 3:
                                    return best_result
    
    # Fallback: local search from multiple initializations
    if time.time() < deadline:
        # Initialize with structured sets
        inits = []
        # Multiples of 6
        inits.append([6 * i for i in range(1, 13)])
        # 6k+3
        inits.append([6 * i + 3 for i in range(0, 12)])
        # 6k+1
        inits.append([6 * i + 1 for i in range(0, 12)])
        # Random
        inits.append(random.sample(range(1, MAX_VAL + 1), 12))
        # Small numbers
        inits.append(list(range(1, 13)))
        
        for init in inits:
            if time.time() > deadline:
                break
            
            curr = init.copy()
            curr_score = evaluate_set_fast(curr)
            
            if curr_score < best_score:
                best_score = curr_score
                best_result = curr.copy()
                if best_score <= 2:
                    return best_result
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                idx = random.randint(0, 11)
                old_val = curr[idx]
                
                # Generate candidates
                candidates = set()
                # Neighbors
                for delta in range(-20, 21):
                    candidates.add(old_val + delta)
                # Random
                for _ in range(5):
                    candidates.add(random.randint(1, MAX_VAL))
                # Structured
                for r in range(6):
                    candidates.add(random.randint(1, 1666) * 6 + r)
                
                for new_val in candidates:
                    if new_val < 1 or new_val > MAX_VAL:
                        continue
                    if new_val in curr:
                        continue
                    
                    new_set = curr.copy()
                    new_set[idx] = new_val
                    new_score = evaluate_set_fast(new_set)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        curr = new_set
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_result = curr.copy()
                            if best_score <= 2:
                                return best_result
                        break
    
    # Return best found
    if best_result is None:
        return list(range(1, 13))
    
    return best_result
# EVOLVE-BLOCK-END
