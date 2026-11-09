import time
import random
import sys

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin
    
    n = 12
    max_val = 10000
    
    # Target prime sets to try, in order of increasing size (lower score potential)
    # We want the union of prime factors of sums to be small.
    # If we construct a_i = k_i * M where M = prod(P), and k_i+k_j is P-smooth,
    # then prime factors of a_i+a_j are subset of P.
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    best_result = None
    best_score = float('inf')
    
    # Helper to generate P-smooth numbers up to limit
    def get_smooth_numbers(P, limit):
        smooth = set([1])
        for p in P:
            new_smooth = set()
            for s in smooth:
                mult = s
                while mult <= limit:
                    new_smooth.add(mult)
                    mult *= p
            smooth = smooth.union(new_smooth)
        return smooth

    for P in prime_sets:
        if time.time() - start_time > timeout:
            break
            
        M = 1
        for p in P:
            M *= p
            
        max_k = max_val // M
        if max_k < n:
            continue
            
        # We need k_1..k_n distinct in [1, max_k] such that k_i+k_j is P-smooth.
        # Precompute P-smooth numbers up to 2*max_k
        limit_sum = 2 * max_k
        smooth_sums = get_smooth_numbers(P, limit_sum)
        
        # Build adjacency list for the graph of k's
        # Node k is connected to node l if k+l is in smooth_sums
        # We want a clique of size n.
        
        # Optimization: If max_k is small (e.g., < 100), we can use bitmask or simple backtracking.
        # If max_k is large (e.g., 1666), backtracking might be slow if the graph is sparse.
        # However, for P={2,3}, smooth numbers are dense enough? 
        # 3-smooth numbers: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36...
        # Density is roughly (log N)^2 / (2 log 2 log 3). For N=3332, this is decent.
        
        # Let's use a randomized greedy with backtracking for the clique search.
        # To speed up, we can order nodes by degree (number of smooth sums).
        
        # Calculate degree for each k
        degrees = [0] * (max_k + 1)
        for k in range(1, max_k + 1):
            for l in range(k + 1, max_k + 1):
                if (k + l) in smooth_sums:
                    degrees[k] += 1
                    degrees[l] += 1
        
        # Sort candidates by degree descending to prioritize "central" nodes
        candidates = list(range(1, max_k + 1))
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        
        found_clique = False
        # Try multiple random restarts or just one deep search?
        # Given the time limit, we try a few restarts with different random orderings if the first fails.
        
        for restart in range(10):
            if time.time() - start_time > timeout:
                break
                
            # Shuffle the tail of candidates to allow different paths, but keep high degree ones prioritized
            # Or just use the sorted list but pick randomly from top K?
            # Let's stick to the sorted list for deterministic first try, then randomize.
            
            search_order = candidates[:]
            if restart > 0:
                # Randomize the order of candidates slightly
                random.shuffle(search_order)
                # Re-sort by degree but with some randomness? 
                # Actually, just shuffling is fine for restarts.
                
            clique = []
            
            def backtrack(clique, start_idx):
                if len(clique) == n:
                    return True
                if time.time() - start_time > timeout:
                    return False
                    
                # Pruning: if remaining candidates < needed, fail
                if start_idx + (n - len(clique)) > len(search_order):
                    return False
                    
                for i in range(start_idx, len(search_order)):
                    k = search_order[i]
                    # Check if k is compatible with all in current clique
                    compatible = True
                    for ck in clique:
                        if (k + ck) not in smooth_sums:
                            compatible = False
                            break
                    if compatible:
                        clique.append(k)
                        if backtrack(clique, i + 1):
                            return True
                        clique.pop()
                return False
            
            if backtrack(clique, 0):
                found_clique = True
                # Construct a_i
                a_set = [k * M for k in clique]
                # Verify
                if len(set(a_set)) == n and all(1 <= x <= max_val for x in a_set):
                    # Calculate score
                    primes_found = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            s = a_set[i] + a_set[j]
                            temp = s
                            for p in P:
                                while temp % p == 0:
                                    temp //= p
                            if temp > 1:
                                # Should not happen if logic is correct, but just in case
                                # Factor temp
                                t = temp
                                d = 2
                                while d * d <= t:
                                    while t % d == 0:
                                        primes_found.add(d)
                                        t //= d
                                    d += 1
                                if t > 1:
                                    primes_found.add(t)
                            primes_found.update(P)
                    
                    score = len(primes_found)
                    if score < best_score:
                        best_score = score
                        best_result = a_set
                break # Found a valid set for this P, move to next P (smaller score)
        
        if found_clique:
            # Since we iterate P from smallest to largest, and we found a solution,
            # this solution has score <= len(P). 
            # Any subsequent P will have score >= len(P) > len(current P).
            # So we can break if we found a solution for a small P.
            # However, it's possible that a solution for P={2,3,5,7} has score 4,
            # but a weird non-scaled solution has score 3? Unlikely.
            # But strictly, if we found a solution for P, the score is bounded by len(P).
            # If we found one for P={2,3}, score is 2 or 3. That's optimal or near-optimal.
            # So we break.
            break
            
    if best_result is None:
        # Fallback: Local Search
        # Precompute prime factors for all numbers up to 20000
        sum_primes = [set() for _ in range(2 * max_val + 1)]
        for p in range(2, 2 * max_val + 1):
            is_prime = True
            for d in range(2, int(p**0.5) + 1):
                if p % d == 0:
                    is_prime = False
                    break
            if is_prime:
                for multiple in range(p, 2 * max_val + 1, p):
                    sum_primes[multiple].add(p)
                    
        def evaluate_set(A):
            all_primes = set()
            for i in range(n):
                for j in range(i + 1, n):
                    s = A[i] + A[j]
                    all_primes.update(sum_primes[s])
            return len(all_primes)
        
        # Simple local search
        best_set = list(range(1, n + 1))
        best_score = evaluate_set(best_set)
        
        current_set = best_set[:]
        
        while time.time() - start_time < timeout:
            # Try to improve by swapping one element
            i = random.randint(0, n - 1)
            old_val = current_set[i]
            new_val = random.randint(1, max_val)
            
            if new_val in current_set and new_val != old_val:
                continue
                
            current_set[i] = new_val
            score = evaluate_set(current_set)
            
            if score < best_score:
                best_score = score
                best_set = current_set[:]
            else:
                current_set[i] = old_val
                
        best_result = best_set

    return best_result
