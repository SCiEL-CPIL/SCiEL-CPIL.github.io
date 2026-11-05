import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 90s limit

    # 1. Generate pool of 5-smooth numbers (primes 2, 3, 5) up to 10000
    pool = set()
    for a in range(0, 15):
        v2 = 2 ** a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3 ** b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                val = v23 * (5 ** c)
                if val > 10000:
                    break
                pool.add(val)
    pool = list(pool)
    pool.sort()
    
    # Sieve for smallest prime factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Precompute prime factors for all sums in pool
    # pool_size ~ 300, so ~45k pairs
    sum_factors = {}
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            s = pool[i] + pool[j]
            if s not in sum_factors:
                sum_factors[s] = get_prime_factors(s)

    # Define target prime sets in order of preference (fewer primes is better)
    target_prime_sets = [
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13}
    ]

    def find_clique_in_constraint_graph(target_primes):
        """
        Finds a set of 12 distinct numbers from the pool such that 
        for all pairs (i, j), PrimeFactors(pool[i] + pool[j]) is a subset of target_primes.
        Uses a greedy backtracking search.
        """
        # Build adjacency list: adj[i] is a list of j such that sum_factors[pool[i]+pool[j]] <= target_primes
        n = len(pool)
        adj = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                factors = sum_factors[pool[i] + pool[j]]
                if factors.issubset(target_primes):
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Greedy clique finding with backtracking
        # We want a clique of size 12
        best_clique = []
        
        def backtrack(current_clique, candidates):
            if len(current_clique) == 12:
                return current_clique[:]
            
            if not candidates:
                return None
                
            # Heuristic: pick candidate with highest degree in the remaining graph
            # To speed up, we can just iterate, but ordering helps
            # Simple strategy: iterate candidates, try adding
            
            # Optimization: If we can't reach 12, prune
            if len(current_clique) + len(candidates) < 12:
                return None

            # Try each candidate
            # Sort candidates by degree to find large cliques faster?
            # For now, simple iteration.
            for idx in range(len(candidates)):
                node = candidates[idx]
                # New candidates must be neighbors of 'node' and appear after 'node' in the list to avoid duplicates
                new_candidates = [c for c in candidates[idx+1:] if c in adj[node]]
                
                result = backtrack(current_clique + [node], new_candidates)
                if result:
                    return result
            return None

        # Initial candidates: all nodes
        # To make it faster, we can try starting from different nodes
        # But the pool is small enough that a single run might suffice if the graph is dense
        # If the graph is sparse, it might fail.
        
        # Let's try a few random starting orders if the first one fails
        indices = list(range(n))
        random.shuffle(indices)
        
        # Try standard backtrack from full list
        # If n is ~300, and we need 12, the branching factor is high.
        # We need a more efficient search.
        
        # Alternative: Greedy construction
        # 1. Pick a node with high degree.
        # 2. Intersect neighbors.
        # 3. Repeat.
        
        def greedy_clique():
            current_clique = []
            candidates = set(range(n))
            
            for _ in range(12):
                if not candidates:
                    return None
                # Pick node with max degree within candidates
                best_node = -1
                max_deg = -1
                for node in candidates:
                    deg = len([x for x in candidates if x in adj[node]])
                    if deg > max_deg:
                        max_deg = deg
                        best_node = node
                
                if best_node == -1:
                    return None
                    
                current_clique.append(best_node)
                candidates.discard(best_node)
                # Update candidates to be intersection of neighbors
                new_candidates = set()
                for x in candidates:
                    if x in adj[best_node]:
                        new_candidates.add(x)
                candidates = new_candidates
                
            return current_clique

        # Try greedy first
        clique = greedy_clique()
        if clique:
            return [pool[i] for i in clique]
            
        # If greedy fails, try a few random restarts of greedy
        for _ in range(20):
            if time.time() > deadline:
                return None
            # Perturb the candidate set or just rerun with random tie-breaking
            # Simple random restart: shuffle the pool indices? No, the graph is fixed.
            # The greedy is deterministic based on degrees. 
            # We can randomize the tie-breaking in "pick node with max degree".
            
            # Let's implement a randomized greedy
            def randomized_greedy():
                current_clique = []
                candidates = set(range(n))
                
                for _ in range(12):
                    if not candidates:
                        return None
                    # Filter candidates by degree > 0
                    valid_candidates = [node for node in candidates if len([x for x in candidates if x in adj[node]]) > 0]
                    if not valid_candidates:
                        return None
                    
                    # Pick randomly from top 10% by degree?
                    # Calculate degrees
                    degs = {}
                    for node in valid_candidates:
                        degs[node] = len([x for x in candidates if x in adj[node]])
                    
                    # Sort by degree
                    sorted_nodes = sorted(valid_candidates, key=lambda x: degs[x], reverse=True)
                    
                    # Pick from top few
                    k = min(5, len(sorted_nodes))
                    best_node = random.choice(sorted_nodes[:k])
                    
                    current_clique.append(best_node)
                    candidates.discard(best_node)
                    new_candidates = set()
                    for x in candidates:
                        if x in adj[best_node]:
                            new_candidates.add(x)
                    candidates = new_candidates
                    
                return current_clique

            clique = randomized_greedy()
            if clique:
                return [pool[i] for i in clique]
                
        return None

    # Try to find a solution for each target prime set
    best_solution = None
    best_score = float('inf')
    
    for target_primes in target_prime_sets:
        if time.time() > deadline:
            break
            
        solution = find_clique_in_constraint_graph(target_primes)
        if solution:
            # Verify solution
            primes_union = set()
            for i in range(12):
                for j in range(i + 1, 12):
                    primes_union.update(sum_factors[solution[i] + solution[j]])
            
            if len(primes_union) <= len(target_primes):
                # Found a valid solution for this target
                # Since we iterate from smallest target, this is optimal for this search
                best_solution = solution
                best_score = len(primes_union)
                break # Found the best possible in our target list
    
    # If no structured solution found, fall back to local search on the 5-smooth pool
    if best_solution is None:
        # Initialize with a random subset of the pool
        current_set = random.sample(pool, 12)
        
        def evaluate_set(numbers):
            prime_union = set()
            for i in range(12):
                for j in range(i + 1, 12):
                    s = numbers[i] + numbers[j]
                    prime_union.update(sum_factors.get(s, set()))
            return len(prime_union)
            
        current_score = evaluate_set(current_set)
        best_score = current_score
        best_solution = current_set[:]
        
        # Local Search
        temp = 2.0
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 2000
        
        for step in range(100000):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            new_val = random.choice(pool)
            
            if new_val == old_val or new_val in current_set:
                continue
            
            current_set[idx] = new_val
            new_score = evaluate_set(current_set)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_solution = current_set[:]
                    if best_score <= 3:
                        break
            else:
                delta = new_score - current_score
                accept_prob = math.exp(-delta / max(temp, 0.01))
                if random.random() < accept_prob:
                    current_score = new_score
                    no_improve += 1
                else:
                    current_set[idx] = old_val
                    no_improve += 1
                    
            temp *= cooling
            if temp < 0.05:
                temp = 0.05
                
            if no_improve > max_no_improve:
                no_improve = 0
                temp = 2.0
                # Perturb
                for _ in range(5):
                    idx = random.randint(0, 11)
                    new_val = random.choice(pool)
                    if new_val not in current_set:
                        current_set[idx] = new_val
                current_score = evaluate_set(current_set)

    if best_solution is None:
        return list(range(1, 13))
        
    return best_solution

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# EVOLVE-BLOCK-END
