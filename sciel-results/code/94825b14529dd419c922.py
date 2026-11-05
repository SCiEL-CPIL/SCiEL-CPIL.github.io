import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    MAX_VAL = 10000
    N = 12
    MAX_SUM = 2 * MAX_VAL
    
    # Precompute SPF for factorization
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def count_distinct_primes(arr):
        primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                primes.update(get_prime_factors(arr[i] + arr[j]))
        return len(primes)
    
    # Initial best: simple arithmetic progression
    best_arr = list(range(1, 13))
    best_score = count_distinct_primes(best_arr)
    
    # Strategy: Try to find sets where all pairwise sums use only primes from a small set
    # Try prime sets in order of increasing size
    prime_set_candidates = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
    ]
    
    def is_smooth(n, prime_set):
        """Check if n is composed only of primes in prime_set"""
        if n <= 0:
            return False
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1
    
    def find_clique_for_smooth_sums(prime_set, time_budget):
        """
        Find N distinct numbers <= MAX_VAL such that all pairwise sums are smooth
        with respect to prime_set.
        """
        start = time.time()
        
        # Precompute which sums are smooth
        smooth_sums = set()
        for s in range(2, MAX_SUM + 1):
            if is_smooth(s, prime_set):
                smooth_sums.add(s)
        
        # Build adjacency: for each number a, which numbers b > a have a+b in smooth_sums
        adj = [[] for _ in range(MAX_VAL + 1)]
        for a in range(1, MAX_VAL + 1):
            for b in range(a + 1, MAX_VAL + 1):
                if a + b in smooth_sums:
                    adj[a].append(b)
                    adj[b].append(a)
        
        # Greedy + backtracking to find a clique of size N
        # Order nodes by degree (descending)
        nodes = list(range(1, MAX_VAL + 1))
        nodes.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # Convert to set for fast lookup
        adj_set = [set(adj[i]) for i in range(MAX_VAL + 1)]
        
        def backtrack(clique, available):
            if len(clique) == N:
                return clique[:]
            if time.time() - start > time_budget:
                return None
            if len(clique) + len(available) < N:
                return None
            
            # Try nodes from available (already all connected to clique)
            # Pick the one with highest degree in available subgraph
            best_node = None
            best_deg = -1
            for node in available:
                deg = len(adj_set[node] & set(available))
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
            
            if best_node is None:
                return None
            
            new_clique = clique + [best_node]
            new_available = [n for n in available if n in adj_set[best_node]]
            
            result = backtrack(new_clique, new_available)
            if result:
                return result
            
            # If greedy choice failed, try other nodes
            for node in available:
                if node == best_node:
                    continue
                if time.time() - start > time_budget:
                    return None
                new_clique = clique + [node]
                new_available = [n for n in available if n in adj_set[node]]
                result = backtrack(new_clique, new_available)
                if result:
                    return result
            
            return None
        
        # Initial available: all nodes with degree >= N-1
        initial_available = [n for n in nodes if len(adj_set[n]) >= N - 1]
        
        if not initial_available:
            return None
        
        result = backtrack([], initial_available)
        return result
    
    # Try each prime set
    for prime_set in prime_set_candidates:
        if time.time() > deadline - 5:
            break
        
        remaining_time = deadline - time.time() - 2
        if remaining_time < 1:
            break
        
        clique = find_clique_for_smooth_sums(prime_set, remaining_time)
        if clique is not None and len(clique) == N:
            # Verify
            if len(set(clique)) == N and all(x <= MAX_VAL for x in clique):
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique
                    if best_score == len(prime_set):
                        # Can't do better than the number of primes in the set
                        break
    
    # If we didn't find a perfect smooth set, try local search
    if time.time() < deadline - 10:
        # Simulated annealing with incremental updates
        current = best_arr[:]
        current_score = best_score
        
        # Precompute current prime set
        current_primes = set()
        for i in range(N):
            for j in range(i + 1, N):
                current_primes.update(get_prime_factors(current[i] + current[j]))
        
        T = 5.0
        cooling = 0.995
        
        while time.time() < deadline - 2:
            if T < 0.1:
                break
            
            idx = random.randint(0, N - 1)
            old_val = current[idx]
            
            # Generate new value: bias towards values that make smooth sums
            # Pick a random other element and try to make the sum smooth
            if random.random() < 0.5:
                # Try to make sum with a random other element smooth
                other_idx = random.choice([i for i in range(N) if i != idx])
                target_sum = random.choice([2**a * 3**b for a in range(15) for b in range(10) if 2**a * 3**b <= MAX_SUM])
                new_val = target_sum - current[other_idx]
                if new_val < 1 or new_val > MAX_VAL:
                    new_val = random.randint(1, MAX_VAL)
            else:
                new_val = random.randint(1, MAX_VAL)
            
            # Check distinctness
            if new_val in current and new_val != old_val:
                continue
            
            # Compute new prime set
            new_primes = set(current_primes)
            # Remove contributions involving old_val
            # This is complex; just recompute
            current[idx] = new_val
            new_score = count_distinct_primes(current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_arr = current[:]
            else:
                delta = new_score - current_score
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_score = new_score
                else:
                    current[idx] = old_val
            
            T *= cooling
    
    return best_arr

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res), "elements, max:", max(res))
    # Print prime count
    spf_local = list(range(20001))
    for i in range(2, int(20001**0.5) + 1):
        if spf_local[i] == i:
            for j in range(i*i, 20001, i):
                if spf_local[j] == j:
                    spf_local[j] = i
    def gf(n):
        f = set()
        while n > 1:
            p = spf_local[n]
            f.add(p)
            while n % p == 0:
                n //= p
        return f
    ps = set()
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            ps.update(gf(res[i]+res[j]))
    print("Distinct primes:", len(ps))
