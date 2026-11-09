import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    n = 12
    max_val = 10000
    max_sum = 2 * max_val
    
    # --- Attempt 1: Find clique of size 12 with 7-smooth sums ---
    
    # Precompute is_7_smooth
    is_7_smooth = [False] * (max_sum + 1)
    for s in range(2, max_sum + 1):
        t = s
        while t % 2 == 0: t //= 2
        while t % 3 == 0: t //= 3
        while t % 5 == 0: t //= 5
        while t % 7 == 0: t //= 7
        if t == 1:
            is_7_smooth[s] = True
            
    # Build graph
    neighbors = [[] for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i + 1, max_val + 1):
            if is_7_smooth[i + j]:
                neighbors[i].append(j)
                neighbors[j].append(i)
                
    degrees = [len(neighbors[i]) for i in range(1, max_val + 1)]
    candidates = list(range(1, max_val + 1))
    candidates.sort(key=lambda x: degrees[x], reverse=True)
    
    neighbor_sets = [set(neighbors[i]) for i in range(1, max_val + 1)]
    best_set = []
    
    def backtrack_clique(current_set, available_candidates_set):
        nonlocal best_set
        if time.time() > deadline:
            return False
        
        if len(current_set) == n:
            best_set = current_set.copy()
            return True
            
        if not available_candidates_set:
            return False
            
        if len(current_set) + len(available_candidates_set) < n:
            return False
            
        for c in candidates:
            if c in available_candidates_set:
                new_available = available_candidates_set.intersection(neighbor_sets[c])
                if backtrack_clique(current_set + [c], new_available):
                    return True
        return False

    initial_candidates_set = set([c for c in candidates if degrees[c] >= n - 1])
    
    if backtrack_clique([], initial_candidates_set):
        return sorted(best_set)
        
    # --- Attempt 2: Greedy minimization of new primes ---
    
    # Precompute prime factors
    prime_factors = [[] for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        if not prime_factors[i]:
            for j in range(i, max_sum + 1, i):
                prime_factors[j].append(i)
                
    current_set = []
    current_primes = set()
    available = set(range(1, max_val + 1))
    
    # Use candidates sorted by degree as a heuristic for "smoothness"
    # But we need to check all available.
    
    for _ in range(n):
        best_candidate = None
        min_new_primes = float('inf')
        
        # Iterate through candidates in order of preference (degree)
        # This is a heuristic to find good candidates early.
        # We break early if we find 0 new primes.
        
        for c in candidates:
            if c not in available:
                continue
                
            new_primes = set()
            for x in current_set:
                s = c + x
                for p in prime_factors[s]:
                    if p not in current_primes:
                        new_primes.add(p)
                        
            if len(new_primes) < min_new_primes:
                min_new_primes = len(new_primes)
                best_candidate = c
                if min_new_primes == 0:
                    break
            
        if best_candidate is not None:
            current_set.append(best_candidate)
            available.remove(best_candidate)
            for x in current_set[:-1]:
                s = best_candidate + x
                for p in prime_factors[s]:
                    current_primes.add(p)
                    
    return sorted(current_set)
