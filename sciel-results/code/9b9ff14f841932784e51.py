def construct():
    import time
    import random
    from math import gcd, lcm
    
    start_time = time.time()
    DEADLINE = 50  # safety margin before 60s limit
    
    n_max = 256
    values = list(range(2, n_max + 1))
    num_vertices = len(values)
    
    # Precompute weights
    weights = [1.0 / v for v in values]
    
    # Precompute conflict matrix
    # conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    conflict = [[False] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        vi = values[i]
        for j in range(i + 1, num_vertices):
            vj = values[j]
            if lcm(vi, vj) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
                
    def is_compatible(idx, current_set_indices):
        """Check if idx is compatible with all indices in current_set_indices"""
        for j in current_set_indices:
            if conflict[idx][j]:
                return False
        return True
        
    def calculate_score(indices):
        return sum(weights[i] for i in indices)
        
    def greedy_construct(order):
        """Construct a set using greedy strategy based on given order of indices"""
        current_set = []
        for idx in order:
            if is_compatible(idx, current_set):
                current_set.append(idx)
        return current_set
        
    def local_search(initial_set):
        """Improve the set using swap and add operations"""
        current_set = list(initial_set)
        current_score = calculate_score(current_set)
        improved = True
        
        while improved:
            improved = False
            
            # 1. Try to add any compatible element
            for idx in range(num_vertices):
                if idx not in current_set:
                    if is_compatible(idx, current_set):
                        new_set = current_set + [idx]
                        new_score = current_score + weights[idx]
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            improved = True
                            break # Restart loop after change
            
            # 2. Try to swap one element with another
            if not improved:
                for i_idx, u in enumerate(current_set):
                    for v in range(num_vertices):
                        if v in current_set:
                            continue
                        # Check if v is compatible with all others in current_set except u
                        compatible = True
                        for w in current_set:
                            if w == u:
                                continue
                            if conflict[v][w]:
                                compatible = False
                                break
                        if compatible:
                            new_score = current_score - weights[u] + weights[v]
                            if new_score > current_score:
                                # Perform swap
                                current_set[i_idx] = v
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                        
        return current_set, current_score

    best_score = 0.0
    best_set = []
    
    # Strategy 1: Standard Greedy (ascending value = descending weight)
    order1 = list(range(num_vertices))
    set1 = greedy_construct(order1)
    set1, score1 = local_search(set1)
    if score1 > best_score:
        best_score = score1
        best_set = set1
        
    # Strategy 2: Randomized Greedy + Local Search
    # Run several iterations to escape local optima
    for _ in range(20):
        if time.time() - start_time > DEADLINE:
            break
            
        order2 = list(range(num_vertices))
        random.shuffle(order2)
        # Bias towards smaller values (higher weights) but with randomness
        # Sort by a mix of weight and random noise
        # Actually, just shuffling the full list and running greedy might be too weak.
        # Let's try a "Randomized Greedy" where we pick the best available from a random subset?
        # Or just shuffle the priority list.
        
        set2 = greedy_construct(order2)
        set2, score2 = local_search(set2)
        if score2 > best_score:
            best_score = score2
            best_set = set2

    # Strategy 3: Greedy with "Conflict Weight" heuristic
    # Prioritize numbers that have fewer conflicts?
    # Calculate conflict count for each
    conflict_counts = [sum(1 for j in range(num_vertices) if conflict[i][j]) for i in range(num_vertices)]
    # Sort by (weight desc, conflict count asc)
    order3 = sorted(range(num_vertices), key=lambda i: (-weights[i], conflict_counts[i]))
    set3 = greedy_construct(order3)
    set3, score3 = local_search(set3)
    if score3 > best_score:
        best_score = score3
        best_set = set3
        
    # Strategy 4: Start with primes?
    # Primes > 128 are always compatible with each other?
    # lcm(p, q) = p*q. If p,q > 128, p*q > 16384 > 256.
    # So all primes in (128, 256] are mutually compatible.
    # Also, small primes might be compatible with large composites.
    # Let's just let the local search handle it.
    
    # Convert indices to values
    result = [values[i] for i in best_set]
    return result
