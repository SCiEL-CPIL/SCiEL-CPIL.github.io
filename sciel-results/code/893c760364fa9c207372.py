import time
import random
import sys

def _count_triangles(adj, n):
    """Count triangles in a graph represented by adjacency bitmasks."""
    count = 0
    for i in range(n):
        # Only consider j > i to avoid double counting in the sum, 
        # but the common neighbor count includes all neighbors.
        # Triangle (i,j,k) with i<j<k is counted when:
        # 1. i, j: common neighbor k (k > j > i) -> counted
        # 2. i, k: common neighbor j (j < k, j > i) -> counted
        # 3. j, k: common neighbor i (i < j < k) -> counted
        # So summing over all i < j where edge exists, and counting common neighbors,
        # counts each triangle 3 times.
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                count += bin(common).count('1')
    return count // 3

def _has_ind_set_10(adj, n):
    """Check if there is an independent set of size 10."""
    # Backtracking with bitmasks
    # available: bitmask of vertices available
    # needed: number of vertices still needed
    
    def backtrack(available, needed):
        if needed == 0:
            return True
        if bin(available).count('1') < needed:
            return False
        
        # Heuristic: Pick vertex with max degree in available subgraph
        # to prune the "include" branch faster? 
        # Actually, for independent set, picking a vertex with high degree 
        # removes many neighbors from the "include" branch.
        
        best_v = -1
        max_deg = -1
        temp_avail = available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            deg = bin(adj[v] & available).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_avail ^= lsb
        
        if best_v == -1:
            return False # No vertices left but needed > 0
            
        v = best_v
        
        # Branch 1: Include v
        # Remove v and its neighbors
        new_avail_1 = available & ~(adj[v] | (1 << v))
        if backtrack(new_avail_1, needed - 1):
            return True
        
        # Branch 2: Exclude v
        new_avail_2 = available & ~(1 << v)
        if backtrack(new_avail_2, needed):
            return True
        
        return False

    return backtrack((1 << n) - 1, 10)

def _fitness(adj, n):
    """
    Fitness: lower is better.
    1. Minimize triangles.
    2. If triangles == 0, minimize independent set size (or just check validity).
    We use:
    - T > 0: cost = T * 100
    - T == 0 and invalid: cost = 10000
    - T == 0 and valid: cost = 0
    """
    t = _count_triangles(adj, n)
    if t > 0:
        return t * 100
    else:
        if _has_ind_set_10(adj, n):
            return 10000
        else:
            return 0

def _crossover(adj1, adj2, n):
    """Uniform crossover on edge existence."""
    child = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            # Pick from parent 1 or 2
            if random.random() < 0.5:
                if adj1[i] & (1 << j):
                    child[i] |= (1 << j)
                    child[j] |= (1 << i)
            else:
                if adj2[i] & (1 << j):
                    child[i] |= (1 << j)
                    child[j] |= (1 << i)
    return child

def _mutate(adj, n, rate):
    """Flip random edges."""
    new_adj = adj[:]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < rate:
                # Flip edge
                if new_adj[i] & (1 << j):
                    new_adj[i] &= ~(1 << j)
                    new_adj[j] &= ~(1 << i)
                else:
                    new_adj[i] |= (1 << j)
                    new_adj[j] |= (1 << i)
    return new_adj

def _random_triangle_free_graph(n):
    """Generate a random triangle-free graph."""
    adj = [0] * n
    # Try adding edges in random order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(edges)
    
    for u, v in edges:
        # Check if adding (u,v) creates a triangle
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        if common == 0:
            # Safe to add
            if random.random() < 0.5: # Add with some probability to vary density
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Known seed: n=26 Andrásfai graph
    n_seed = 26
    adj_seed = [0] * n_seed
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if (j - i) % n_seed % 3 == 1:
                adj_seed[i] |= (1 << j)
                adj_seed[j] |= (1 << i)
    
    best_result = {'n': n_seed, 'edges': [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if adj_seed[i] & (1 << j)]}
    
    # Try to improve n
    # R(3,10) <= 41, so max n is 40.
    # Try 40, then 39.
    
    for target_n in [40, 39]:
        if time.time() > deadline - 10:
            break
            
        pop_size = 30
        # Initialize population
        population = []
        
        # Add seed (padded? No, seed is n=26. We need n=target_n.
        # We can't just pad with isolated vertices because that increases independent set size.
        # So we start with random triangle-free graphs for target_n.
        # Maybe add a "blow-up" of the seed? 
        # For now, just random init.
        
        for _ in range(pop_size):
            if time.time() > deadline - 5:
                break
            population.append(_random_triangle_free_graph(target_n))
        
        # Evaluate initial fitness
        fitnesses = [_fitness(adj, target_n) for adj in population]
        
        # Check for valid solution in initial population
        for i, fit in enumerate(fitnesses):
            if fit == 0:
                adj = population[i]
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': target_n, 'edges': edges}
        
        # GA Loop
        generation = 0
        while time.time() < deadline - 5:
            generation += 1
            
            # Sort by fitness
            indexed_pop = sorted(range(pop_size), key=lambda i: fitnesses[i])
            
            # Check best
            if fitnesses[indexed_pop[0]] == 0:
                adj = population[indexed_pop[0]]
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': target_n, 'edges': edges}
            
            # Selection: Tournament Selection
            new_population = []
            
            # Keep best few (Elitism)
            elite_count = max(1, pop_size // 5)
            for i in range(elite_count):
                new_population.append(population[indexed_pop[i]])
            
            # Generate rest
            while len(new_population) < pop_size:
                # Pick two parents
                def tournament():
                    idx1 = random.randint(0, pop_size - 1)
                    idx2 = random.randint(0, pop_size - 1)
                    if fitnesses[idx1] < fitnesses[idx2]:
                        return idx1
                    else:
                        return idx2
                
                p1_idx = tournament()
                p2_idx = tournament()
                
                child = _crossover(population[p1_idx], population[p2_idx], target_n)
                
                # Mutation
                # Rate: 0.01 to 0.05
                child = _mutate(child, target_n, 0.02)
                
                new_population.append(child)
            
            population = new_population
            fitnesses = [_fitness(adj, target_n) for adj in population]
            
            # Periodic check for valid
            if generation % 5 == 0:
                for i, fit in enumerate(fitnesses):
                    if fit == 0:
                        adj = population[i]
                        edges = []
                        for u in range(target_n):
                            for v in range(u + 1, target_n):
                                if adj[u] & (1 << v):
                                    edges.append([u, v])
                        return {'n': target_n, 'edges': edges}

    # Fallback to seed
    return best_result
