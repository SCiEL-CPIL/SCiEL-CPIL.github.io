import time
import random

def _deadline_guard(start_time, limit=145.0):
    """Return True if we should stop (deadline exceeded)."""
    return time.time() - start_time > limit

def _count_triangles(adj, n):
    """Count triangles using bitset operations."""
    count = 0
    for i in range(n):
        ai = adj[i]
        j_mask = ai
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            # For triangle i-j-k, k must be in adj[i] & adj[j] and k > j to avoid double counting
            # Actually, let's just count each triangle 3 times (once per vertex) and divide by 3
            # Or count each triangle once by requiring i < j < k
            common = ai & adj[j]
            if common & (1 << i) == 0:  # k != i
                count += common.bit_count()
            j_mask ^= lsb
    return count // 3

def _has_triangle(adj, n):
    """Quick check if any triangle exists."""
    for i in range(n):
        ai = adj[i]
        j_mask = ai
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            if adj[j] & ai:
                return True
            j_mask ^= lsb
    return False

def _independence_number_upper_bound(adj, n):
    """
    Find an independent set using greedy heuristic.
    Returns the size of the found independent set.
    """
    remaining = (1 << n) - 1
    size = 0
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = -1
        best_deg = n + 1
        r = remaining
        while r:
            lsb = r & -r
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            r ^= lsb
        size += 1
        remaining &= ~adj[best_v] & ~(1 << best_v)
    return size

def _has_independent_set_10(adj, n):
    """
    Check if there is an independent set of size 10 using branch and bound.
    Returns True if such a set exists.
    """
    if n < 10:
        return False
        
    full_mask = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full_mask ^ adj[i] ^ (1 << i)

    def dfs(candidates, size):
        if size >= 10:
            return True
        if candidates == 0:
            return False
        if size + candidates.bit_count() < 10:
            return False
            
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        
        new_candidates = candidates & comp[v]
        if dfs(new_candidates, size + 1):
            return True
            
        new_candidates = candidates & ~v_bit
        if dfs(new_candidates, size):
            return True
            
        return False

    return dfs(full_mask, 0)

def _fitness(adj, n):
    """
    Fitness function: lower is better.
    - Large penalty for triangles
    - Large penalty for independent set of size 10
    - Small penalty for independence number (to encourage lower alpha)
    """
    # Count triangles
    tri_count = 0
    for i in range(n):
        ai = adj[i]
        j_mask = ai
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            if adj[j] & ai:
                tri_count += 1
            j_mask ^= lsb
    tri_count //= 3  # Each triangle counted 3 times

    # Check for independent set of size 10
    has_alpha_10 = _has_independent_set_10(adj, n)

    # Greedy independence number estimate
    alpha_est = _independence_number_upper_bound(adj, n)

    # Fitness: minimize triangles, avoid alpha >= 10, minimize alpha
    fit = 1000 * tri_count + 10000 * (1 if has_alpha_10 else 0) + alpha_est
    return fit

def _is_valid(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    if _has_triangle(adj, n):
        return False
    if _has_independent_set_10(adj, n):
        return False
    return True

def _edges_to_adj(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        mask = adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
    return edges

def _random_adj(n, p=0.3):
    """Generate a random adjacency matrix."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _crossover(adj1, adj2, n):
    """Uniform crossover: for each edge, randomly pick from parent1 or parent2."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            bit1 = (adj1[i] >> j) & 1
            bit2 = (adj2[i] >> j) & 1
            if random.random() < 0.5:
                bit = bit1
            else:
                bit = bit2
            if bit:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _mutate(adj, n, mutation_rate=0.02):
    """Flip random edges with given probability."""
    adj = adj[:]  # Copy
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < mutation_rate:
                if adj[i] & (1 << j):
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj

def _tournament_select(population, fitnesses, k=3):
    """Tournament selection."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    best = min(candidates, key=lambda i: fitnesses[i])
    return population[best][:]  # Copy

def _genetic_algorithm(n, deadline):
    """
    Run genetic algorithm to find a valid graph on n vertices.
    Returns edge list or None.
    """
    pop_size = 50
    max_generations = 200
    
    # Initialize population
    population = [_random_adj(n, p=0.3) for _ in range(pop_size)]
    
    # Evaluate initial fitness
    fitnesses = [_fitness(adj, n) for adj in population]
    
    best_adj = None
    best_fit = float('inf')
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Track best
        for i in range(pop_size):
            if fitnesses[i] < best_fit:
                best_fit = fitnesses[i]
                best_adj = population[i][:]
        
        # Check if we found a valid solution
        if best_fit == 0:  # No triangles, no alpha-10, alpha <= 9
            if best_adj is not None:
                return _adj_to_edges(best_adj, n)
        
        # Create next generation
        next_pop = []
        
        # Elitism: keep best individual
        best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        next_pop.append(population[best_idx][:])
        
        # Fill rest with crossover + mutation
        while len(next_pop) < pop_size:
            if time.time() > deadline:
                break
            p1 = _tournament_select(population, fitnesses)
            p2 = _tournament_select(population, fitnesses)
            child = _crossover(p1, p2, n)
            child = _mutate(child, n, mutation_rate=0.03)
            next_pop.append(child)
        
        population = next_pop
        fitnesses = [_fitness(adj, n) for adj in population]
    
    # Check best found
    if best_adj is not None and best_fit == 0:
        return _adj_to_edges(best_adj, n)
    
    return None

def _local_search(adj, n, deadline):
    """
    Local search: try to remove triangles and break independent sets.
    Returns improved adj or None.
    """
    current = adj[:]
    current_fit = _fitness(current, n)
    
    for step in range(1000):
        if time.time() > deadline:
            break
        
        if current_fit == 0:
            return current
        
        # Try flipping a random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        # Flip edge (i, j)
        if current[i] & (1 << j):
            current[i] ^= (1 << j)
            current[j] ^= (1 << i)
        else:
            current[i] |= (1 << j)
            current[j] |= (1 << i)
        
        new_fit = _fitness(current, n)
        
        if new_fit < current_fit:
            current_fit = new_fit
        else:
            # Revert
            if current[i] & (1 << j):
                current[i] ^= (1 << j)
                current[j] ^= (1 << i)
            else:
                current[i] |= (1 << j)
                current[j] |= (1 << i)
    
    if current_fit == 0:
        return current
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try n=40 first (optimal if exists)
    t0 = time.time()
    edges_40 = _genetic_algorithm(40, t0 + 60)
    if edges_40:
        return {'n': 40, 'edges': edges_40}
    
    # Try n=39
    t1 = time.time()
    edges_39 = _genetic_algorithm(39, t1 + 50)
    if edges_39:
        return {'n': 39, 'edges': edges_39}
    
    # Try n=38
    t2 = time.time()
    edges_38 = _genetic_algorithm(38, t2 + 30)
    if edges_38:
        return {'n': 38, 'edges': edges_38}
    
    # Fallback: try smaller n with local search
    for n in range(37, 25, -1):
        if time.time() > deadline:
            break
        adj = _random_adj(n, p=0.3)
        adj = _local_search(adj, n, deadline)
        if adj is not None:
            fit = _fitness(adj, n)
            if fit == 0:
                return {'n': n, 'edges': _adj_to_edges(adj, n)}
    
    # Absolute fallback: seed graph (n=26)
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Result n={res['n']}")
    if _is_valid(res['n'], res['edges']):
        print("Valid")
    else:
        print("INVALID")
