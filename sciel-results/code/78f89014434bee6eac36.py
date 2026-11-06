import time
import random

_DEADLINE = 120.0
_MARGIN = 8.0

def _elapsed(start):
    return time.time() - start

def _deadline_guard(start):
    return _elapsed(start) > (_DEADLINE - _MARGIN)

def _build_adj(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _count_triangles(adj, n):
    """Count triangles using bitsets."""
    count = 0
    for i in range(n):
        ai = adj[i]
        j = 0
        while j < n:
            if (ai >> j) & 1 and j > i:
                common = adj[j] & ai
                # Only count triangles where k > j to avoid double counting
                # Actually, let's just count all and divide by 3
                count += common.bit_count()
            j += 1
    return count // 3  # Each triangle counted 3 times

def _has_independent_set_10(adj, n):
    """Check if there's an independent set of size >= 10 using branch and bound."""
    full_mask = (1 << n) - 1
    # Complement adjacency
    comp = [0] * n
    for i in range(n):
        comp[i] = full_mask ^ adj[i] ^ (1 << i)
    
    def dfs(candidates, depth):
        if depth >= 10:
            return True
        if not candidates:
            return False
        if depth + candidates.bit_count() < 10:
            return False
        # Pick vertex with max degree in complement within candidates
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            c ^= lb
            deg = (comp[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
        if best_v < 0:
            return False
        v_bit = 1 << best_v
        # Include
        if dfs(candidates & comp[best_v] & ~v_bit, depth + 1):
            return True
        # Exclude
        if dfs(candidates & ~v_bit, depth):
            return True
        return False
    
    return dfs(full_mask, 0)

def _fitness(adj, n, target_n=40):
    """
    Fitness: higher is better.
    Penalize triangles heavily, penalize independent set size.
    """
    tri = _count_triangles(adj, n)
    if tri > 0:
        return -1000 - tri * 10
    
    # Check independent set
    if _has_independent_set_10(adj, n):
        # Estimate independence number roughly
        # Just return a penalty
        return -100 - n
    
    # Valid! Score based on n
    return n * 100
    
def _make_andrasfai(k, n):
    """Create an Andrásfai-type graph on n vertices."""
    adj = [0] * n
    # Andrásfai graph And(k) has k^2 vertices
    # For n vertices, use a cyclic construction
    # i ~ j if (j - i) mod n is in {1, 2, ..., k//2} (approximately)
    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            j1 = (i + d) % n
            j2 = (i - d) % n
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
    return adj

def _random_graph(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _remove_triangles(adj, n):
    """Remove edges to make graph triangle-free."""
    adj = adj[:]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            ai = adj[i]
            j = 0
            while j < n:
                if (ai >> j) & 1 and j > i:
                    common = adj[j] & ai
                    if common:
                        # Remove edge (i, j)
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        changed = True
                        ai = adj[i]  # Update
                j += 1
    return adj

def _add_edge_safe(adj, n, u, v):
    """Add edge (u,v) if it doesn't create a triangle."""
    if adj[u] & (1 << v):
        return False
    # Check if adding (u,v) creates a triangle
    common = adj[u] & adj[v]
    if common:
        return False
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    return True

def _local_improve(adj, n, start):
    """Try to improve the graph by adding edges to break independent sets."""
    adj = adj[:]
    for step in range(200):
        if _deadline_guard(start):
            break
        if not _has_independent_set_10(adj, n):
            break
        # Find an independent set of size 10 (or as large as possible)
        # Try to add edges between non-adjacent pairs that don't create triangles
        improved = False
        random.shuffle(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if _add_edge_safe(adj, n, u, v):
                        improved = True
                        if not _has_independent_set_10(adj, n):
                            return adj
                        break
            if improved:
                break
        if not improved:
            break
    return adj

def construct():
    start = time.time()
    random.seed(12345)
    
    best_n = 0
    best_edges = []
    best_adj = None
    
    # Start with a structured graph
    # Andrásfai graph on 36 vertices (6x6)
    n_init = 36
    adj_init = _make_andrasfai(6, n_init)
    adj_init = _remove_triangles(adj_init, n_init)
    
    # Check if valid
    if not _has_independent_set_10(adj_init, n_init):
        best_n = n_init
        best_adj = adj_init[:]
        best_edges = [[i, j] for i in range(n_init) for j in range(i + 1, n_init) if (adj_init[i] >> j) & 1]
    
    # Genetic algorithm
    pop_size = 40
    elite_size = 5
    mutation_rate = 0.15
    crossover_rate = 0.7
    
    # Initialize population
    population = []
    
    # Add the initial structured graph
    population.append(adj_init[:])
    
    # Add random graphs
    for _ in range(pop_size - 1):
        if _deadline_guard(start):
            break
        n = random.choice([38, 39, 40, 41])
        p = random.uniform(0.2, 0.4)
        adj = _random_graph(n, p)
        adj = _remove_triangles(adj, n)
        population.append(adj)
    
    # GA loop
    generation = 0
    while not _deadline_guard(start) and generation < 50:
        generation += 1
        
        # Evaluate fitness
        fitnesses = []
        for adj in population:
            n = len(adj)
            f = _fitness(adj, n)
            fitnesses.append(f)
        
        # Update best
        for i, f in enumerate(fitnesses):
            if f > 0:  # Valid graph
                n = len(population[i])
                if n > best_n:
                    best_n = n
                    best_adj = population[i][:]
                    best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
        
        # Selection: tournament
        new_population = []
        
        # Elitism
        sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        for i in range(min(elite_size, pop_size)):
            new_population.append(population[sorted_idx[i]][:])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if _deadline_guard(start):
                break
            
            # Tournament selection
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if fitnesses[idx1] >= fitnesses[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            idx3 = random.randint(0, pop_size - 1)
            idx4 = random.randint(0, pop_size - 1)
            if fitnesses[idx3] >= fitnesses[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]
            
            # Uniform crossover
            if random.random() < crossover_rate and len(parent1) == len(parent2):
                child = [0] * len(parent1)
                n = len(parent1)
                for i in range(n):
                    for j in range(i + 1, n):
                        if (parent1[i] >> j) & 1:
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
                        elif (parent2[i] >> j) & 1:
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
            else:
                child = parent1[:]
            
            # Mutation: flip some edges
            n = len(child)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < mutation_rate:
                        if (child[i] >> j) & 1:
                            child[i] &= ~(1 << j)
                            child[j] &= ~(1 << i)
                        else:
                            # Try to add edge safely
                            common = child[i] & child[j]
                            if not common:
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
            
            # Remove any triangles
            child = _remove_triangles(child, n)
            
            # Local improvement
            if not _deadline_guard(start):
                child = _local_improve(child, n, start)
            
            new_population.append(child)
        
        population = new_population
    
    # Final attempt: try to extend the best graph
    if best_adj is not None and not _deadline_guard(start):
        n = best_n
        # Try to add vertices
        for target_n in range(n + 1, 42):
            if _deadline_guard(start):
                break
            adj_ext = [0] * target_n
            # Copy existing edges
            for i in range(n):
                adj_ext[i] = best_adj[i]
            # Add new vertices with random connections
            for i in range(n, target_n):
                for j in range(target_n):
                    if i != j and random.random() < 0.3:
                        # Check if adding edge creates triangle
                        if i < n:
                            common = adj_ext[i] & adj_ext[j] if j < n else 0
                        else:
                            common = 0
                        if not common:
                            adj_ext[i] |= (1 << j)
                            adj_ext[j] |= (1 << i)
            adj_ext = _remove_triangles(adj_ext, target_n)
            adj_ext = _local_improve(adj_ext, target_n, start)
            if not _has_independent_set_10(adj_ext, target_n):
                best_n = target_n
                best_adj = adj_ext
                best_edges = [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if (best_adj[u] >> v) & 1]
    
    if best_adj is None:
        # Fallback: return empty graph (n=1)
        return {'n': 1, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
