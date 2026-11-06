"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_triangle_free(n, adj):
    """Check if graph with adjacency matrix is triangle-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _max_independent_set_size(n, adj):
    """Exact branch and bound for maximum independent set size."""
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    best = 0
    
    def branch(candidates_mask, size):
        nonlocal best
        if size + bin(candidates_mask).count('1') <= best:
            return
        if not candidates_mask:
            if size > best:
                best = size
            return
        
        # Pick vertex with max degree in candidates
        max_deg = -1
        chosen = -1
        cm = candidates_mask
        v_idx = 0
        while cm:
            if cm & 1:
                deg = bin(adj_mask[v_idx] & candidates_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v_idx
            cm >>= 1
            v_idx += 1
            
        if chosen == -1:
            return

        # Branch 1: Include chosen
        new_cand = candidates_mask & ~(adj_mask[chosen] | (1 << chosen))
        branch(new_cand, size + 1)
        
        # Branch 2: Exclude chosen
        new_cand = candidates_mask & ~(1 << chosen)
        branch(new_cand, size)

    branch((1 << n) - 1, 0)
    return best

def _fitness(n, adj):
    """
    Fitness function: higher is better.
    Penalize triangles heavily, then penalize large independent sets.
    """
    if not _is_triangle_free(n, adj):
        # Count triangles for gradient
        tri_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            tri_count += 1
        return -1000 - tri_count
    
    alpha = _max_independent_set_size(n, adj)
    if alpha < 10:
        return 1000 + n * 100 - alpha
    else:
        return -100 - (alpha - 10) * 50

def _crossover(parent1, parent2, n):
    """Uniform crossover."""
    child = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                child[i][j] = parent1[i][j]
                child[j][i] = parent1[i][j]
            else:
                child[i][j] = parent2[i][j]
                child[j][i] = parent2[i][j]
    return child

def _mutate(adj, n, rate=0.05):
    """Random mutation: flip some edges."""
    child = [row[:] for row in adj]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < rate:
                child[i][j] = 1 - child[i][j]
                child[j][i] = child[i][j]
    return child

def _tournament_select(population, fitnesses, k=3):
    """Tournament selection."""
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Fallback: Andrásfai graph And(3) on 9 vertices, or use a known valid graph
    # Let's use a simple valid graph: C5 blow-up or just return a small valid graph
    # For now, use n=9 with no edges (independent set = 9 < 10, no triangles)
    # Actually, let's use a better fallback: complete bipartite K_{4,4} on 8 vertices
    # No triangles, alpha = 4 < 10
    n_seed = 8
    best_n = n_seed
    best_edges = [[i, j] for i in range(4) for j in range(4, 8)]
    
    # Genetic algorithm to find larger graphs
    for target_n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if time.time() > deadline - 30:
            break
        
        n = target_n
        pop_size = 50
        max_generations = 200
        
        # Initialize population
        population = []
        fitnesses = []
        
        # Seed with some structured graphs
        # 1. Random sparse
        for _ in range(pop_size // 2):
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:
                        adj[i][j] = 1
                        adj[j][i] = 1
            population.append(adj)
        
        # 2. Structured: try to make bipartite-like
        for _ in range(pop_size // 4):
            adj = [[0] * n for _ in range(n)]
            # Random bipartition
            left = set(random.sample(range(n), n // 2))
            for i in left:
                for j in left:
                    if i < j and random.random() < 0.4:
                        adj[i][j] = 1
                        adj[j][i] = 1
            population.append(adj)
        
        # 3. Random dense
        for _ in range(pop_size // 4):
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 1
            population.append(adj)
        
        # Evaluate initial population
        for adj in population:
            fitnesses.append(_fitness(n, adj))
        
        best_fitness = max(fitnesses)
        best_adj = population[fitnesses.index(best_fitness)]
        
        # Evolution
        gen = 0
        while gen < max_generations:
            if time.time() > deadline - 10:
                break
            
            gen += 1
            
            # Select and create offspring
            new_population = []
            
            # Elitism: keep top 2
            sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            for idx in sorted_indices[:2]:
                new_population.append(population[idx])
            
            # Fill rest with offspring
            while len(new_population) < pop_size:
                p1 = _tournament_select(population, fitnesses)
                p2 = _tournament_select(population, fitnesses)
                child = _crossover(p1, p2, n)
                child = _mutate(child, n, rate=0.08)
                new_population.append(child)
            
            population = new_population
            fitnesses = []
            
            # Evaluate new population
            for adj in population:
                f = _fitness(n, adj)
                fitnesses.append(f)
            
            current_best = max(fitnesses)
            if current_best > best_fitness:
                best_fitness = current_best
                best_adj = population[fitnesses.index(current_best)]
            
            # Check if we found a valid graph
            if best_fitness > 1000:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_adj[i][j]:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break
        
        # If we found a valid graph for this n, we can stop trying smaller n
        if best_n >= target_n:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
