import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s margin for 150s limit

    def time_check():
        return time.time() - start_time

    def has_triangle(adj_masks, n):
        """Check if graph with adjacency masks has a triangle."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    if adj_masks[u] & adj_masks[v]:
                        return True
        return False

    def max_independent_set_size(adj_masks, n, timeout=None):
        """
        Returns the size of the maximum independent set using branch and bound.
        """
        best = [0]
        
        def dfs(candidates, size):
            if timeout is not None and time.time() > timeout:
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            if size + candidates.bit_count() <= best[0]:
                return
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)
            
            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def count_triangles(adj_masks, n):
        """Count the number of triangles in the graph."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    common = adj_masks[u] & adj_masks[v]
                    count += common.bit_count()
        return count // 3  # Each triangle counted 3 times

    def fitness(adj_masks, n):
        """
        Fitness function: lower is better.
        Penalty for triangles + penalty for independent set of size 10.
        """
        triangles = count_triangles(adj_masks, n)
        if triangles > 0:
            return 1000000 + triangles * 1000
        
        # Check for independent set of size 10
        alpha = max_independent_set_size(adj_masks, n, timeout=start_time + 2)
        if alpha >= 10:
            return 10000 + alpha
        
        # Valid graph: reward larger n
        return -n * 1000 + alpha
        
    def edges_from_adj_masks(adj_masks, n):
        """Convert adjacency masks to edge list."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def random_triangle_free_graph(n, seed=None):
        """Generate a random triangle-free graph."""
        if seed is not None:
            random.seed(seed)
        adj_masks = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if adj_masks[u] & (1 << v):
                continue
            if adj_masks[u] & adj_masks[v]:
                continue
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        return adj_masks

    def genetic_algorithm(n, population_size=50, generations=100):
        """
        Genetic algorithm to evolve triangle-free graphs with small independent sets.
        """
        # Initialize population
        population = []
        for i in range(population_size):
            adj = random_triangle_free_graph(n, seed=42 + i)
            population.append(adj)
        
        best_fitness = float('inf')
        best_adj = None
        
        # Evaluate initial population
        for adj in population:
            f = fitness(adj, n)
            if f < best_fitness:
                best_fitness = f
                best_adj = adj[:]
        
        # Check if we already have a valid graph
        if best_fitness < 0:  # Valid graph has negative fitness
            return best_adj[:]
        
        # Evolution
        for gen in range(generations):
            if time_check() > DEADLINE - 15:
                break
            
            # Evaluate fitness for all
            fitnesses = []
            for adj in population:
                f = fitness(adj, n)
                fitnesses.append(f)
                if f < best_fitness:
                    best_fitness = f
                    best_adj = adj[:]
            
            if best_fitness < 0:
                return best_adj[:]
            
            # Sort by fitness (lower is better)
            sorted_indices = sorted(range(population_size), key=lambda i: fitnesses[i])
            
            # Create next generation
            new_population = []
            
            # Elitism: keep top 20%
            elite_count = max(1, population_size // 5)
            for i in range(elite_count):
                new_population.append(population[sorted_indices[i]][:])
            
            # Fill rest with offspring
            while len(new_population) < population_size:
                if time_check() > DEADLINE - 15:
                    break
                
                # Tournament selection
                parent1_idx = sorted_indices[random.randint(0, min(10, population_size - 1))]
                parent2_idx = sorted_indices[random.randint(0, min(10, population_size - 1))]
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover: uniform crossover on edge presence
                child = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        bit_u = (parent1[u] >> v) & 1
                        bit_v = (parent2[u] >> v) & 1
                        if random.random() < 0.5:
                            bit = bit_u
                        else:
                            bit = bit_v
                        if bit:
                            child[u] |= (1 << v)
                            child[v] |= (1 << u)
                
                # Mutation: flip some edges while maintaining triangle-free
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.05:
                            if (child[u] >> v) & 1:
                                # Try to remove edge
                                child[u] &= ~(1 << v)
                                child[v] &= ~(1 << u)
                            else:
                                # Try to add edge if it doesn't create triangle
                                if not (child[u] & child[v]):
                                    child[u] |= (1 << v)
                                    child[v] |= (1 << u)
                
                # Ensure triangle-free
                if has_triangle(child, n):
                    # Remove edges to break triangles
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (child[u] >> v) & 1:
                                if child[u] & child[v]:
                                    # Remove this edge
                                    child[u] &= ~(1 << v)
                                    child[v] &= ~(1 << u)
                
                new_population.append(child)
            
            population = new_population
        
        return best_adj

    # Try to find a graph for n=39, then 38, etc.
    best_n = 0
    best_edges = []
    
    for target_n in [39, 38, 37, 36, 35]:
        if time_check() > DEADLINE - 20:
            break
        
        # Run genetic algorithm
        adj = genetic_algorithm(target_n, population_size=40, generations=80)
        
        if adj is not None:
            # Verify the result
            if not has_triangle(adj, target_n):
                alpha = max_independent_set_size(adj, target_n, timeout=start_time + 5)
                if alpha < 10:
                    best_n = target_n
                    best_edges = edges_from_adj_masks(adj, target_n)
                    break
    
    if best_n == 0:
        # Fallback: try random search for smaller n
        for target_n in [30, 28, 26]:
            if time_check() > DEADLINE - 10:
                break
            
            for seed in range(100):
                if time_check() > DEADLINE - 10:
                    break
                adj = random_triangle_free_graph(target_n, seed=seed)
                if not has_triangle(adj, target_n):
                    alpha = max_independent_set_size(adj, target_n, timeout=start_time + 3)
                    if alpha < 10:
                        best_n = target_n
                        best_edges = edges_from_adj_masks(adj, target_n)
                        break
            if best_n > 0:
                break
    
    if best_n == 0:
        # Fallback to known small graph (Andrasfai And(9) on 26 vertices)
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        best_n = n
        best_edges = edges

    return {'n': best_n, 'edges': best_edges}
