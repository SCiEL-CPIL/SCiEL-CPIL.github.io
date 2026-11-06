import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    # Parameters
    CLIQUE_SIZE = 5
    INDEPENDENT_SIZE = 5
    MAX_VERTICES = 46  # Upper bound from Angeltveit and McKay
    TARGET_N = 42      # Try to find graphs up to this size
    
    # Helper: Check if a graph is valid (no K5, no independent 5)
    # graph is a list of sets or bitmasks. Let's use bitmasks for speed.
    # adj[i] is a bitmask where bit j is 1 if i~j.
    
    def is_valid_graph(n, adj):
        """
        Check if graph on n vertices with adjacency bitmasks 'adj' 
        has no clique of size 5 and no independent set of size 5.
        """
        # Precompute all 5-combinations of vertices
        # There are C(n, 5) combinations. For n=42, C(42,5) = 850668.
        # This is too many to check naively for every generation if we do it in Python loops.
        # We need a faster check or smarter search.
        
        # Let's try a randomized check first for speed, then exact if needed?
        # No, we must be exact.
        
        # Optimization: 
        # A clique of size 5 exists if there are 5 vertices all connected.
        # An independent set of size 5 exists if there are 5 vertices with no edges between them.
        
        # We can check for K5 by looking for K4s and extending?
        # Or just iterate 5-subsets. 
        # 850k subsets * 42 generations * ... might be slow in pure Python.
        
        # Let's use a faster method: 
        # For each vertex v, let N(v) be its neighbors.
        # If there is a K5, it contains a K4. 
        # This is still complex.
        
        # Let's stick to the 5-subset check but optimize the inner loop.
        # Actually, for n=38-42, C(n,5) is around 500k-850k.
        # Checking each subset takes O(1) with bitmasks if we precompute?
        # Check if subset S is a clique:
        # For all pairs (u,v) in S, is u~v?
        # This is O(10) checks per subset. 8.5M checks per validation.
        # In Python, 8.5M operations is ~0.5-1 second.
        # If we do 100 validations, that's 50-100 seconds. Feasible.
        
        vertices = list(range(n))
        
        # Check for K5
        for subset in itertools.combinations(vertices, CLIQUE_SIZE):
            is_clique = True
            # Check all pairs in subset
            for i in range(CLIQUE_SIZE):
                u = subset[i]
                for j in range(i+1, CLIQUE_SIZE):
                    v = subset[j]
                    if not (adj[u] >> v) & 1:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
                
        # Check for Independent Set of Size 5
        for subset in itertools.combinations(vertices, INDEPENDENT_SIZE):
            is_independent = True
            for i in range(INDEPENDENT_SIZE):
                u = subset[i]
                for j in range(i+1, INDEPENDENT_SIZE):
                    v = subset[j]
                    if (adj[u] >> v) & 1:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
                
        return True

    # Helper: Convert edge list to adj bitmasks
    def edges_to_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Helper: Convert adj bitmasks to edge list
    def adj_to_edges(n, adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            # Only consider v > u to avoid duplicates
            for v in range(u+1, n):
                if (mask >> v) & 1:
                    edges.append((u, v))
        return edges

    # Initialize with Paley Graph (37 vertices)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    paley_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_n = q
    best_edges = paley_edges
    best_adj = edges_to_adj(q, paley_edges)
    
    # Verify seed
    if not is_valid_graph(q, best_adj):
        # Should not happen, but if it does, return empty or minimal
        pass

    # Genetic Algorithm Setup
    population_size = 20
    population = []
    
    # Add seed to population
    population.append((q, best_adj))
    
    # Add some random graphs of size 38-42
    for _ in range(population_size - 1):
        n = random.randint(38, min(42, MAX_VERTICES))
        # Random graph with p=0.5
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edges.append((i, j))
        adj = edges_to_adj(n, edges)
        # Only add if valid to keep population clean? 
        # Or add anyway and filter later?
        # Let's check validity immediately for initial population.
        if is_valid_graph(n, adj):
            population.append((n, adj))
            if n > best_n:
                best_n = n
                best_adj = adj

    # If we didn't find any valid random graphs, we might need to be smarter.
    # But let's proceed with the loop.
    
    # If population is too small, pad with variations of best_adj
    while len(population) < population_size:
        # Mutate the best graph
        n = best_n
        adj = list(best_adj)
        # Randomly flip some edges
        for _ in range(10):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        if is_valid_graph(n, adj):
            population.append((n, adj))

    # Evolution Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        offspring = []
        
        for _ in range(population_size // 2):
            # Pick two parents
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # If sizes differ, we can't easily crossover.
            # Strategy: Try to grow the smaller one or just mutate the larger one.
            # Let's focus on mutation of the best known valid graph and random restarts.
            
            # Mutation of a random parent
            parent_n, parent_adj = random.choice(population)
            
            # Try to increase size?
            # If parent_n < 42, try to add a vertex.
            # This is hard because we need to connect the new vertex correctly.
            
            # Simpler: Just mutate edges of a valid graph.
            new_adj = list(parent_adj)
            new_n = parent_n
            
            # Random edge flips
            num_flips = random.randint(1, 5)
            for _ in range(num_flips):
                u = random.randint(0, new_n-1)
                v = random.randint(0, new_n-1)
                if u != v:
                    if (new_adj[u] >> v) & 1:
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
            
            if is_valid_graph(new_n, new_adj):
                offspring.append((new_n, new_adj))
                if new_n > best_n:
                    best_n = new_n
                    best_adj = new_adj

        # If we found a larger best, we might want to explore around it.
        
        # Selection: Keep top population_size by size (then random)
        # Since all in population are valid, we prefer larger ones.
        # But we also want diversity.
        # Let's just keep a mix.
        
        # Replace population with offspring + some survivors
        # To avoid losing good large graphs, keep best_adj in population.
        if (best_n, best_adj) not in offspring:
            offspring.append((best_n, best_adj))
            
        # Pad offspring with random valid graphs if needed
        while len(offspring) < population_size:
             # Generate a random valid graph? Hard.
             # Just reuse existing valid ones from population
             if population:
                 offspring.append(random.choice(population))
             else:
                 offspring.append((best_n, best_adj))
                 
        population = offspring
        
        # Occasionally, try to construct a new random graph of a larger size
        if generation % 10 == 0:
             target_size = min(best_n + 1, 42)
             for _ in range(5): # Try 5 times
                 n = target_size
                 edges = []
                 for i in range(n):
                     for j in range(i+1, n):
                         if random.random() < 0.5:
                             edges.append((i, j))
                 adj = edges_to_adj(n, edges)
                 if is_valid_graph(n, adj):
                     population.append((n, adj))
                     if n > best_n:
                         best_n = n
                         best_adj = adj
                     break

    # Convert best_adj to edge list
    final_edges = adj_to_edges(best_n, best_adj)
    return best_n, final_edges

# EVOLVE-BLOCK-END
