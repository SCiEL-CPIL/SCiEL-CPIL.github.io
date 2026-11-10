import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    
    def time_ok():
        return time.time() < deadline

    # Precompute all possible edges as indices in a flattened upper-triangular matrix
    edge_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices.append((i, j))
    m_edges = len(edge_indices)
    
    # Map edge index to (u, v)
    edge_map = {idx: edge_indices[idx] for idx in range(m_edges)}
    
    # Precompute for each edge index, the pair (u, v)
    # We will represent the graph as a numpy array of 0/1 of length m_edges
    # But for fast K3,3 checking, we need adjacency bitmasks.
    # Let's stick to bitmasks for the core logic but use the GA on the bitmask directly.
    # Bitmask is an integer of 20 bits. A population is a list of such integers.
    
    # Fitness function:
    # Returns (is_valid, edge_count, penalty)
    # We want to maximize edge_count if valid.
    # If invalid, we penalize heavily to guide repair.
    # Penalty = number of violating triples * 1000 (arbitrary large number)
    # Actually, for GA, we want a continuous-ish signal if possible, 
    # but "valid" is a hard constraint. 
    # Let's define fitness = edge_count * 1000 - violation_count * 10000.
    # This prioritizes validity, then edge count.
    
    def calculate_fitness(adj):
        """
        adj: list of n integers (bitmasks)
        Returns: fitness score (higher is better)
        """
        # Count edges
        edges = 0
        for i in range(n):
            edges += adj[i].bit_count()
        edges //= 2
        
        # Count violations (triples with >= 3 common neighbors)
        violations = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                if aij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        violations += 1
        
        # If valid, high score. If invalid, lower score proportional to violations.
        # We want valid graphs to always beat invalid ones.
        # Max possible edges is 190. 
        # Let's say score = edges if valid, else -violations.
        # This might be too jagged. 
        # Better: score = edges * 10 - violations * 100.
        # If valid: score = edges * 10.
        # If invalid: score = edges * 10 - violations * 100.
        # A valid graph with 100 edges: 1000.
        # An invalid graph with 150 edges and 1 violation: 1500 - 100 = 1400. 
        # This is bad! Invalid should be worse.
        
        # Correct logic: 
        # Primary goal: Validity. Secondary: Edge count.
        # Score = (1 if valid else 0) * 10000 + edges - violations * 100
        # This ensures any valid graph (min score ~10000) beats any invalid graph (max score < 10000).
        # Among valid graphs, higher edges win.
        # Among invalid graphs, fewer violations and higher edges win (to guide repair).
        
        if violations == 0:
            return 10000 + edges
        else:
            return edges - violations * 100

    # Initialize Population
    pop_size = 50
    population = []
    
    # Seed 1: Paley-like or Random Dense
    def random_graph(density=0.6):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Seed 2: Complete graph (will be heavily penalized, but provides high edge count signal)
    # Actually, complete graph has massive violations.
    # Let's use a mix of random densities.
    
    for i in range(pop_size):
        # Vary density to explore different regions
        density = 0.4 + 0.4 * (i / pop_size)
        population.append(random_graph(density))

    # Evaluate initial population
    fitnesses = [calculate_fitness(adj) for adj in population]
    
    # Helper to select parents (Tournament Selection)
    def select_parent(fitnesses):
        # Tournament of 5
        contenders = random.sample(range(len(fitnesses)), 5)
        best_idx = max(contenders, key=lambda x: fitnesses[x])
        return best_idx

    # Crossover: Uniform crossover on bitmasks
    def crossover(parent1, parent2):
        # Each vertex's adjacency mask is mixed
        child = [0] * n
        for i in range(n):
            # For each vertex, pick the adjacency mask from p1 or p2 randomly
            # But we must ensure symmetry. 
            # If we pick mask for i from p1, we must ensure the bits corresponding to i are consistent in the other vertices.
            # Simplest valid crossover for bitmasks: 
            # Create a random mask of which vertices' "ownership" comes from p1 vs p2?
            # No, adjacency is pairwise.
            # Let's do edge-based crossover.
            pass
            
    # Since bitmasks are symmetric, edge-based crossover is safer.
    # Let's convert to edge arrays for crossover, or just do random edge swapping?
    # Actually, a simple "Uniform Crossover" on the bitset of edges is easy.
    # Represent graph as a single integer? No, 190 bits is too big for standard int ops in some contexts, but Python handles big ints.
    # Let's represent the graph as a single Python integer where bit k corresponds to edge_indices[k].
    
    def adj_to_int(adj):
        g = 0
        for idx, (u, v) in enumerate(edge_indices):
            if adj[u] & (1 << v):
                g |= (1 << idx)
        return g

    def int_to_adj(g):
        adj = [0] * n
        for idx, (u, v) in enumerate(edge_indices):
            if g & (1 << idx):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Convert population to int representation for fast ops
    pop_ints = [adj_to_int(adj) for adj in population]
    
    # Re-evaluate fitnesses on int representation? 
    # calculate_fitness takes adj list. Let's make a version that takes int.
    def calculate_fitness_int(g):
        adj = int_to_adj(g)
        return calculate_fitness(adj)

    fitnesses = [calculate_fitness_int(g) for g in pop_ints]
    
    # GA Loop
    max_generations = 200
    mutation_rate = 0.05
    
    for gen in range(max_generations):
        if not time_ok():
            break
            
        new_population = []
        
        # Elitism: keep the best 5
        sorted_indices = sorted(range(len(pop_ints)), key=lambda x: fitnesses[x], reverse=True)
        for i in range(min(5, len(pop_ints))):
            new_population.append(pop_ints[sorted_indices[i]])
            
        while len(new_population) < pop_size:
            if not time_ok():
                break
                
            p1_idx = select_parent(fitnesses)
            p2_idx = select_parent(fitnesses)
            
            p1 = pop_ints[p1_idx]
            p2 = pop_ints[p2_idx]
            
            # Uniform Crossover
            # Create a random mask of 190 bits
            # We can generate a random integer of 190 bits
            mask = 0
            for i in range(m_edges):
                if random.random() < 0.5:
                    mask |= (1 << i)
            
            child = (p1 & mask) | (p2 & ~mask)
            
            # Mutation
            # Flip some edges
            if random.random() < mutation_rate:
                # Flip 1-5 edges
                num_flips = random.randint(1, 5)
                for _ in range(num_flips):
                    edge_idx = random.randint(0, m_edges - 1)
                    child ^= (1 << edge_idx)
            
            new_population.append(child)
            
        # Evaluate new population
        pop_ints = new_population
        fitnesses = [calculate_fitness_int(g) for g in pop_ints]
        
        # Check if we found a valid graph with high edge count
        # We track the best valid graph found so far
        best_valid_score = -1
        best_valid_adj = None
        
        for i, g in enumerate(pop_ints):
            if fitnesses[i] >= 10000: # Valid
                score = fitnesses[i] - 10000
                if score > best_valid_score:
                    best_valid_score = score
                    best_valid_adj = int_to_adj(g)
                    
        # Optional: Local Search on the best individual every few generations?
        # The directive says "Genetic Algorithm". Pure GA might be slow to converge to optimal.
        # But let's stick to GA first. If we have time, we can add a "local repair" step for the elite.
        
        # Let's add a simple "Repair" step for the best individual in the population
        # to help it become valid.
        best_idx = max(range(len(fitnesses)), key=lambda x: fitnesses[x])
        best_g = pop_ints[best_idx]
        
        if fitnesses[best_idx] < 10000: # If best is invalid, try to repair it
            adj = int_to_adj(best_g)
            # Simple greedy repair: remove edges causing most violations
            # This is a hybrid, but helps GA significantly.
            for _ in range(10): # Limited repair steps
                if not time_ok():
                    break
                viols = []
                for i in range(n):
                    ai = adj[i]
                    for j in range(i + 1, n):
                        aij = ai & adj[j]
                        if aij.bit_count() < 3: continue
                        for k in range(j + 1, n):
                            if (aij & adj[k]).bit_count() >= 3:
                                viols.append((i,j,k))
                if not viols:
                    break
                
                # Find edge in most violations
                edge_counts = {}
                for i,j,k in viols:
                    cn = adj[i] & adj[j] & adj[k]
                    for w in range(n):
                        if cn & (1 << w):
                            for x in [i,j,k]:
                                if x != w:
                                    key = (min(x,w), max(x,w))
                                    edge_counts[key] = edge_counts.get(key, 0) + 1
                if not edge_counts:
                    break
                worst_edge = max(edge_counts, key=edge_counts.get)
                u,v = worst_edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
            # Update best individual in population with repaired version
            pop_ints[best_idx] = adj_to_int(adj)
            fitnesses[best_idx] = calculate_fitness(adj)

    # Find the best valid graph in the final population
    best_adj = None
    best_edges = -1
    
    for g in pop_ints:
        adj = int_to_adj(g)
        edges = 0
        for i in range(n):
            edges += adj[i].bit_count()
        edges //= 2
        
        # Check validity
        is_valid = True
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                if aij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        if is_valid and edges > best_edges:
            best_edges = edges
            best_adj = adj

    # Fallback: If no valid graph found (unlikely), generate a safe one
    if best_adj is None:
        # Use a simple known valid graph (e.g., sparse random)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.2:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Ensure validity
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                aij = ai & adj[j]
                if aij.bit_count() >= 3:
                    # Remove one common neighbor
                    k = aij.bit_length() - 1
                    adj[i] &= ~(1 << k)
                    adj[k] &= ~(1 << i)
                    adj[j] &= ~(1 << k)
                    adj[k] &= ~(1 << j)
        best_adj = adj

    # Convert to edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return edge_list
