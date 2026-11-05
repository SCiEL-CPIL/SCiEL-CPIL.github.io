import numpy as np
import time
import random

def _greedy_set_size_from_order(adj, order):
    """
    Run the fixed greedy decoder given a specific order of vertices.
    Returns the size of the independent set.
    """
    n = adj.shape[0]
    chosen = set()
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        v = int(v)
        if not blocked[v]:
            chosen.add(v)
            count += 1
            # Block v and its neighbors
            blocked[v] = True
            nbrs = np.nonzero(adj[v])[0]
            if len(nbrs) > 0:
                blocked[nbrs] = True
    return count

def _order_crossover(parent1, parent2, n):
    """
    Order Crossover (OX1) for permutations.
    """
    child = [-1] * n
    # Pick two cut points
    start, end = sorted(random.sample(range(n), 2))
    
    # Copy segment from parent1
    for i in range(start, end + 1):
        child[i] = parent1[i]
        
    # Fill remaining positions with elements from parent2 in order
    # that are not already in the child
    fill_values = [v for v in parent2 if v not in child[start:end+1]]
    
    # Start filling from end+1, wrapping around
    fill_idx = 0
    for i in range(end + 1, n):
        child[i] = fill_values[fill_idx]
        fill_idx += 1
    for i in range(0, start):
        child[i] = fill_values[fill_idx]
        fill_idx += 1
        
    return child

def _uniform_crossover(parent1, parent2, n):
    """
    Uniform Crossover for permutations.
    For each position, randomly pick from parent1 or parent2.
    If a value is already placed, skip and pick the next available from the other parent.
    """
    child = [-1] * n
    used = [False] * n
    
    # First pass: randomly assign
    for i in range(n):
        if random.random() < 0.5:
            val = parent1[i]
        else:
            val = parent2[i]
            
        if not used[val]:
            child[i] = val
            used[val] = True
        else:
            # Leave as -1 for now, will fill in second pass
            pass
            
    # Second pass: fill in missing values
    missing = [v for v in range(n) if not used[v]]
    missing_idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = missing[missing_idx]
            missing_idx += 1
            
    return child

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute degrees for initial heuristics
    deg = adj.sum(axis=1)
    
    # Define deadline with safety margin
    deadline = start_time + 140.0
    
    # Population size
    pop_size = 60
    elite_size = 5
    
    # Initialize Population
    population = []
    fitnesses = []
    
    # Strategy 1: Degree-based (low degree first)
    keys = np.random.rand(n)
    order_deg = np.lexsort((keys, deg))
    population.append(order_deg)
    
    # Strategy 2: High degree first (sometimes better to block hubs early? No, usually low degree is better for MIS, but let's try variations)
    keys2 = np.random.rand(n)
    order_deg_high = np.lexsort((keys2, -deg))
    population.append(order_deg_high)
    
    # Strategy 3: Random
    order_rand = np.random.permutation(n)
    population.append(order_rand)
    
    # Strategy 4: Spectral-like (using eigenvector of Laplacian if fast enough, else random)
    # For n=200, scipy.linalg.eigh is fast.
    try:
        # Laplacian L = D - A
        L = np.diag(deg) - adj.astype(float)
        # Get smallest eigenvector (Fiedler vector)
        # Actually, we want an ordering. Let's use the eigenvector components as keys.
        # Smallest non-zero eigenvalue eigenvector gives a smooth ordering.
        # But for MIS, maybe we want to separate connected components?
        # Let's just use the eigenvector of the adjacency matrix (principal eigenvector) 
        # to prioritize high-centrality nodes? Or low?
        # Let's try: Sort by eigenvector component ascending.
        vals, vecs = np.linalg.eigh(adj.astype(float))
        # Principal eigenvector is the last column for eigh (ascending eigenvalues)
        fiedler_vec = vecs[:, -1]
        # Use this as a key
        order_spec = np.argsort(fiedler_vec)
        population.append(order_spec)
    except:
        population.append(np.random.permutation(n))
        
    # Fill rest of population with random permutations
    while len(population) < pop_size:
        population.append(np.random.permutation(n))
        
    # Evaluate initial population
    for ind in population:
        fitnesses.append(_greedy_set_size_from_order(adj, ind))
        
    best_fitness = max(fitnesses)
    best_order = population[fitnesses.index(best_fitness)].copy()
    
    generation = 0
    no_improve_gens = 0
    
    while time.time() < deadline:
        # Select parents
        # Tournament selection
        def tournament_select():
            idxs = random.sample(range(pop_size), 3)
            best_idx = max(idxs, key=lambda i: fitnesses[i])
            return population[best_idx].copy()
            
        # Create new population
        new_population = []
        new_fitnesses = []
        
        # Elitism: keep best individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(elite_size):
            idx = sorted_indices[i]
            new_population.append(population[idx].copy())
            new_fitnesses.append(fitnesses[idx])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if random.random() < 0.5:
                child = _order_crossover(parent1, parent2, n)
            else:
                child = _uniform_crossover(parent1, parent2, n)
                
            # Mutation: Random Swap
            if random.random() < 0.8:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
                
            # Mutation: Inversion
            if random.random() < 0.2:
                i, j = random.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                child[i:j+1] = child[i:j+1][::-1]
                
            new_population.append(child)
            new_fitnesses.append(0) # Placeholder
            
        # Evaluate new offspring
        for i in range(elite_size, pop_size):
            new_fitnesses[i] = _greedy_set_size_from_order(adj, new_population[i])
            
        # Check for improvement
        gen_best_fitness = max(new_fitnesses)
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_order = new_population[new_fitnesses.index(gen_best_fitness)].copy()
            no_improve_gens = 0
        else:
            no_improve_gens += 1
            
        # Replace population
        population = new_population
        fitnesses = new_fitnesses
        
        generation += 1
        
        # Early stopping if stuck too long
        if no_improve_gens > 50:
            break
            
    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
