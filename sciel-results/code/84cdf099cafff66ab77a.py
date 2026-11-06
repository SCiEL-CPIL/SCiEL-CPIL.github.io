"""Construct a K4-free graph with no independent 6-set, maximizing n."""
import time
import random
import itertools
import sys

DEADLINE = 145.0  # safety margin before 150s kill

def _has_k4(adj, n):
    """Check if graph has a K4. adj is a set of frozensets or a 2D bool array."""
    # Use adjacency matrix for speed
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            for k in range(j+1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k+1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return True
    return False

def _has_indep_6(adj, n):
    """Check if graph has an independent set of size 6."""
    # Independent set of 6 means 6 vertices with no edges between them
    # Equivalently, in the complement graph, a K6
    # But let's just check directly
    for combo in itertools.combinations(range(n), 6):
        is_indep = True
        for a in range(6):
            for b in range(a+1, 6):
                if adj[combo[a]][combo[b]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            return True
    return False

def _is_valid(adj, n):
    """Check if the graph is K4-free and has no independent 6-set."""
    if _has_k4(adj, n):
        return False
    if _has_indep_6(adj, n):
        return False
    return True

def _count_k4s(adj, n):
    """Count number of K4s in the graph."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            for k in range(j+1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k+1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        count += 1
    return count

def _count_indep_6s(adj, n):
    """Count number of independent 6-sets."""
    count = 0
    for combo in itertools.combinations(range(n), 6):
        is_indep = True
        for a in range(6):
            for b in range(a+1, 6):
                if adj[combo[a]][combo[b]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            count += 1
    return count

def _paley_graph(n):
    """Create a Paley graph on n vertices (n should be prime ≡ 1 mod 4)."""
    # Find quadratic residues mod n
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % n
            if diff in squares:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _random_graph(n, p=0.5):
    """Create a random graph with edge probability p."""
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _fitness(adj, n):
    """
    Fitness function: maximize n, penalize K4s and independent 6-sets.
    Higher is better.
    """
    k4_count = _count_k4s(adj, n)
    indep6_count = _count_indep_6s(adj, n)
    
    # Penalty: large for each violation
    penalty = 1000 * (k4_count + indep6_count)
    return n - penalty

def _crossover(adj1, adj2, n):
    """Uniform crossover: for each edge, pick from either parent."""
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = adj1[i][j]
                adj[j][i] = adj1[i][j]
            else:
                adj[i][j] = adj2[i][j]
                adj[j][i] = adj2[i][j]
    return adj

def _mutate(adj, n, mutation_rate=0.05):
    """Random mutation: flip edges with probability mutation_rate."""
    adj = [row[:] for row in adj]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < mutation_rate:
                adj[i][j] = not adj[i][j]
                adj[j][i] = adj[i][j]
    return adj

def _walksat_step(adj, n, max_steps=1000):
    """
    WalkSAT-style local search to eliminate K4s and independent 6-sets.
    Returns improved adjacency matrix.
    """
    adj = [row[:] for row in adj]
    
    for step in range(max_steps):
        # Check if valid
        if not _has_k4(adj, n) and not _has_indep_6(adj, n):
            break
        
        # Pick a random violating structure
        # Try to find a K4 or independent 6-set
        found_violation = False
        
        # Try to fix a K4
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            # Found a K4: {i, j, k, l}
                            # Remove a random edge from this K4
                            edges_to_remove = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                            e = random.choice(edges_to_remove)
                            adj[e[0]][e[1]] = False
                            adj[e[1]][e[0]] = False
                            found_violation = True
                            break
                    if found_violation:
                        break
                if found_violation:
                    break
            if found_violation:
                break
        
        if found_violation:
            continue
        
        # Try to fix an independent 6-set
        for combo in itertools.combinations(range(n), 6):
            is_indep = True
            for a in range(6):
                for b in range(a+1, 6):
                    if adj[combo[a]][combo[b]]:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                # Add a random edge within this independent set
                a, b = random.sample(range(6), 2)
                adj[combo[a]][combo[b]] = True
                adj[combo[b]][combo[a]] = True
                found_violation = True
                break
        
        if found_violation:
            continue
        
        # No specific violation found (shouldn't happen if has_k4/has_indep_6 said yes)
        # Random flip
        i, j = random.sample(range(n), 2)
        adj[i][j] = not adj[i][j]
        adj[j][i] = adj[i][j]
    
    return adj

def construct():
    """
    Construct a K4-free graph with no independent 6-set, maximizing n.
    Uses a genetic algorithm with WalkSAT local search.
    """
    start_time = time.time()
    
    # Target: try to reach n=35 (Exoo's record)
    # Start with smaller n and grow
    
    best_n = 0
    best_adj = None
    
    # Strategy: 
    # 1. Try to build up from a valid small graph
    # 2. Use genetic algorithm on adjacency matrices
    # 3. Use WalkSAT for local repair
    
    # Start with known good seeds
    seeds = []
    
    # Paley graph on 5 vertices (trivially valid, small)
    seeds.append(_paley_graph(5))
    
    # Random graphs of various sizes
    for n in [10, 15, 20, 25, 30, 35]:
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            seeds.append(_random_graph(n, p))
    
    # Initialize population
    pop_size = 50
    population = []
    
    for seed in seeds:
        n = len(seed)
        adj = _walksat_step(seed, n, max_steps=200)
        if _is_valid(adj, n):
            population.append((adj, n, _fitness(adj, n)))
    
    # Add random individuals
    while len(population) < pop_size:
        n = random.randint(10, 35)
        p = random.uniform(0.3, 0.7)
        adj = _random_graph(n, p)
        adj = _walksat_step(adj, n, max_steps=200)
        population.append((adj, n, _fitness(adj, n)))
    
    # Sort by fitness
    population.sort(key=lambda x: x[2], reverse=True)
    
    # Update best
    if population:
        best_adj, best_n, _ = population[0]
    
    # Genetic algorithm loop
    gen = 0
    max_gens = 200
    
    while gen < max_gens:
        if time.time() - start_time > DEADLINE:
            break
        
        gen += 1
        new_population = []
        
        # Elitism: keep top 20%
        elite_size = max(1, pop_size // 5)
        for i in range(elite_size):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > DEADLINE:
                break
            
            # Selection: tournament
            candidates = random.sample(population, min(3, len(population)))
            parent1 = max(candidates, key=lambda x: x[2])
            candidates = random.sample(population, min(3, len(population)))
            parent2 = max(candidates, key=lambda x: x[2])
            
            # Crossover (only if same size)
            if parent1[1] == parent2[1]:
                n = parent1[1]
                adj = _crossover(parent1[0], parent2[0], n)
            else:
                # Take the larger parent and mutate
                if parent1[1] >= parent2[1]:
                    adj = parent1[0]
                    n = parent1[1]
                else:
                    adj = parent2[0]
                    n = parent2[1]
            
            # Mutation
            adj = _mutate(adj, n, mutation_rate=0.05)
            
            # WalkSAT repair
            adj = _walksat_step(adj, n, max_steps=100)
            
            fitness = _fitness(adj, n)
            new_population.append((adj, n, fitness))
        
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        # Update best
        if population and population[0][1] > best_n:
            best_adj = population[0][0]
            best_n = population[0][1]
        
        # Try to grow: take best valid graph and add a vertex
        if best_adj is not None and best_n < 35:
            if time.time() - start_time > DEADLINE - 5:
                break
            new_n = best_n + 1
            # Create new graph with one more vertex
            new_adj = [[False] * new_n for _ in range(new_n)]
            for i in range(best_n):
                for j in range(best_n):
                    new_adj[i][j] = best_adj[i][j]
            
            # Try different connection patterns for the new vertex
            for trial in range(100):
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Randomly connect new vertex to existing vertices
                for i in range(best_n):
                    if random.random() < 0.5:
                        new_adj[i][new_n-1] = True
                        new_adj[new_n-1][i] = True
                
                # Repair with WalkSAT
                repaired = _walksat_step(new_adj, new_n, max_steps=200)
                if _is_valid(repaired, new_n):
                    best_adj = repaired
                    best_n = new_n
                    break
    
    # Final check and return
    if best_adj is None or not _is_valid(best_adj, best_n):
        # Fallback: return a small valid graph
        # Empty graph on 5 vertices: no K4, no independent 6-set
        return {'n': 5, 'edges': []}
    
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
