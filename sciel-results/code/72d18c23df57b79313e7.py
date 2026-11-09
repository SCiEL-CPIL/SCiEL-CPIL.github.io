"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Precompute Q3 structure
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    # Edge index mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    
    # Precompute: for each pair of edges (e1, e2), do they share a vertex?
    # And for each Q3 edge pattern, which graph edge pairs could form it
    
    # Fast Q3 check: given a bitmask of edges, check if it contains Q3
    # We'll use a different approach: maintain adjacency and check incrementally
    
    def build_adj_from_mask(mask):
        adj = [set() for _ in range(n)]
        for e_idx in range(total_edges):
            if mask & (1 << e_idx):
                u, v = idx_to_edge[e_idx]
                adj[u].add(v)
                adj[v].add(u)
        return adj
    
    def has_q3_fast(adj):
        """Check if graph with adjacency sets contains Q3 as subgraph.
        Uses the fact that Q3 has 8 vertices, each of degree 3.
        Strategy: try all 8-vertex subsets and check if they can host Q3."""
        # For efficiency, first check if there are enough vertices with degree >= 3
        deg3_plus = [i for i in range(n) if len(adj[i]) >= 3]
        if len(deg3_plus) < 8:
            return False
        
        # Try embeddings: pick a vertex as Q3 vertex 0
        # Q3 vertex 0 connects to 1, 2, 4
        # Use backtracking with constraint propagation
        
        def backtrack(assignment, q3_idx):
            if q3_idx == 8:
                return True
            
            q = q3_idx
            # Get candidates: vertices not yet used
            used = set(assignment.values())
            
            # For each candidate, check compatibility with assigned neighbors
            for g in deg3_plus:
                if g in used:
                    continue
                
                compatible = True
                for p in range(q3_idx):
                    if p in q3_adj[q]:
                        if assignment[p] not in adj[g]:
                            compatible = False
                            break
                if compatible:
                    assignment[q] = g
                    if backtrack(assignment, q3_idx + 1):
                        return True
                    assignment[q] = None
            
            return False
        
        # Try starting with each vertex as Q3 vertex 0
        for start_v in deg3_plus:
            assignment = {0: start_v}
            if backtrack(assignment, 1):
                return True
        
        return False
    
    def count_q3_embeddings(adj):
        """Count number of Q3 subgraphs (not just check existence)."""
        count = 0
        deg3_plus = [i for i in range(n) if len(adj[i]) >= 3]
        if len(deg3_plus) < 8:
            return 0
        
        def backtrack(assignment, q3_idx):
            nonlocal count
            if q3_idx == 8:
                count += 1
                return
            
            q = q3_idx
            used = set(assignment.values())
            
            for g in deg3_plus:
                if g in used:
                    continue
                
                compatible = True
                for p in range(q3_idx):
                    if p in q3_adj[q]:
                        if assignment[p] not in adj[g]:
                            compatible = False
                            break
                if compatible:
                    assignment[q] = g
                    backtrack(assignment, q3_idx + 1)
                    assignment[q] = None
            
            return
        
        for start_v in deg3_plus:
            assignment = {0: start_v}
            backtrack(assignment, 1)
        
        return count
    
    # Genetic algorithm
    population_size = 40
    mutation_rate = 0.05
    crossover_rate = 0.8
    elite_size = 5
    penalty_weight = 100  # Large penalty per Q3 embedding
    
    def fitness(mask):
        """Fitness = number of edges - penalty * number of Q3 embeddings"""
        edge_count = bin(mask).count('1')
        adj = build_adj_from_mask(mask)
        q3_count = count_q3_embeddings(adj)
        return edge_count - penalty_weight * q3_count
    
    def get_q3_free_count(mask):
        """Return (edge_count, q3_count) for a mask"""
        edge_count = bin(mask).count('1')
        adj = build_adj_from_mask(mask)
        q3_count = count_q3_embeddings(adj)
        return edge_count, q3_count
    
    # Initialize population
    # Start with diverse initial solutions
    population = []
    
    # Solution 1: K_{3,13}
    mask1 = 0
    part_a = [0, 1, 2]
    part_b = list(range(3, n))
    for a in part_a:
        for b in part_b:
            mask1 |= (1 << edge_to_idx[(a, b)])
    population.append(mask1)
    
    # Solution 2: Random dense graph
    for _ in range(population_size - 1):
        mask = 0
        for e_idx in range(total_edges):
            if random.random() < 0.4:
                mask |= (1 << e_idx)
        population.append(mask)
    
    # Evaluate initial population
    best_mask = max(population, key=fitness)
    best_fitness = fitness(best_mask)
    best_q3_free = None
    best_edge_count = 0
    
    # Track best Q3-free solution
    for mask in population:
        ec, qc = get_q3_free_count(mask)
        if qc == 0 and ec > best_edge_count:
            best_edge_count = ec
            best_q3_free = mask
    
    generations = 0
    max_generations = 200
    
    while time.time() < deadline - 20 and generations < max_generations:
        generations += 1
        
        # Evaluate fitness
        fitnesses = [fitness(mask) for mask in population]
        
        # Find best Q3-free in current population
        for i, mask in enumerate(population):
            ec, qc = get_q3_free_count(mask)
            if qc == 0 and ec > best_edge_count:
                best_edge_count = ec
                best_q3_free = mask
        
        # Selection: tournament selection
        def tournament_select(pop, fits, k=3):
            candidates = random.sample(range(len(pop)), k)
            best_idx = max(candidates, key=lambda i: fits[i])
            return pop[best_idx]
        
        # Create next generation
        new_population = []
        
        # Keep elites
        elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        for i in elite_indices[:elite_size]:
            new_population.append(population[i])
        
        # Generate rest
        while len(new_population) < population_size:
            if random.random() < crossover_rate and len(new_population) >= 2:
                # Crossover
                parent1 = tournament_select(population, fitnesses)
                parent2 = tournament_select(population, fitnesses)
                # Uniform crossover
                child = 0
                for e_idx in range(total_edges):
                    if random.random() < 0.5:
                        if parent1 & (1 << e_idx):
                            child |= (1 << e_idx)
                    else:
                        if parent2 & (1 << e_idx):
                            child |= (1 << e_idx)
            else:
                # Clone
                parent = tournament_select(population, fitnesses)
                child = parent
            
            # Mutation: flip random bits
            for e_idx in range(total_edges):
                if random.random() < mutation_rate:
                    child ^= (1 << e_idx)
            
            new_population.append(child)
        
        population = new_population
    
    # Final polish: local search on best Q3-free solution
    if best_q3_free is not None:
        adj = build_adj_from_mask(best_q3_free)
        edge_count = bin(best_q3_free).count('1')
        
        # Try to add edges greedily
        for e_idx in range(total_edges):
            if time.time() > deadline - 5:
                break
            if best_q3_free & (1 << e_idx):
                continue
            
            u, v = idx_to_edge[e_idx]
            # Try adding this edge
            new_mask = best_q3_free | (1 << e_idx)
            new_adj = build_adj_from_mask(new_mask)
            
            # Check if adding this edge creates Q3
            # Fast check: only need to check Q3 embeddings that include this edge
            creates_q3 = False
            for (a, b) in q3_edges:
                # Case 1: a->u, b->v
                if _check_edge_embedding(new_adj, u, v, a, b):
                    creates_q3 = True
                    break
                # Case 2: a->v, b->u
                if _check_edge_embedding(new_adj, v, u, a, b):
                    creates_q3 = True
                    break
            
            if not creates_q3:
                best_q3_free = new_mask
                edge_count += 1
                adj = new_adj
    
    # Convert best mask to edge list
    result_edges = []
    for e_idx in range(total_edges):
        if best_q3_free & (1 << e_idx):
            u, v = idx_to_edge[e_idx]
            result_edges.append([u, v])
    
    return result_edges

def _check_edge_embedding(adj, u, v, a, b):
    """Check if Q3 can be embedded with vertex a mapped to u and vertex b mapped to v.
    a and b are adjacent in Q3."""
    n = 16
    q3_adj = [set() for _ in range(8)]
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
                q3_adj[i].add(j)
                q3_adj[j].add(i)
    
    assignment = [None] * 8
    assignment[a] = u
    assignment[b] = v
    
    remaining = [i for i in range(8) if i not in (a, b)]
    
    def backtrack(rem_idx):
        if rem_idx == len(remaining):
            return True
        q = remaining[rem_idx]
        
        for g in range(n):
            # Check if g is already used
            used = False
            for p in range(8):
                if assignment[p] is not None and assignment[p] == g:
                    used = True
                    break
            if used:
                continue
            
            # Check compatibility
            compatible = True
            for p in range(8):
                if assignment[p] is not None:
                    if p in q3_adj[q]:
                        if assignment[p] not in adj[g]:
                            compatible = False
                            break
            if compatible:
                assignment[q] = g
                if backtrack(rem_idx + 1):
                    return True
                assignment[q] = None
        
        return False
    
    return backtrack(0)
# EVOLVE-BLOCK-END
