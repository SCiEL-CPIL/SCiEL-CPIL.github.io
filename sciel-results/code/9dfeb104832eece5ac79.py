import time
import sys
from itertools import combinations, permutations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    num_edges_total = n * (n - 1) // 2  # 120
    
    # 1. Edge Indexing
    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)
            idx += 1
            
    # 2. Precompute Q3 masks using bitmask representation
    # Q3 has 8 vertices and 12 edges
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Generate all distinct Q3 masks on subsets of 8 vertices from 0..15
    # Each mask is a 120-bit integer representing which edges of K16 form a Q3
    all_q3_masks = []
    seen_masks = set()
    
    # For efficiency, we'll generate Q3s by choosing 8 vertices and all 8! labelings
    # But many will be duplicates, so we use a set
    
    # Precompute: for each subset of 8 vertices, and each permutation of labels 0..7,
    # compute the global edge mask
    
    # To speed up, precompute local edge index mapping
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        count = 0
        for i in range(u):
            count += (7 - i)
        count += (v - u - 1)
        return count
    
    # Precompute local Q3 edge index sets (unique patterns)
    local_q3_patterns = set()
    for p in permutations(range(8)):
        pattern = frozenset()
        for u, v in q3_abstract_edges:
            lu, lv = p[u], p[v]
            if lu > lv:
                lu, lv = lv, lu
            le = local_edge_index(lu, lv)
            pattern.add(le)
        if len(pattern) == 12:
            local_q3_patterns.add(pattern)
    
    local_q3_list = list(local_q3_patterns)
    
    # For each local pattern, precompute the edge indices
    local_q3_edge_indices = []
    for pattern in local_q3_list:
        local_q3_edge_indices.append(list(pattern))
    
    # Now generate global Q3 masks
    # For each subset of 8 vertices, and each local pattern, compute global mask
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute for each subset the mapping from local edge index to global edge index
    subset_l2g_maps = {}
    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                u_min, v_min = min(u_glob, v_glob), max(u_glob, v_glob)
                l2g[le] = edge_idx_map[(u_min, v_min)]
        subset_l2g_maps[subset] = l2g
    
    # Generate all global Q3 masks
    for subset in all_subsets:
        l2g = subset_l2g_maps[subset]
        for edge_indices in local_q3_edge_indices:
            mask = 0
            for le in edge_indices:
                ge = l2g[le]
                mask |= (1 << ge)
            if mask not in seen_masks:
                seen_masks.add(mask)
                all_q3_masks.append(mask)
    
    num_q3s = len(all_q3_masks)
    
    if time.time() > deadline:
        # Fallback: return a sparse valid graph
        return [[i, i+1] for i in range(0, 16, 2)]
    
    # 3. Precompute for each edge, which Q3s contain it
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for i, mask in enumerate(all_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
    
    # Precompute edge lists for each Q3
    q3_edge_lists = []
    for mask in all_q3_masks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edges.append(e_idx)
            m ^= lsb
        q3_edge_lists.append(edges)
    
    # 4. Genetic Algorithm on bitmasks
    # State: 120-bit integer where bit e is 1 if edge e is present
    # Fitness: count of present edges - penalty * (number of Q3s fully present)
    
    PENALTY = 1000  # Large penalty for each embedded Q3
    
    def count_q3s_in_mask(graph_mask):
        """Count how many Q3s are fully present in the graph represented by graph_mask."""
        count = 0
        for i in range(num_q3s):
            if (graph_mask & all_q3_masks[i]) == all_q3_masks[i]:
                count += 1
        return count
    
    def fitness(graph_mask):
        """Fitness = number of edges - penalty * number of Q3s."""
        edge_count = graph_mask.bit_count()
        q3_count = count_q3s_in_mask(graph_mask)
        return edge_count - PENALTY * q3_count
    
    # Initialize population
    POP_SIZE = 50
    population = []
    
    # Start with some diverse initial solutions
    # 1. Random dense graphs
    for i in range(POP_SIZE // 2):
        mask = 0
        for e in range(num_edges_total):
            if random.random() < 0.7:  # 70% edge probability
                mask |= (1 << e)
        population.append(mask)
    
    # 2. Greedy hitting set result (from seed)
    # First, run a quick greedy to get a good starting point
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    removed_edges = [False] * num_edges_total
    
    for i, mask in enumerate(all_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            counts[e_idx] += 1
            m ^= lsb
    
    # Greedy: remove edges with highest count
    for _ in range(num_q3s):
        if time.time() > deadline - 10:
            break
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
        if best_edge == -1 or max_count == 0:
            break
        removed_edges[best_edge] = True
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = all_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb
    
    # Build greedy solution mask
    greedy_mask = 0
    for e in range(num_edges_total):
        if not removed_edges[e]:
            greedy_mask |= (1 << e)
    population.append(greedy_mask)
    
    # 3. Local search improvement on greedy
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for e in range(num_edges_total):
            if removed_edges[e]:
                # Check if we can add back edge e
                safe = True
                for q3_idx in q3s_containing_edge[e]:
                    edges_in_q = q3_edge_lists[q3_idx]
                    all_others_kept = True
                    for other_e in edges_in_q:
                        if other_e == e:
                            continue
                        if removed_edges[other_e]:
                            all_others_kept = False
                            break
                    if all_others_kept:
                        safe = False
                        break
                if safe:
                    removed_edges[e] = False
                    changed = True
    
    # Rebuild mask after local search
    improved_mask = 0
    for e in range(num_edges_total):
        if not removed_edges[e]:
            improved_mask |= (1 << e)
    population.append(improved_mask)
    
    # Fill rest with variations
    while len(population) < POP_SIZE:
        # Mutate the best so far
        best_mask = max(population, key=fitness)
        new_mask = best_mask
        for _ in range(random.randint(1, 5)):
            e = random.randint(0, num_edges_total - 1)
            if random.random() < 0.5:
                new_mask |= (1 << e)
            else:
                new_mask &= ~(1 << e)
        population.append(new_mask)
    
    # Track best solution found
    best_mask = max(population, key=fitness)
    best_fitness = fitness(best_mask)
    
    # Genetic algorithm loop
    GENERATIONS = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    
    for gen in range(GENERATIONS):
        if time.time() > deadline - 2:
            break
        
        # Evaluate fitness for all individuals
        fitnesses = [fitness(ind) for ind in population]
        
        # Update best
        best_idx = max(range(POP_SIZE), key=lambda i: fitnesses[i])
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_mask = population[best_idx]
        
        # Create new generation
        new_population = []
        
        # Elitism: keep top 10%
        elite_size = max(1, POP_SIZE // 10)
        sorted_indices = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
        
        # Fill rest with crossover and mutation
        while len(new_population) < POP_SIZE:
            # Select parents via tournament selection
            def tournament_select(k=3):
                candidates = random.sample(range(POP_SIZE), k)
                winner = max(candidates, key=lambda i: fitnesses[i])
                return population[winner]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if random.random() < CROSSOVER_RATE:
                # Uniform crossover
                child = 0
                for e in range(num_edges_total):
                    if random.random() < 0.5:
                        if parent1 & (1 << e):
                            child |= (1 << e)
                    else:
                        if parent2 & (1 << e):
                            child |= (1 << e)
            else:
                child = parent1
            
            # Mutation
            for e in range(num_edges_total):
                if random.random() < MUTATION_RATE:
                    if random.random() < 0.5:
                        child |= (1 << e)
                    else:
                        child &= ~(1 << e)
            
            new_population.append(child)
        
        population = new_population
    
    # 5. Final local search on the best solution
    # Convert best_mask to removed_edges representation
    removed_edges = [False] * num_edges_total
    for e in range(num_edges_total):
        if not (best_mask & (1 << e)):
            removed_edges[e] = True
    
    # Try to add back edges
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for e in range(num_edges_total):
            if not removed_edges[e]:
                continue
            # Check if e can be added back
            safe = True
            for q3_idx in q3s_containing_edge[e]:
                edges_in_q = q3_edge_lists[q3_idx]
                all_others_kept = True
                for other_e in edges_in_q:
                    if other_e == e:
                        continue
                    if removed_edges[other_e]:
                        all_others_kept = False
                        break
                if all_others_kept:
                    safe = False
                    break
            if safe:
                removed_edges[e] = False
                changed = True
    
    # Try to remove edges that are not needed (i.e., their removal doesn't affect Q3-freeness)
    # Actually, we want to maximize edges, so we only add back edges, not remove
    
    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges
