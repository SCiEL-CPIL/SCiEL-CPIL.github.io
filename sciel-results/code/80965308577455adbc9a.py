import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def count_triangles(mask, adj):
        """Count triangles in the induced subgraph defined by mask."""
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            neigh = adj[u] & mask & ~(1 << u)
            n_neigh = neigh
            while n_neigh:
                lsb2 = n_neigh & -n_neigh
                v = lsb2.bit_length() - 1
                n_neigh ^= lsb2
                count += (adj[v] & neigh & ~((1 << v))).bit_count()
        return count // 3

    def count_indep_5(mask, adj):
        """Count independent sets of size 5 in the induced subgraph defined by mask."""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        count = 0
        n_v = len(vertices)
        if n_v < 5:
            return 0
        
        for i in range(n_v):
            for j in range(i+1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j+1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k+1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        for m_idx in range(l+1, n_v):
                            if (adj[vertices[i]] >> vertices[m_idx]) & 1 or (adj[vertices[j]] >> vertices[m_idx]) & 1 or (adj[vertices[k]] >> vertices[m_idx]) & 1 or (adj[vertices[l]] >> vertices[m_idx]) & 1:
                                continue
                            count += 1
        return count

    def calculate_cost(n, adj):
        k4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    k4_count += count_triangles(common, adj)
        
        i6_count = 0
        total_i5_in_nonn = 0
        all_mask = (1 << n) - 1
        for u in range(n):
            non_n = all_mask & ~(adj[u] | (1 << u))
            total_i5_in_nonn += count_indep_5(non_n, adj)
            
        i6_count = total_i5_in_nonn // 6
        
        return k4_count + i6_count

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def is_valid(n, adj):
        return calculate_cost(n, adj) == 0

    def known_35_graph():
        """Return the known Exoo 35-vertex K4-free graph with alpha < 6."""
        # This is a well-known extremal graph for R(4,6)
        # Using a construction based on Paley-like structure
        n = 35
        adj = [0] * n
        # Use a specific construction: take 5 groups of 7, with specific inter-group edges
        # For now, use a random valid graph found through local search
        # We'll generate it via local search
        return None

    def local_search_extend(base_n, base_adj, target_n, max_steps=20000):
        """Try to extend a valid graph to target_n vertices."""
        if target_n <= base_n:
            return base_adj, base_n
        
        # Start with base graph + random new vertices
        adj = [0] * target_n
        for u in range(base_n):
            adj[u] = base_adj[u]
        
        # Add new vertices with random connections
        for u in range(base_n, target_n):
            for v in range(u):
                if random.random() < 0.4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        current_cost = calculate_cost(target_n, adj)
        if current_cost == 0:
            return adj, target_n
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        for s in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Pick random edge
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_edge = bool((adj[u] >> v) & 1)
            
            # Calculate delta
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            all_mask = (1 << target_n) - 1
            non_common = all_mask & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
            
            tri_count = count_triangles(common, adj)
            i5_count = count_indep_5(non_common, adj)
            
            if is_edge:
                delta = -tri_count + i5_count
            else:
                delta = tri_count - i5_count
            
            if delta < 0:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_cost += delta
                if current_cost == 0:
                    return adj, target_n
            else:
                p = math.exp(-delta / T)
                if random.random() < p:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_cost += delta
                    if current_cost == 0:
                        return adj, target_n
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return adj, target_n

    def genetic_crossover(adj1, adj2, n1, n2, target_n):
        """Perform vertex-union crossover of two graphs."""
        if target_n <= max(n1, n2):
            # Just return the larger one with random modifications
            if n1 >= n2:
                adj = adj1[:]
                n = n1
            else:
                adj = adj2[:]
                n = n2
            
            # Extend to target_n if needed
            if n < target_n:
                adj = adj + [0] * (target_n - n)
                for u in range(n, target_n):
                    for v in range(u):
                        if random.random() < 0.4:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                n = target_n
            return adj, n
        
        # Combine: take first n1 from adj1, then add new vertices
        adj = [0] * target_n
        for u in range(n1):
            adj[u] = adj1[u]
        
        # Add vertices from adj2 that are "new"
        for u in range(n1, target_n):
            # Mix with adj2 if available
            if u - n1 < n2:
                src = u - n1
                for v in range(u):
                    # Combine edges from both parents
                    edge1 = (adj1[u] >> v) & 1 if u < n1 else 0
                    edge2 = (adj2[src] >> v) & 1 if v < n2 else 0
                    # Use OR or AND or random
                    if random.random() < 0.5:
                        if edge1 or edge2:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    else:
                        if edge1 and edge2:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            else:
                for v in range(u):
                    if random.random() < 0.4:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        return adj, target_n

    def blow_up_construction(base_adj, base_n, expansion_factor, target_n):
        """Perform blow-up construction: substitute base graph into each vertex."""
        if expansion_factor * base_n != target_n:
            # Adjust expansion factor
            expansion_factor = max(1, target_n // base_n)
        
        new_n = expansion_factor * base_n
        if new_n > target_n:
            new_n = target_n
        
        new_adj = [0] * new_n
        
        for i in range(base_n):
            for j in range(base_n):
                if i == j:
                    # Within each group: no edges (independent set)
                    continue
                if (base_adj[i] >> j) & 1:
                    # Between groups i and j: complete bipartite
                    for a in range(expansion_factor * i, min(expansion_factor * (i+1), new_n)):
                        for b in range(expansion_factor * j, min(expansion_factor * (j+1), new_n)):
                            if a < new_n and b < new_n:
                                new_adj[a] |= (1 << b)
                                new_adj[b] |= (1 << a)
        
        return new_adj, new_n

    best_graph = None
    best_n = 0

    # Strategy 1: Start from random 35-vertex graph and try to extend to 36
    for target_n in [36, 35, 34]:
        if time.time() > deadline - 15:
            break
        
        # Multiple attempts with different initializations
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            
            # Initialize random graph
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if random.random() < 0.45:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_cost = calculate_cost(target_n, adj)
            if current_cost == 0:
                best_graph = adj
                best_n = target_n
                break
            
            # Local search
            T = 5.0
            T_min = 0.05
            cooling = 0.995
            max_steps = 30000 if target_n >= 35 else 50000
            
            for s in range(max_steps):
                if time.time() > deadline - 15:
                    break
                
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                is_edge = bool((adj[u] >> v) & 1)
                
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                all_mask = (1 << target_n) - 1
                non_common = all_mask & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
                
                tri_count = count_triangles(common, adj)
                i5_count = count_indep_5(non_common, adj)
                
                if is_edge:
                    delta = -tri_count + i5_count
                else:
                    delta = tri_count - i5_count
                
                if delta < 0:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_cost += delta
                    if current_cost == 0:
                        best_graph = adj
                        best_n = target_n
                        break
                else:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        current_cost += delta
                        if current_cost == 0:
                            best_graph = adj
                            best_n = target_n
                            break
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if best_n == target_n:
                break
        
        if best_n == target_n:
            break

    # Strategy 2: Genetic algorithm if not found yet
    if best_n < 36 and time.time() < deadline - 20:
        population = []
        pop_size = 10
        
        # Initialize population
        for i in range(pop_size):
            n_init = random.choice([34, 35, 36])
            adj = [0] * n_init
            for u in range(n_init):
                for v in range(u+1, n_init):
                    if random.random() < 0.45:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            population.append((adj, n_init))
        
        # Evolve
        generations = 5
        for gen in range(generations):
            if time.time() > deadline - 15:
                break
            
            new_population = []
            
            for i in range(pop_size):
                if time.time() > deadline - 15:
                    break
                
                # Crossover
                if random.random() < 0.7 and i > 0:
                    parent1 = population[i-1]
                    parent2 = population[random.randint(0, pop_size-1)]
                    child_adj, child_n = genetic_crossover(
                        parent1[0], parent1[1], parent2[0], parent2[1], 36
                    )
                else:
                    child_adj, child_n = population[i][0][:], population[i][1]
                
                # Mutation: random edge flips
                for _ in range(20):
                    u = random.randint(0, child_n - 1)
                    v = random.randint(0, child_n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    child_adj[u] ^= (1 << v)
                    child_adj[v] ^= (1 << u)
                
                # Local search on child
                current_cost = calculate_cost(child_n, child_adj)
                if current_cost == 0:
                    if child_n > best_n:
                        best_graph = child_adj
                        best_n = child_n
                    new_population.append((child_adj, child_n))
                    continue
                
                T = 3.0
                T_min = 0.1
                cooling = 0.995
                max_steps = 5000
                
                for s in range(max_steps):
                    if time.time() > deadline - 15:
                        break
                    
                    u = random.randint(0, child_n - 1)
                    v = random.randint(0, child_n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    is_edge = bool((child_adj[u] >> v) & 1)
                    
                    common = child_adj[u] & child_adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    all_mask = (1 << child_n) - 1
                    non_common = all_mask & ~(child_adj[u] | child_adj[v]) & ~(1 << u) & ~(1 << v)
                    
                    tri_count = count_triangles(common, child_adj)
                    i5_count = count_indep_5(non_common, child_adj)
                    
                    if is_edge:
                        delta = -tri_count + i5_count
                    else:
                        delta = tri_count - i5_count
                    
                    if delta < 0:
                        child_adj[u] ^= (1 << v)
                        child_adj[v] ^= (1 << u)
                        current_cost += delta
                        if current_cost == 0:
                            if child_n > best_n:
                                best_graph = child_adj
                                best_n = child_n
                            break
                    else:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            child_adj[u] ^= (1 << v)
                            child_adj[v] ^= (1 << u)
                            current_cost += delta
                            if current_cost == 0:
                                if child_n > best_n:
                                    best_graph = child_adj
                                    best_n = child_n
                                break
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                
                new_population.append((child_adj, child_n))
            
            population = new_population[:pop_size]

    # Fallback: known valid 35-vertex graph via local search
    if best_graph is None:
        # Try to find any valid graph
        for target_n in [35, 34, 33, 32, 31, 30, 25, 20, 17]:
            if time.time() > deadline - 5:
                break
            
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u+1, target_n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                current_cost = calculate_cost(target_n, adj)
                if current_cost == 0:
                    best_graph = adj
                    best_n = target_n
                    break
                
                T = 5.0
                T_min = 0.1
                cooling = 0.99
                max_steps = 10000
                
                for s in range(max_steps):
                    if time.time() > deadline - 5:
                        break
                    
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    is_edge = bool((adj[u] >> v) & 1)
                    
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    all_mask = (1 << target_n) - 1
                    non_common = all_mask & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
                    
                    tri_count = count_triangles(common, adj)
                    i5_count = count_indep_5(non_common, adj)
                    
                    if is_edge:
                        delta = -tri_count + i5_count
                    else:
                        delta = tri_count - i5_count
                    
                    if delta < 0:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        current_cost += delta
                        if current_cost == 0:
                            best_graph = adj
                            best_n = target_n
                            break
                    else:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                            current_cost += delta
                            if current_cost == 0:
                                best_graph = adj
                                best_n = target_n
                                break
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                
                if best_n == target_n:
                    break
            
            if best_graph is not None:
                break
    
    # Final fallback: Paley 17
    if best_graph is None:
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
