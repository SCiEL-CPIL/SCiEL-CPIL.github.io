    import time
    import random
    import sys

    def construct():
        start_time = time.time()
        deadline = start_time + 110
        
        n = 20
        best_edges = []
        best_count = 0
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges_possible = n * (n - 1) // 2
        
        def is_valid_add(adj, u, v):
            common_uv = adj[u] & adj[v]
            if common_uv.bit_count() > 2:
                return False
            
            w_mask = adj[u]
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                common_vw = adj[v] & adj[w]
                if common_vw.bit_count() > 1:
                    return False
                w_mask ^= lsb
                
            w_mask = adj[v]
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                common_uw = adj[u] & adj[w]
                if common_uw.bit_count() > 1:
                    return False
                w_mask ^= lsb
                
            return True

        def add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        def remove_edge(adj, u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

        def greedy_fill(adj):
            # Try to add all possible missing edges
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        if is_valid_add(adj, i, j):
                            add_edge(adj, i, j)
            # Recount
            for i in range(n):
                count += adj[i].bit_count()
            return count // 2

        def local_search(adj):
            improved = True
            while improved and time.time() < deadline:
                improved = False
                missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] >> j) & 1]
                random.shuffle(missing)
                for u, v in missing:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        improved = True
                # Optional: try swaps if stuck?
                # For now, greedy add is the main driver.
            return adj

        def initialize_population():
            pop = []
            # 1 greedy start
            adj0 = [0] * n
            order = all_edges[:]
            random.shuffle(order)
            for u, v in order:
                if is_valid_add(adj0, u, v):
                    add_edge(adj0, u, v)
            adj0 = local_search(adj0)
            pop.append(adj0)
            
            # Rest: random sparse graphs
            for _ in range(49):
                adj = [0] * n
                for u, v in all_edges:
                    if random.random() < 0.1: # Sparse
                        if is_valid_add(adj, u, v):
                            add_edge(adj, u, v)
                adj = local_search(adj)
                pop.append(adj)
            return pop

        def crossover(p1, p2):
            child = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        if (p1[i] >> j) & 1:
                            add_edge(child, i, j)
                    else:
                        if (p2[i] >> j) & 1:
                            add_edge(child, i, j)
            return child

        def mutate(adj):
            # Flip a few edges
            for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if u > v: u, v = v, u
                if (adj[u] >> v) & 1:
                    remove_edge(adj, u, v)
                else:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
            return adj

        def get_fitness(adj):
            return sum(adj[i].bit_count() for i in range(n)) // 2

        pop = initialize_population()
        
        while time.time() < deadline - 5:
            # Evaluate
            fitnesses = [get_fitness(adj) for adj in pop]
            best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
            if fitnesses[best_idx] > best_count:
                best_count = fitnesses[best_idx]
                # Save best
                best_adj = pop[best_idx]
                best_edges = []
                for i in range(n):
                    mask = best_adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            best_edges.append([i, j])
                        mask ^= lsb

            # Selection and Crossover
            new_pop = [pop[best_idx][:]] # Elitism
            
            while len(new_pop) < len(pop):
                # Tournament selection
                p1_idx = random.randint(0, len(pop) - 1)
                p2_idx = random.randint(0, len(pop) - 1)
                if fitnesses[p1_idx] > fitnesses[p2_idx]:
                    p1 = pop[p1_idx]
                else:
                    p1 = pop[p2_idx]
                
                p3_idx = random.randint(0, len(pop) - 1)
                p4_idx = random.randint(0, len(pop) - 1)
                if fitnesses[p3_idx] > fitnesses[p4_idx]:
                    p2 = pop[p3_idx]
                else:
                    p2 = pop[p4_idx]
                
                child = crossover(p1, p2)
                child = mutate(child)
                child = local_search(child)
                new_pop.append(child)
            
            pop = new_pop

        return best_edges
    