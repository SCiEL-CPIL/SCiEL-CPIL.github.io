import random
import time

def construct():
    n = 20
    start_time = time.time()
    time_limit = 140.0
    
    # Total possible edges
    total_edges = n * (n - 1) // 2
    
    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def is_valid(adj):
        # Check K2,3 free
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    def fitness(adj):
        # Higher is better
        # Penalize violations heavily to guide towards validity
        violations = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                c = bin(common).count('1')
                if c > 2:
                    violations += (c - 2)
        
        edges = count_edges(adj)
        return edges - 100 * violations

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def random_graph():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.3: # Sparse initial
                    add_edge(adj, u, v)
        return adj

    def greedy_repair(adj):
        # Remove edges that cause violations until valid
        # This is a simple local search to get valid individuals
        valid = False
        attempts = 0
        while not is_valid(adj) and attempts < 1000:
            attempts += 1
            # Find a violating pair
            viol_u, viol_v = -1, -1
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if bin(common).count('1') > 2:
                        viol_u, viol_v = u, v
                        break
                if viol_u != -1:
                    break
            
            if viol_u == -1:
                break
                
            # Remove a random edge incident to viol_u or viol_v
            # Prefer edges that are part of the common neighborhood?
            # Just pick random incident edge
            if random.random() < 0.5:
                neighbors = bin(adj[viol_u]).count('1')
                if neighbors > 0:
                    # Pick random neighbor
                    mask = adj[viol_u]
                    # Get random set bit
                    # Simple way: list of bits
                    bits = [i for i in range(n) if (adj[viol_u] >> i) & 1]
                    if bits:
                        v_rand = random.choice(bits)
                        remove_edge(adj, viol_u, v_rand)
            else:
                neighbors = bin(adj[viol_v]).count('1')
                if neighbors > 0:
                    bits = [i for i in range(n) if (adj[viol_v] >> i) & 1]
                    if bits:
                        u_rand = random.choice(bits)
                        remove_edge(adj, viol_u, u_rand) # Note: u_rand is neighbor of viol_v, so edge is (viol_v, u_rand)
                        # Wait, remove_edge(adj, viol_v, u_rand)
                        remove_edge(adj, viol_v, u_rand)
        
        return adj

    # Initialize population
    pop_size = 60
    population = []
    best_adj = None
    best_score = -1
    
    for _ in range(pop_size):
        adj = random_graph()
        adj = greedy_repair(adj)
        population.append(adj)
        score = fitness(adj)
        if score > best_score:
            best_score = score
            best_adj = adj[:]

    # Genetic Algorithm Loop
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        new_population = []
        
        # Elitism
        sorted_pop = sorted(population, key=fitness, reverse=True)
        new_population.append(sorted_pop[0][:])
        if len(sorted_pop) > 1:
            new_population.append(sorted_pop[1][:])
            
        # Generate rest
        while len(new_population) < pop_size:
            # Tournament Selection
            p1 = random.choice(sorted_pop)
            p2 = random.choice(sorted_pop)
            parent1 = p1 if fitness(p1) > fitness(p2) else p2
            
            p1 = random.choice(sorted_pop)
            p2 = random.choice(sorted_pop)
            parent2 = p1 if fitness(p1) > fitness(p2) else p2
            
            # Uniform Crossover
            child = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    # Bit from parent1 or parent2
                    if random.random() < 0.5:
                        if (parent1[u] >> v) & 1:
                            add_edge(child, u, v)
                    else:
                        if (parent2[u] >> v) & 1:
                            add_edge(child, u, v)
            
            # Mutation
            # Add/Remove random edges
            for _ in range(random.randint(1, 5)):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if (child[u] >> v) & 1:
                        if random.random() < 0.5: # Remove with lower prob
                            remove_edge(child, u, v)
                    else:
                        if random.random() < 0.3: # Add with higher prob
                            add_edge(child, u, v)
            
            # Repair to validity
            child = greedy_repair(child)
            
            new_population.append(child)
            
            # Update best
            score = fitness(child)
            if score > best_score:
                best_score = score
                best_adj = child[:]

        population = new_population
        
        # Periodically try to improve best via local search (SA-like but on best)
        if gen % 10 == 0 and best_adj:
            # Quick local search on best_adj
            temp_adj = best_adj[:]
            improved = True
            while improved and time.time() - start_time < time_limit:
                improved = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (temp_adj[u] >> v) & 1:
                            # Try add
                            if is_valid(temp_adj): # Check validity before adding? No, is_valid is expensive.
                                pass
                            # Use fast check
                            common = temp_adj[u] & temp_adj[v]
                            if bin(common).count('1') <= 2:
                                # Check neighbors
                                valid_add = True
                                for w in range(n):
                                    if w == u or w == v: continue
                                    if (temp_adj[u] >> w) & 1:
                                        c_wv = temp_adj[w] & temp_adj[v]
                                        if bin(c_wv).count('1') > 1:
                                            valid_add = False
                                            break
                                    if (temp_adj[v] >> w) & 1:
                                        c_wu = temp_adj[w] & temp_adj[u]
                                        if bin(c_wu).count('1') > 1:
                                            valid_add = False
                                            break
                                if valid_add:
                                    add_edge(temp_adj, u, v)
                                    improved = True
                                    if fitness(temp_adj) > best_score:
                                        best_score = fitness(temp_adj)
                                        best_adj = temp_adj[:]
                                    break
                    if improved:
                        break

    # Finalize best_adj
    # Ensure it's valid
    if not is_valid(best_adj):
        # Fallback to a known valid graph if best is invalid (shouldn't happen if repair works)
        # Construct a simple valid graph: 5 disjoint K4s
        best_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    add_edge(best_adj, u, v)
    
    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
