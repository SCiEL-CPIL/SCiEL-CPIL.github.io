import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Leave margin for 150s limit
    
    # Helper to get bit count
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count("1")
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added to maintain K2,3-free property."""
        # Common neighbors of u and v must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # For each neighbor b of v, common neighbors of u and b must be <= 1
        mask_v = adj[v]
        i = 0
        while mask_v:
            if mask_v & 1:
                if popcount(adj[u] & adj[i]) > 1:
                    return False
            mask_v >>= 1
            i += 1
        
        # For each neighbor a of u, common neighbors of v and a must be <= 1
        mask_u = adj[u]
        i = 0
        while mask_u:
            if mask_u & 1:
                if popcount(adj[v] & adj[i]) > 1:
                    return False
            mask_u >>= 1
            i += 1
        
        return True
    
    def edges_to_adj(edges):
        """Convert list of edge tuples to adjacency bitmasks."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_valid_k23_free(edges):
        """Check if a set of edges forms a K2,3-free graph."""
        adj = edges_to_adj(edges)
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:  # u and v are adjacent
                    if popcount(adj[u] & adj[v]) > 2:
                        return False
                # Check all pairs for common neighbors <= 2
                if popcount(adj[u] & adj[v]) > 2:
                    return False
        return True
    
    def local_search(adj, edges, max_iterations=50):
        """Try to improve by swapping edges: remove one, add another (1-for-1 or 1-for-2)."""
        edge_set = set(edges)
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            random.shuffle(edges)
            random.shuffle(all_possible)
            
            for rem_u, rem_v in edges:
                if time.time() > deadline:
                    break
                # Try removing this edge
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                edge_set.discard((rem_u, rem_v))
                edge_set.discard((rem_v, rem_u))
                
                # First try 1-for-2 swap (remove 1, add 2)
                found_swap = False
                for add_u1, add_v1 in all_possible:
                    if (add_u1, add_v1) in edge_set or (add_v1, add_u1) in edge_set:
                        continue
                    if add_u1 == rem_u and add_v1 == rem_v:
                        continue
                    if not can_add_edge(adj, add_u1, add_v1):
                        continue
                    # Tentatively add first edge
                    adj[add_u1] |= (1 << add_v1)
                    adj[add_v1] |= (1 << add_u1)
                    edge_set.add((add_u1, add_v1))
                    
                    for add_u2, add_v2 in all_possible:
                        if (add_u2, add_v2) in edge_set or (add_v2, add_u2) in edge_set:
                            continue
                        if add_u2 == rem_u and add_v2 == rem_v:
                            continue
                        if add_u2 == add_u1 and add_v2 == add_v1:
                            continue
                        if add_u2 == add_v1 and add_v2 == add_u1:
                            continue
                        if not can_add_edge(adj, add_u2, add_v2):
                            continue
                        # Found valid 1-for-2 swap
                        adj[add_u2] |= (1 << add_v2)
                        adj[add_v2] |= (1 << add_u2)
                        edge_set.add((add_u2, add_v2))
                        # Update edges list
                        idx = edges.index((rem_u, rem_v))
                        edges[idx] = (add_u1, add_v1)
                        edges.append((add_u2, add_v2))
                        improved = True
                        found_swap = True
                        break
                    
                    if found_swap:
                        break
                    # Undo first addition
                    adj[add_u1] &= ~(1 << add_v1)
                    adj[add_v1] &= ~(1 << add_u1)
                    edge_set.discard((add_u1, add_v1))
                
                if found_swap:
                    break
                
                # Try 1-for-1 swap (remove 1, add 1)
                for add_u, add_v in all_possible:
                    if (add_u, add_v) in edge_set or (add_v, add_u) in edge_set:
                        continue
                    if add_u == rem_u and add_v == rem_v:
                        continue
                    if can_add_edge(adj, add_u, add_v):
                        adj[add_u] |= (1 << add_v)
                        adj[add_v] |= (1 << add_u)
                        edge_set.add((add_u, add_v))
                        edges[edges.index((rem_u, rem_v))] = (add_u, add_v)
                        improved = True
                        break
                else:
                    # Couldn't find a swap, put back
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                    edge_set.add((rem_u, rem_v))
                    continue
                break  # Found a swap, restart
            if time.time() > deadline:
                break
        
        return edges
    
    def greedy_construct(order):
        """Greedy construction from a given edge ordering."""
        adj = [0] * n
        current_edges = []
        
        for u, v in order:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        return current_edges, adj
    
    def genetic_crossover(parent1, parent2):
        """
        Validity-preserving crossover: take disjoint edge subsets from parents.
        Only combine edges that don't conflict with each other.
        """
        adj1 = edges_to_adj(parent1)
        adj2 = edges_to_adj(parent2)
        
        # Build combined adjacency for checking
        combined_adj = [0] * n
        for i in range(n):
            combined_adj[i] = adj1[i] | adj2[i]
        
        # Try to build child by adding edges from parents in random order
        all_parent_edges = parent1 + parent2
        random.shuffle(all_parent_edges)
        
        child_adj = [0] * n
        child_edges = []
        
        for u, v in all_parent_edges:
            if can_add_edge(child_adj, u, v):
                child_adj[u] |= (1 << v)
                child_adj[v] |= (1 << u)
                child_edges.append((u, v))
        
        return child_edges
    
    def mutation(edges, adj, rate=0.1):
        """Random mutation: try to add or remove random edges."""
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        edge_set = set(edges)
        
        # Try some additions
        for _ in range(int(rate * len(all_possible))):
            u, v = random.choice(all_possible)
            if (u, v) not in edge_set and (v, u) not in edge_set:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append((u, v))
        
        # Try some removals (less likely)
        for _ in range(int(rate * len(edges) * 0.5)):
            if edges:
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_set.discard((u, v))
                edge_set.discard((v, u))
                edges.pop(idx)
        
        return edges, adj
    
    def ga_search(pop_size=12, generations=8, mutation_rate=0.15):
        """Genetic algorithm with edge-bitset chromosomes."""
        # Initialize population with diverse greedy solutions
        population = []
        for i in range(pop_size):
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            edges, adj = greedy_construct(all_edges)
            # Local search to improve initial individuals
            if time.time() < deadline:
                edges = local_search(adj, edges, max_iterations=20)
                adj = edges_to_adj(edges)
            population.append(edges)
        
        best_edges = max(population, key=len)
        best_count = len(best_edges)
        
        gen = 0
        while gen < generations and time.time() < deadline:
            gen += 1
            new_population = []
            
            # Elitism: keep best individuals
            population.sort(key=len, reverse=True)
            elite_count = max(2, pop_size // 4)
            for i in range(elite_count):
                new_population.append(population[i][:])
            
            # Create offspring via crossover and mutation
            while len(new_population) < pop_size and time.time() < deadline:
                # Select parents (tournament selection)
                p1 = random.choice(population[:max(4, pop_size // 2)])
                p2 = random.choice(population[:max(4, pop_size // 2)])
                
                # Crossover
                child = genetic_crossover(p1, p2)
                
                # Mutation
                child_adj = edges_to_adj(child)
                child, child_adj = mutation(child, child_adj, rate=mutation_rate)
                
                # Local search to polish
                if time.time() < deadline:
                    child = local_search(child_adj, child, max_iterations=15)
                    child_adj = edges_to_adj(child)
                
                new_population.append(child)
                
                # Update best
                if len(child) > best_count:
                    best_count = len(child)
                    best_edges = child[:]
            
            population = new_population
        
        return best_edges
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Strategy 1: Multiple greedy + local search runs
    greedy_budget = time.time() + 40
    strategy_count = 0
    while time.time() < greedy_budget:
        strategy_count += 1
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        current_edges, adj = greedy_construct(edges_copy)
        
        if time.time() < deadline:
            current_edges = local_search(adj, current_edges[:])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
    
    # Strategy 2: Genetic algorithm for the remaining time
    if time.time() < deadline:
        ga_result = ga_search(pop_size=10, generations=6, mutation_rate=0.12)
        if len(ga_result) > best_count:
            best_count = len(ga_result)
            best_edges = ga_result[:]
    
    # Final polish: local search on the best result
    if best_edges and time.time() < deadline:
        adj = edges_to_adj(best_edges)
        best_edges = local_search(adj, best_edges[:], max_iterations=30)
    
    # Verify validity
    if best_edges and not is_valid_k23_free(best_edges):
        # Fallback: rebuild from scratch with simple greedy
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        best_edges, _ = greedy_construct(edges_copy)
    
    # Convert list of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
