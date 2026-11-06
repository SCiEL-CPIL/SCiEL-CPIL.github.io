import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 115s.
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        adj is a list of integers (bitmasks) where adj[i] has bit j set if (i,j) is an edge.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 3. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # We need current common(v, w) <= 1.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            # Calculate common neighbors of v and w
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 4. Check constraints on neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # We need current common(u, w) <= 1.
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

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        """
        Try to add as many edges as possible from the missing list.
        """
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count

    def repair_union(adj1, adj2):
        """
        Take the union of two valid graphs and greedily remove edges to fix K2,3 violations.
        Returns (adj, count) of a valid graph.
        """
        adj = [0] * n
        for i in range(n):
            adj[i] = adj1[i] | adj2[i]
        
        count = sum(adj[i].bit_count() for i in range(n)) // 2
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Find all violating pairs
            violations = []
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[u] & adj[v]
                    if common.bit_count() > 2:
                        violations.append((u, v, common))
            
            if not violations:
                break
            
            # Pick a random violation
            u, v, common = random.choice(violations)
            
            # Remove a random edge from a common neighbor to fix the violation
            common_list = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                common_list.append(w)
                mask ^= lsb
            
            w = random.choice(common_list)
            if random.random() < 0.5:
                remove_edge(adj, u, w)
            else:
                remove_edge(adj, v, w)
            count -= 1
            improved = True
        
        return adj, count

    def simulated_annealing(adj, count):
        """
        Simulated annealing to escape local optima.
        """
        if time.time() > deadline - 5:
            return count
        
        T = 2.0
        cooling_rate = 0.999
        min_T = 0.05
        
        while T > min_T and time.time() < deadline - 2:
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
            elif move_type < 0.8:
                # Kick: Remove a present edge and refill
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    count = greedy_fill(adj, count)
            else:
                # Swap: Remove one, try to add another
                present = get_present_edges(adj)
                missing = get_missing_edges(adj)
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    
                    remove_edge(adj, u1, v1)
                    count -= 1
                    
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        count += 1
                    else:
                        # Revert if invalid
                        add_edge(adj, u1, v1)
                        count += 1
            
            T *= cooling_rate
            
        return count

    # Phase 1: Build a population of diverse solutions
    population = []
    pop_size = 15
    
    for _ in range(pop_size):
        if time.time() > deadline - 30:
            break
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        count = greedy_fill(adj, count)
        population.append((adj[:], count))
    
    population.sort(key=lambda x: -x[1])
    
    if population:
        best_count = max(best_count, population[0][1])
        best_adj = population[0][0][:]
    
    # Phase 2: Genetic algorithm with crossover
    while time.time() < deadline - 15 and population:
        # Selection
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, min(k, len(pop)))
            return max(candidates, key=lambda x: x[1])
        
        new_population = []
        
        # Elitism
        for adj, count in population[:2]:
            new_population.append((adj[:], count))
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline - 15:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover: Union + Repair
            child_adj, child_count = repair_union(parent1[0], parent2[0])
            child_count = greedy_fill(child_adj, child_count)
            new_population.append((child_adj, child_count))
            
            # Mutation
            if random.random() < 0.3 and len(new_population) < pop_size:
                parent = random.choice(population)
                adj = parent[0][:]
                count = parent[1]
                
                present = get_present_edges(adj)
                if present:
                    for _ in range(random.randint(1, 3)):
                        if not present:
                            break
                        u, v = random.choice(present)
                        remove_edge(adj, u, v)
                        count -= 1
                        present.remove((u, v))
                
                count = greedy_fill(adj, count)
                new_population.append((adj, count))
        
        # Update best
        for adj, count in new_population:
            if count > best_count:
                best_count = count
                best_adj = adj[:]
        
        # Update population
        population = sorted(new_population, key=lambda x: -x[1])[:pop_size]
    
    # Phase 3: Simulated annealing on best solution
    if best_count > 0 and time.time() < deadline - 5:
        adj = best_adj[:]
        count = best_count
        count = simulated_annealing(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    # Construct the output list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    # Final verification and safety fallback
    if edges:
        verify_adj = [0] * n
        for u, v in edges:
            verify_adj[u] |= (1 << v)
            verify_adj[v] |= (1 << u)
        
        valid = True
        for u in range(n):
            for v in range(u+1, n):
                if (verify_adj[u] & verify_adj[v]).bit_count() > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a simple valid construction if something went wrong
            adj = [0] * n
            count = 0
            random.shuffle(all_edges)
            for u, v in all_edges:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
            best_adj = adj
            
            edges = []
            for i in range(n):
                mask = best_adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    mask ^= lsb

    return edges

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
    # Verify validity
    n = 20
    adj = [0] * n
    for u, v in res:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                valid = False
                print(f"Invalid: {u}, {v} have {common.bit_count()} common neighbors")
                break
        if not valid:
            break
    
    print(f"Valid: {valid}")
