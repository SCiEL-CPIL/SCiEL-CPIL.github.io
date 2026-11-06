import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        
        # Common neighbors of u and v must be <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # For each w in N(u), common(v, w) must be <= 1
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if (adj[v] & adj[w]).bit_count() > 1:
                return False
            w_mask ^= lsb
        
        # For each w in N(v), common(u, w) must be <= 1
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if (adj[u] & adj[w]).bit_count() > 1:
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
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count

    def remove_k23_violations(adj, count):
        """
        Remove edges to fix any K2,3 violations.
        A K2,3 violation exists if there are two vertices u, v with >= 3 common neighbors.
        We remove edges incident to the common neighbors or between u and v.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    cnt = common.bit_count()
                    if cnt > 2:
                        # We need to reduce common neighbors to <= 2
                        # Strategy: remove edge (u, w) or (v, w) for some w in common
                        # Pick a random w from common to break
                        common_list = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            w = lsb.bit_length() - 1
                            common_list.append(w)
                            mask ^= lsb
                        
                        # Remove one edge incident to a common neighbor
                        w = random.choice(common_list)
                        
                        # Choose to remove (u, w) or (v, w)
                        if random.random() < 0.5:
                            remove_edge(adj, u, w)
                            count -= 1
                        else:
                            remove_edge(adj, v, w)
                            count -= 1
                        changed = True
        return count

    def crossover(parent1, parent2):
        """
        Crossover: take union of edges, then repair violations, then greedy fill.
        """
        adj = [0] * n
        count = 0
        
        # Start with union
        for i in range(n):
            adj[i] = parent1[i] | parent2[i]
        
        # Count edges in union
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    count += 1
                mask ^= lsb
        
        # Repair violations
        count = remove_k23_violations(adj, count)
        
        # Greedy fill
        count = greedy_fill(adj, count)
        
        return adj, count

    def simulated_annealing(adj, count):
        if time.time() > deadline - 5:
            return count
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        while T > min_T and time.time() < deadline - 2:
            move_type = random.random()
            
            if move_type < 0.4:
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
            elif move_type < 0.8:
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    count = greedy_fill(adj, count)
            else:
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
                        add_edge(adj, u1, v1)
                        count += 1
            
            T *= cooling_rate
        
        return count

    # Population-based genetic algorithm
    pop_size = 8
    population = []
    
    # Initialize population with random greedy fills
    for _ in range(pop_size):
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        count = greedy_fill(adj, count)
        population.append((adj, count))
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Genetic algorithm loop
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        
        # Sort by fitness
        population.sort(key=lambda x: x[1], reverse=True)
        
        # Elitism: keep top 2
        new_pop = [population[0], population[1]]
        
        # Generate offspring
        while len(new_pop) < pop_size and time.time() < deadline - 10:
            # Selection: tournament selection
            idx1 = random.randint(0, min(3, len(population) - 1))
            idx2 = random.randint(0, min(3, len(population) - 1))
            if population[idx1][1] >= population[idx2][1]:
                parent1 = population[idx1][0]
            else:
                parent1 = population[idx2][0]
            
            idx3 = random.randint(0, min(3, len(population) - 1))
            idx4 = random.randint(0, min(3, len(population) - 1))
            if population[idx3][1] >= population[idx4][1]:
                parent2 = population[idx3][0]
            else:
                parent2 = population[idx4][0]
            
            # Crossover
            child_adj, child_count = crossover(parent1, parent2)
            
            # Mutation: random edge removal and refill
            if random.random() < 0.3:
                present = get_present_edges(child_adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(child_adj, u, v)
                    child_count -= 1
                    child_count = greedy_fill(child_adj, child_count)
            
            new_pop.append((child_adj, child_count))
        
        population = new_pop
        
        # Update best
        for adj, count in population:
            if count > best_count:
                best_count = count
                best_adj = adj[:]
        
        # Run simulated annealing on the best individual
        best_adj_sa, best_count_sa = population[0]
        best_count_sa = simulated_annealing(best_adj_sa, best_count_sa)
        if best_count_sa > best_count:
            best_count = best_count_sa
            best_adj = best_adj_sa[:]
            population[0] = (best_adj_sa, best_count_sa)
    
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
            
    return edges

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
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
