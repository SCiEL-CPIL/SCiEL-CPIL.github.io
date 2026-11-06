import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Target n
    n_target = 42
    
    # Helper to generate Paley 37 edges
    def get_paley_edges(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return edges

    paley_edges = get_paley_edges(37)
    
    # Function to count edges in a subgraph defined by a bitmask
    # adj_mask is a list of ints
    def count_edges_in_mask(mask, adj_mask):
        count = 0
        m = mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            # Count bits in intersection of adj_mask[idx] and remaining m
            count += (adj_mask[idx] & m).bit_count()
        return count

    # Function to calculate total cost (K5 + Ind5) for a graph represented by adj_mask
    # This is used for final verification and initial cost
    def calculate_total_cost(adj_mask, n):
        k5_count = 0
        ind5_count = 0
        # Iterate over all 5-combinations
        for comb in itertools.combinations(range(n), 5):
            v1, v2, v3, v4, v5 = comb
            
            # K5 check
            mask = adj_mask[v1] & adj_mask[v2] & adj_mask[v3] & adj_mask[v4]
            if mask & (1 << v5):
                k5_count += 1
                
            # Independent set check
            union_mask = adj_mask[v1] | adj_mask[v2] | adj_mask[v3] | adj_mask[v4]
            if not (union_mask & (1 << v5)):
                ind5_count += 1
                
        return k5_count + ind5_count

    # Helper to initialize the graph with Paley 37 + random extension
    def initialize_graph(seed_val):
        adj = [0] * n_target
        all_mask = (1 << n_target) - 1
        
        for u, v in paley_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        rng = random.Random(seed_val)
        for i in range(37, n_target):
            for j in range(i + 1, n_target):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            for k in range(i):
                if rng.random() < 0.5:
                    adj[i] |= (1 << k)
                    adj[k] |= (1 << i)
        
        return adj

    # Simulated annealing function
    def simulated_annealing(adj_mask, time_limit, T_initial=5.0, T_min=0.01):
        current_cost = calculate_total_cost(adj_mask, n_target)
        best_cost = current_cost
        best_adj = adj_mask[:]
        all_vertices_mask = (1 << n_target) - 1
        
        steps = 0
        total_steps = 200000  # Estimate for time allocation
        log_T_initial = math.log(T_initial)
        log_T_min = math.log(T_min)
        
        while time.time() < time_limit:
            # Pick random edge to flip
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            edge_exists = bool(adj_mask[u] & (1 << v))
            
            # Calculate Delta Cost
            common_neigh = adj_mask[u] & adj_mask[v]
            common_neigh &= ~(1 << u)
            common_neigh &= ~(1 << v)
            delta_k5 = count_edges_in_mask(common_neigh, adj_mask)
            
            common_non_neigh = ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
            common_non_neigh &= ~(1 << u)
            common_non_neigh &= ~(1 << v)
            k = common_non_neigh.bit_count()
            total_pairs = k * (k - 1) // 2
            edges_in_g = count_edges_in_mask(common_non_neigh, adj_mask)
            delta_ind = total_pairs - edges_in_g
            
            if edge_exists:
                delta_cost = -delta_k5 + delta_ind
            else:
                delta_cost = delta_k5 - delta_ind
            
            # Temperature schedule
            progress = min(steps / total_steps, 1.0)
            log_T = log_T_initial + progress * (log_T_min - log_T_initial)
            T = math.exp(log_T)
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                if edge_exists:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
                    
                    if best_cost == 0:
                        if calculate_total_cost(best_adj, n_target) == 0:
                            return best_adj, 0

            steps += 1
            
            # Periodic full check to correct drift
            if steps % 3000 == 0:
                current_cost = calculate_total_cost(adj_mask, n_target)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
                    if best_cost == 0:
                        if calculate_total_cost(best_adj, n_target) == 0:
                            return best_adj, 0

        return best_adj, best_cost

    # Try multiple random initializations with different seeds
    best_final_adj = None
    best_final_cost = float('inf')
    
    # Allocate time for multiple attempts
    num_attempts = 5
    time_per_attempt = (deadline - start_time) / num_attempts
    
    for attempt in range(num_attempts):
        if time.time() >= deadline:
            break
        
        seed_val = 42 + attempt * 1000
        adj = initialize_graph(seed_val)
        
        # Give more time to early attempts, less to later ones
        attempt_deadline = start_time + (attempt + 1) * time_per_attempt
        attempt_deadline = min(attempt_deadline, deadline)
        
        adj, cost = simulated_annealing(adj, attempt_deadline)
        
        if cost < best_final_cost:
            best_final_cost = cost
            best_final_adj = adj[:]
            
            if cost == 0:
                # Found valid solution, verify and return
                if calculate_total_cost(best_final_adj, n_target) == 0:
                    edges = []
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            if best_final_adj[i] & (1 << j):
                                edges.append((i, j))
                    return n_target, edges

    # Final polish on the best solution found
    if best_final_adj is not None and best_final_cost > 0:
        polish_deadline = deadline
        polish_adj = best_final_adj[:]
        polish_cost = best_final_cost
        all_vertices_mask = (1 << n_target) - 1
        polish_T = 0.1
        
        while time.time() < polish_deadline and polish_cost > 0:
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            edge_exists = bool(polish_adj[u] & (1 << v))
            
            common_neigh = polish_adj[u] & polish_adj[v]
            common_neigh &= ~(1 << u)
            common_neigh &= ~(1 << v)
            delta_k5 = count_edges_in_mask(common_neigh, polish_adj)
            
            common_non_neigh = ~(polish_adj[u] | polish_adj[v]) & all_vertices_mask
            common_non_neigh &= ~(1 << u)
            common_non_neigh &= ~(1 << v)
            k = common_non_neigh.bit_count()
            total_pairs = k * (k - 1) // 2
            edges_in_g = count_edges_in_mask(common_non_neigh, polish_adj)
            delta_ind = total_pairs - edges_in_g
            
            if edge_exists:
                delta_cost = -delta_k5 + delta_ind
            else:
                delta_cost = delta_k5 - delta_ind
            
            if delta_cost < 0 or (delta_cost == 0 and random.random() < 0.5) or random.random() < math.exp(-delta_cost / polish_T):
                if edge_exists:
                    polish_adj[u] &= ~(1 << v)
                    polish_adj[v] &= ~(1 << u)
                else:
                    polish_adj[u] |= (1 << v)
                    polish_adj[v] |= (1 << u)
                    
                polish_cost += delta_cost
                
                if polish_cost < best_final_cost:
                    best_final_cost = polish_cost
                    best_final_adj = polish_adj[:]
                    
                    if best_final_cost == 0:
                        if calculate_total_cost(best_final_adj, n_target) == 0:
                            edges = []
                            for i in range(n_target):
                                for j in range(i + 1, n_target):
                                    if best_final_adj[i] & (1 << j):
                                        edges.append((i, j))
                            return n_target, edges
                
                polish_T = max(0.001, polish_T * 0.9995)

    # If we found a valid 42-graph, return it
    if best_final_cost == 0 and best_final_adj is not None:
        if calculate_total_cost(best_final_adj, n_target) == 0:
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if best_final_adj[i] & (1 << j):
                        edges.append((i, j))
            return n_target, edges

    # Fallback: Return Paley 37
    return 37, paley_edges
