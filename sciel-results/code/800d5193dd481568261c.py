import random
import time
import math

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def is_valid_addition(adj, u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False

        # 2. For any w in N(v), v becomes a common neighbor of u and w.
        #    So common neighbors of (u, w) must currently be < 2.
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            if w != u:
                common_uw = adj[u] & adj[w]
                if popcount(common_uw) >= 2:
                    return False
            neighbors_v ^= lsb

        # 3. Similarly for w in N(u)
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            if w != v:
                common_vw = adj[v] & adj[w]
                if popcount(common_vw) >= 2:
                    return False
            neighbors_u ^= lsb

        return True

    def try_add_edge(adj, u, v):
        if is_valid_addition(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def greedy_build():
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            try_add_edge(adj, u, v)
        return adj

    def local_search_add(adj):
        improved = True
        while improved:
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n)
                       if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(adj, u, v):
                    improved = True

    best_edges_list = []
    best_count = 0

    # Run Simulated Annealing
    # We will do multiple short SA runs or one long one.
    # Let's do one long SA with reheats, starting from a greedy solution.
    
    adj = greedy_build()
    local_search_add(adj)
    
    current_count = count_edges(adj)
    if current_count > best_count:
        best_count = current_count
        best_edges_list = [[i, j] for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]

    T = 1.0
    min_T = 0.01
    cooling = 0.9995
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
        
        if not existing:
            break
        if not missing:
            # Graph is complete? Unlikely for K2,3 free.
            break

        # 30% Add, 30% Remove, 40% Swap
        if move_type < 0.3:
            # Try to add a random missing edge
            u, v = random.choice(missing)
            if try_add_edge(adj, u, v):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges_list = [[i, j] for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                    no_improve_count = 0
            else:
                no_improve_count += 1
                
        elif move_type < 0.6:
            # Try to remove a random existing edge
            u, v = random.choice(existing)
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_count -= 1
            
            # Accept if random < exp(-delta/T), delta = -1, so exp(1/T)
            # Actually, we are going DOWN in objective.
            # Objective is #edges. We want to maximize.
            # Delta = new_obj - old_obj = -1.
            # Accept prob = exp(-Delta/T) = exp(1/T).
            # Wait, standard SA for maximization:
            # Accept if delta > 0 OR random < exp(delta/T).
            # Here delta = -1. exp(-1/T).
            # So accept with prob exp(-1/T).
            if random.random() < math.exp(-1.0 / T):
                # Accepted
                no_improve_count += 1
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_count += 1
                no_improve_count += 1

        else:
            # Swap: Remove one, try to add another
            u, v = random.choice(existing)
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Try to add a random missing edge (from the list *before* removal, 
            # but we must check if it's still missing/valid)
            # Note: (u,v) is now missing. We should pick from missing + (u,v)
            # But (u,v) was existing, so it wasn't in 'missing'.
            # We can just pick from 'missing' or pick (u,v) again? 
            # If we pick (u,v) again, it's a no-op (remove then add same). 
            # Better to pick from 'missing' to change structure.
            
            if missing:
                x, y = random.choice(missing)
                # Check if (x,y) is still missing (it is, unless x,y == u,v which is false)
                # And check if adding (x,y) is valid
                if try_add_edge(adj, x, y):
                    # Net change 0
                    # Accept always? Or accept with prob 1?
                    # It's a neutral move. Accept to explore.
                    # current_count unchanged
                    no_improve_count += 1
                else:
                    # Revert removal of (u,v)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    no_improve_count += 1
            else:
                # No missing edges to swap to? Just revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                no_improve_count += 1

        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Reheat if stuck
        if no_improve_count > max_no_improve:
            T = 1.0
            no_improve_count = 0

    # Final polish
    local_search_add(adj)
    final_count = count_edges(adj)
    if final_count > best_count:
        best_count = final_count
        best_edges_list = [[i, j] for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]

    return best_edges_list
