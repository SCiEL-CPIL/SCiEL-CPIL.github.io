import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator kills at 150s)
    
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v)\{u}, pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = lsb.bit_length() - 1
            neighbors_v ^= lsb
            if popcount(adj[u] & adj[w]) > 1:
                return False
                
        # 3. For all w in N(u)\{v}, pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = lsb.bit_length() - 1
            neighbors_u ^= lsb
            if popcount(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        total_deg = 0
        for i in range(n):
            total_deg += popcount(adj[i])
        return total_deg // 2

    def get_neighbors(i):
        result = []
        temp = adj[i]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            result.append(w)
            temp ^= lsb
        return result

    # === Strategy: Multi-start with diverse initialization and aggressive local search ===
    
    best_adj = [0] * n
    best_count = 0
    
    def greedy_fill(order):
        nonlocal adj
        adj = [0] * n
        count = 0
        for u, v in order:
            if can_add(u, v):
                add_edge(u, v)
                count += 1
        return count

    def local_search(initial_adj, initial_count, time_limit=10.0):
        nonlocal adj
        adj = initial_adj[:]
        count = initial_count
        t0 = time.time()
        
        # Phase 1: Try to add any possible edges (greedy completion)
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                continue
            if can_add(u, v):
                add_edge(u, v)
                count += 1
        
        # Phase 2: Simulated Annealing with swaps and removals
        T = 5.0
        cooling = 0.9995
        min_T = 0.05
        
        while time.time() < t0 + time_limit:
            if T < min_T:
                T = min_T
            
            r = random.random()
            
            if r < 0.5:
                # Try to add an edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    count += 1
                # Always accept if valid (delta = -1)
                    
            elif r < 0.8:
                # Try to remove an edge
                u = random.randint(0, n - 1)
                neighbors = adj[u]
                if neighbors == 0:
                    continue
                neigh_list = get_neighbors(u)
                if not neigh_list:
                    continue
                v = random.choice(neigh_list)
                
                remove_edge(u, v)
                delta = 1  # count decreases by 1
                if random.random() < math.exp(-delta / T):
                    count -= 1
                else:
                    add_edge(u, v)
                    
            else:
                # Swap: remove one edge, add another
                u1 = random.randint(0, n - 1)
                neighbors1 = adj[u1]
                if neighbors1 == 0:
                    continue
                neigh_list1 = get_neighbors(u1)
                if not neigh_list1:
                    continue
                v1 = random.choice(neigh_list1)
                
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                if (adj[u2] >> v2) & 1:
                    continue
                
                remove_edge(u1, v1)
                if can_add(u2, v2):
                    add_edge(u2, v2)
                    # Count stays the same, always accept for exploration
                else:
                    add_edge(u1, v1)
                    
            T *= cooling
        
        return adj[:], count

    # Multiple restarts with different initial structures
    n_restarts = 8
    restart_time = (deadline - start_time - 5.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Different initialization strategies
        if restart == 0:
            # Random edge order
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            count = greedy_fill(all_edges)
        elif restart == 1:
            # Start with a sparse random graph, then fill
            adj = [0] * n
            for _ in range(15):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and not ((adj[u] >> v) & 1):
                    add_edge(u, v)
            count = edge_count()
        else:
            # Perturb the best solution found so far
            adj = best_adj[:]
            count = best_count
            # Remove some random edges to create space
            for _ in range(random.randint(3, 8)):
                if count == 0:
                    break
                u = random.randint(0, n - 1)
                neighbors = adj[u]
                if neighbors == 0:
                    continue
                neigh_list = get_neighbors(u)
                if not neigh_list:
                    continue
                v = random.choice(neigh_list)
                remove_edge(u, v)
                count -= 1
        
        # Local search with adaptive time
        time_left = deadline - start_time - 5.0
        if time_left <= 0:
            break
        ls_time = min(restart_time, time_left * 0.6)
        if ls_time < 1.0:
            ls_time = max(0.5, time_left * 0.3)
            
        new_adj, new_count = local_search(adj[:], count, ls_time)
        
        if new_count > best_count:
            best_count = new_count
            best_adj = new_adj
    
    # Final greedy completion on best solution
    if time.time() < deadline - 2.0:
        adj = best_adj[:]
        count = best_count
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                continue
            if can_add(u, v):
                add_edge(u, v)
                count += 1
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
