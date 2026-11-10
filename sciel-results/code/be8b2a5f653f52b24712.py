import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    n = 20
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Helper functions
    def can_add(adj, u, v):
        return (adj[u] & adj[v]).bit_count() < 3
        
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                edges.append([u, v])
                mask -= lsb
        return edges
        
    def adj_to_count(adj):
        return sum(a.bit_count() for a in adj) // 2

    # Randomized Greedy
    def greedy_run(seed):
        random.seed(seed)
        adj = [0] * n
        edges_list = all_edges[:]
        random.shuffle(edges_list)
        count = 0
        for u, v in edges_list:
            if can_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    # Local Search: Try to add non-edges, if blocked, try swap
    def local_search(adj, time_limit):
        current_count = adj_to_count(adj)
        best_adj = adj[:]
        best_count = current_count
        
        # Pre-calculate non-edges is expensive, just pick random pairs
        while time.time() < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            if adj[u] & (1 << v):
                continue # Already an edge
                
            if can_add(adj, u, v):
                add_edge(adj, u, v)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Blocked
                common = adj[u] & adj[v]
                if common:
                    # Pick a random common neighbor
                    mask = common
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    
                    # Try removing (u, w)
                    remove_edge(adj, u, w)
                    current_count -= 1
                    if can_add(adj, u, v):
                        add_edge(adj, u, v)
                        current_count += 1
                        # Try to greedily add more? 
                        # Just check if improved
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        # Revert
                        add_edge(adj, u, w)
                        current_count += 1
                        
                    # Try removing (v, w)
                    if current_count <= best_count:
                        remove_edge(adj, v, w)
                        current_count -= 1
                        if can_add(adj, u, v):
                            add_edge(adj, u, v)
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_adj = adj[:]
                        else:
                            add_edge(adj, v, w)
                            current_count += 1

        return best_adj, best_count

    # Main Loop
    iterations = 0
    # Run greedy restarts for 80% of the time
    greedy_deadline = start_time + (deadline - start_time) * 0.8
    
    while time.time() < greedy_deadline:
        seed = iterations
        adj, count = greedy_run(seed)
        
        if count > best_count:
            best_count = count
            best_edges = get_edges(adj)
            
        iterations += 1
        
    # Use remaining time for local search on the best candidate
    if best_count > 0:
        # Reconstruct adj from best_edges
        adj = [0] * n
        for u, v in best_edges:
            add_edge(adj, u, v)
            
        best_adj, best_count_ls = local_search(adj, deadline)
        if best_count_ls > best_count:
            best_count = best_count_ls
            best_edges = get_edges(best_adj)
            
    return best_edges
