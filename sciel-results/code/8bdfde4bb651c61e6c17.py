import random
import time
import math

def construct():
    n = 20
    
    # Precompute edge indices for mapping back to list
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            idx += 1
    total_edges_possible = idx
    all_edges = list(edge_indices.keys())

    def popcount(x):
        return bin(x).count('1')

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    def is_c4_free_add(adj, u, v):
        # Check if adding (u, v) to adj creates a C4
        # A C4 is created if u and v share more than 1 common neighbor in adj
        common = adj[u] & adj[v]
        # If common has 2 or more bits, it's invalid
        if common & (common - 1):
            return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def init_graphs():
        # Start with empty graphs
        adj1 = [0] * n
        adj2 = [0] * n
        return adj1, adj2

    # Main search
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    best_adj1 = [0] * n
    best_adj2 = [0] * n
    best_edges = 0

    # Multi-restart SA
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
            
        adj1, adj2 = init_graphs()
        current_edges = 0
        
        # Greedy initialization: try to add edges randomly to fill up
        random.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() > deadline - 1.0:
                break
            # Try adding to adj1
            if not ((adj1[u] >> v) & 1) and is_c4_free_add(adj1, u, v):
                add_edge(adj1, u, v)
                current_edges += 1
                continue
            # Try adding to adj2
            if not ((adj2[u] >> v) & 1) and is_c4_free_add(adj2, u, v):
                add_edge(adj2, u, v)
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj1 = [x for x in adj1]
            best_adj2 = [x for x in adj2]
            
        # SA phase
        T = 1.0
        cooling_rate = 0.9999
        restart_deadline = start_time + time_per_restart * (r + 1)
        
        while time.time() < restart_deadline:
            move_type = random.random()
            
            if move_type < 0.5:
                # Add to adj1
                u, v = random.choice(all_edges)
                if not ((adj1[u] >> v) & 1):
                    if is_c4_free_add(adj1, u, v):
                        add_edge(adj1, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj1 = [x for x in adj1]
                            best_adj2 = [x for x in adj2]
                        T = max(0.01, T * 0.99)
                    else:
                        T = max(0.01, T * 0.999)
                else:
                    T = max(0.01, T * 0.999)
            else:
                # Add to adj2
                u, v = random.choice(all_edges)
                if not ((adj2[u] >> v) & 1):
                    if is_c4_free_add(adj2, u, v):
                        add_edge(adj2, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj1 = [x for x in adj1]
                            best_adj2 = [x for x in adj2]
                        T = max(0.01, T * 0.99)
                    else:
                        T = max(0.01, T * 0.999)
                else:
                    T = max(0.01, T * 0.999)
            
            # Cooling
            T = max(0.01, T * cooling_rate)

    # Convert best_adj1 and best_adj2 to list of edges
    result = []
    for (u, v), idx in edge_indices.items():
        if ((best_adj1[u] >> v) & 1) or ((best_adj2[u] >> v) & 1):
            result.append([u, v])
    return result
