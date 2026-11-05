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

    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        return edge_indices[(u, v)]

    # Bitmask adjacency
    # adj[u] is an int with bit v set if (u, v) is an edge
    def init_graph():
        adj = [0] * n
        # Start with 5 disjoint K4s
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def is_valid_addition(adj, u, v):
        # Check if adding (u, v) keeps graph K2,3-free
        # u and v are not connected yet
        # We need |N(u) \cap N(w)| <= 1 for all w in N(v)
        # and |N(v) \cap N(w)| <= 1 for all w in N(u)
        # Note: N(u) and N(v) in the current graph adj
        
        # Check for u
        neighbors_v = adj[v]
        # Iterate through neighbors of v
        temp_v = neighbors_v
        while temp_v:
            # Get lowest set bit
            lsb = temp_v & -temp_v
            w = lsb.bit_length() - 1
            temp_v ^= lsb
            
            # Check common neighbors of u and w
            common = adj[u] & adj[w]
            if bin(common).count('1') > 1:
                return False
        
        # Check for v
        neighbors_u = adj[u]
        temp_u = neighbors_u
        while temp_u:
            lsb = temp_u & -temp_u
            w = lsb.bit_length() - 1
            temp_u ^= lsb
            
            # Check common neighbors of v and w
            common = adj[v] & adj[w]
            if bin(common).count('1') > 1:
                return False
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Main search
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    current_adj = init_graph()
    current_edges = edge_count(current_adj)
    
    best_adj = [x for x in current_adj]
    best_edges = current_edges

    T = 1.0
    cooling_rate = 0.9999
    
    # Precompute list of all possible edges for random selection
    all_edges = list(edge_indices.keys())
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add a random edge
            u, v = random.choice(all_edges)
            if not ((current_adj[u] >> v) & 1):
                if is_valid_addition(current_adj, u, v):
                    # Accept
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [x for x in current_adj]
                    T = 1.0 # Reset temperature on improvement? Or keep cooling?
                    # Let's keep cooling but boost if stuck
                else:
                    # Failed addition, maybe cool slightly more
                    T *= 0.99
            else:
                T *= 0.999
        else:
            # Try a swap: remove one edge, add another
            # Pick edge to remove
            # Prefer edges that are "redundant" or part of dense clusters?
            # Just random for now.
            u, v = random.choice(all_edges)
            if ((current_adj[u] >> v) & 1):
                # Pick edge to add
                x, y = random.choice(all_edges)
                if ((current_adj[x] >> y) & 1) or (x == u and y == v) or (x == v and y == u):
                    T *= 0.999
                    continue
                
                # Perform swap on a copy
                new_adj = [a for a in current_adj]
                remove_edge(new_adj, u, v)
                
                # Check if adding (x, y) is valid in new_adj
                if is_valid_addition(new_adj, x, y):
                    add_edge(new_adj, x, y)
                    new_edges = current_edges # Count is same
                    
                    # Accept swap?
                    # Since edge count is same, we accept to explore.
                    # But we should be careful not to cycle.
                    # For now, always accept valid swaps.
                    current_adj = new_adj
                    # T stays same or cools slowly
                    T *= 0.999
                else:
                    T *= 0.99
            else:
                T *= 0.999
        
        # Cooling
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Convert best_adj to list of edges
    result = []
    for (u, v), idx in edge_indices.items():
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
    return result
