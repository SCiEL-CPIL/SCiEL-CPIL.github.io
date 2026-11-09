import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    N = 16
    target_edges = 65
    
    # Initial state: K_{8,8} + edge (0,1)
    adj = [0] * N
    deg = [0] * N
    
    def add_edge_to_adj(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1

    for u in range(8):
        for v in range(8, 16):
            add_edge_to_adj(u, v)
    add_edge_to_adj(0, 1)
    
    edge_count = target_edges
    
    def get_score(adj, deg):
        max_sum = 0
        for u in range(N):
            au = adj[u]
            du = deg[u]
            for v in range(u + 1, N):
                if (au >> v) & 1:
                    common = au & adj[v]
                    common &= ~((1 << (v + 1)) - 1) 
                    while common:
                        w_bit = common & -common
                        w = w_bit.bit_length() - 1
                        s = du + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= w_bit
        return max_sum

    best_score = get_score(adj, deg)
    best_adj = adj[:]
    best_deg = deg[:]
    
    current_adj = adj[:]
    current_deg = deg[:]
    current_score = best_score
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    def get_edge_lists(adj):
        edges = []
        non_edges = []
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                edges.append((u, v))
            else:
                non_edges.append((u, v))
        return edges, non_edges

    cur_edges, cur_non_edges = get_edge_lists(current_adj)
    
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Find worst triangles for targeting
        worst_triangles = []
        if current_score > 0:
            for u in range(N):
                au = current_adj[u]
                du = current_deg[u]
                for v in range(u + 1, N):
                    if (au >> v) & 1:
                        common = au & current_adj[v]
                        common &= ~((1 << (v + 1)) - 1)
                        while common:
                            w_bit = common & -common
                            w = w_bit.bit_length() - 1
                            s = du + current_deg[v] + current_deg[w]
                            if s == current_score:
                                worst_triangles.append((u, v, w))
                            common ^= w_bit
        
        move_type = random.random()
        
        if worst_triangles and move_type < 0.6:
            # Targeted move
            tri = random.choice(worst_triangles)
            u, v, w = tri
            # Edges in triangle
            edges_in_tri = [(u, v), (u, w), (v, w)]
            # Filter to those actually present
            valid_edges = [e for e in edges_in_tri if e in cur_edges]
            if not valid_edges:
                valid_edges = [random.choice(cur_edges)]
            edge_to_remove = random.choice(valid_edges)
            
            if not cur_non_edges:
                continue
            edge_to_add = random.choice(cur_non_edges)
        else:
            # Random swap
            if not cur_edges or not cur_non_edges:
                continue
            edge_to_remove = random.choice(cur_edges)
            edge_to_add = random.choice(cur_non_edges)
        
        # Apply move
        u1, v1 = edge_to_remove
        current_adj[u1] ^= (1 << v1)
        current_adj[v1] ^= (1 << u1)
        current_deg[u1] -= 1
        current_deg[v1] -= 1
        
        u2, v2 = edge_to_add
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        current_deg[u2] += 1
        current_deg[v2] += 1
        
        # Update lists
        cur_edges.remove(edge_to_remove)
        cur_non_edges.append(edge_to_remove)
        cur_non_edges.remove(edge_to_add)
        cur_edges.append(edge_to_add)
        
        # Evaluate
        new_score = get_score(current_adj, current_deg)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
                best_deg = current_deg[:]
        else:
            # Metropolis
            if random.random() < math.exp(-(new_score - current_score) / T):
                current_score = new_score
            else:
                # Revert
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_deg[u1] += 1
                current_deg[v1] += 1
                
                current_adj[u2] ^= (1 << v2)
                current_adj[v2] ^= (1 << u2)
                current_deg[u2] -= 1
                current_deg[v2] -= 1
                
                cur_edges.remove(edge_to_add)
                cur_non_edges.remove(edge_to_remove)
                cur_edges.append(edge_to_remove)
                cur_non_edges.append(edge_to_add)
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Construct output
    edges = []
    for u in range(N):
        au = best_adj[u]
        for v in range(u + 1, N):
            if (au >> v) & 1:
                edges.append([u, v])
                
    return {'n': N, 'edges': edges}
