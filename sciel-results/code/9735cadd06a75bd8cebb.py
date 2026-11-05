import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    n = 31
    target_edges = 91
    
    def count_c4(adj):
        # adj is a numpy array of shape (n, n) with 0s and 1s
        # Count C4s: sum over u < v of C(common_neighbors, 2)
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = np.bitwise_and(adj[u], adj[v]).sum()
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    # Greedy Construction
    adj = np.zeros((n, n), dtype=np.int8)
    current_c4 = 0
    
    # Precompute pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for _ in range(target_edges):
        best_delta = float('inf')
        best_pair = None
        # Evaluate all non-existing edges
        for u, v in all_pairs:
            if adj[u, v]:
                continue
            
            # Calculate change in C4 count
            # New C4s = C(common(u,v) + 1, 2) - C(common(u,v), 2) + ...
            # Actually, adding (u,v) affects C4s involving u or v.
            # C4s are counted by pairs of vertices.
            # A C4 is formed by (x,y) if they share 2 neighbors.
            # Adding (u,v) increases common neighbors for pairs (u, w) and (v, w) and (u, v)?
            # Let's just recompute or use a faster delta.
            # Delta method:
            # New C4s created involving the new edge (u,v):
            # Path u-a-b-v.
            # This is equivalent to counting paths of length 3 between u and v.
            # In a graph, number of C4s added by edge (u,v) is:
            # sum_{w in N(u)} sum_{z in N(v)} 1 if w!=z? No.
            # A C4 containing (u,v) is u-w-x-v-u.
            # This requires w in N(u), x in N(v), and (w,x) is an edge? No.
            # The cycle is u-w-x-v-u.
            # Edges: (u,w), (w,x), (x,v), (v,u).
            # So we need w in N(u), x in N(v), and (w,x) in E.
            # Wait, the previous formula was sum_{u<v} C(common(u,v), 2).
            # Let's stick to the full recount if n=31 is small enough, or optimize delta.
            # Delta for adding (u,v):
            # The term for pair (u,v) changes from C(common(u,v), 2) to C(common(u,v)+1, 2).
            # common(u,v) is the number of common neighbors BEFORE adding.
            # Also, pairs (u, w) for w in N(v) change?
            # If we add (u,v), then for any w in N(v), u and w now share v as a common neighbor?
            # Yes. So common(u, w) increases by 1 for all w in N(v).
            # And common(v, w) increases by 1 for all w in N(u).
            
            # Let's calculate the delta precisely.
            # Old C4 = sum_{i<j} C(c_{ij}, 2)
            # New C4 = sum_{i<j} C(c'_{ij}, 2)
            # c'_{uv} = c_{uv} + 1
            # c'_{uw} = c_{uw} + 1 for w in N(v)
            # c'_{vw} = c_{vw} + 1 for w in N(u)
            # Other pairs unchanged.
            
            delta = 0
            # Pair (u,v)
            c_uv = np.bitwise_and(adj[u], adj[v]).sum()
            delta += (c_uv + 1) * c_uv // 2 - c_uv * (c_uv - 1) // 2
            # Simplifies to c_uv
            
            # Pairs (u, w) for w in N(v)
            for w in range(n):
                if w == u or w == v: continue
                if adj[v, w]:
                    c_uw = np.bitwise_and(adj[u], adj[w]).sum()
                    delta += (c_uw + 1) * c_uw // 2 - c_uw * (c_uw - 1) // 2
                    # Simplifies to c_uw
            # Pairs (v, w) for w in N(u)
            for w in range(n):
                if w == u or w == v: continue
                if adj[u, w]:
                    c_vw = np.bitwise_and(adj[v], adj[w]).sum()
                    delta += (c_vw + 1) * c_vw // 2 - c_vw * (c_vw - 1) // 2
                    # Simplifies to c_vw
            
            if delta < best_delta:
                best_delta = delta
                best_pair = (u, v)
        
        if best_pair:
            u, v = best_pair
            adj[u, v] = 1
            adj[v, u] = 1
            current_c4 += best_delta

    # Local Search Polish
    best_edges = get_edges(adj)
    best_c4 = current_c4
    T = 10.0
    t_step = 0
    
    while time.time() < deadline - 5.0:
        # Pick random edge to remove
        if not best_edges: break
        idx = random.randint(0, len(best_edges) - 1)
        u1, v1 = best_edges[idx]
        
        # Pick random non-edge to add
        # Find a random pair not in adj
        while True:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 != v2 and not adj[u2, v2]:
                break
        
        # Calculate delta for removing (u1,v1) and adding (u2,v2)
        # This is complex. Simpler: Just recompute?
        # Given n=31, recompute is fast.
        
        # Remove
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        c4_after_remove = count_c4(adj)
        
        # Add
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        c4_after_add = count_c4(adj)
        
        delta = c4_after_add - best_c4
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            best_edges = get_edges(adj)
            best_c4 = c4_after_add
            T *= 0.99
        else:
            # Revert
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            
        t_step += 1
        if t_step % 1000 == 0:
            T = max(0.1, T * 0.95)

    return best_edges
