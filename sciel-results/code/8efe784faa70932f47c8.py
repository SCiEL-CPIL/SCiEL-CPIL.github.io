"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _count_c4(adj):
    """Count C4s using matrix multiplication."""
    # Number of C4s = sum over all pairs of vertices of C(common_neighbors, 2)
    # = 0.5 * (sum_{u,v} (A^2[u,v])^2 - sum_u A^2[u,u] - sum_{u,v} A^2[u,v])
    # More precisely: each C4 is counted 4 times in sum C(A2[u,v],2) over u!=v
    A2 = adj @ adj
    # For each pair (u,v), u != v, count C(A2[u,v], 2)
    # A2[u,v] for u!=v is the number of common neighbors
    c4_count = 0
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            cn = A2[u, v]
            if cn >= 2:
                c4_count += cn * (cn - 1) // 2
    # Each C4 is counted exactly once per pair of opposite vertices, and each C4 has 2 pairs of opposite vertices
    # Wait: a C4 has 4 vertices. The pairs of opposite vertices are 2 pairs.
    # So each C4 is counted 2 times in the sum over all unordered pairs.
    return c4_count // 2

def _build_polarity_graph(q=5):
    """Build the polarity graph of PG(2,q)."""
    n = q * q + q + 1
    # Label vertices as points of PG(2,q)
    # Points are 1D subspaces of F_q^3
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    # Remove duplicates (there shouldn't be any in this enumeration)
    
    # Polarity: (x,y,z) perp (x',y',z') iff x*x' + y*y' + z*z' + x*z' + y*x' = 0 mod q
    # This is a standard polarity for PG(2,5)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, zi = points[i]
            xj, yj, zj = points[j]
            val = (xi * xj + yi * yj + zi * zj + xi * zj + yi * xj) % q
            if val == 0:
                edges.add((i, j))
    return list(edges), n

def _local_search(edges, n, time_limit=100):
    """Local search to minimize C4 count."""
    start_time = time.time()
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    current_c4 = _count_c4(adj)
    best_edges = list(edges)
    best_c4 = current_c4
    
    edge_set = set(edges)
    
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    # Local search: try swapping edges
    import random
    random.seed(42)
    
    max_iter = 50000
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if no_improve_count > max_no_improve:
            break
        
        # Pick a random edge to remove
        if not best_edges:
            break
        u, v = random.choice(best_edges)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            continue
        # Find a non-edge to add
        # Try random non-edges
        found = False
        for _ in range(20):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) in edge_set or (b, a) in edge_set:
                continue
            if (a, b) == (u, v) or (a, b) == (v, u):
                continue
            
            # Evaluate the swap
            # Remove edge (u,v), add edge (a,b)
            # Delta in C4 count
            # C4s involving edge (u,v): for each w in N(u)∩N(v), ... 
            # Actually, let's just compute the change directly
            
            # Common neighbors of u and v
            cn_uv = np.sum(adj[u] & adj[v])
            # Common neighbors of a and b  
            cn_ab = np.sum(adj[a] & adj[b])
            
            # When we remove (u,v):
            # - We lose C4s that use edge (u,v): these are C4s where u-v is an edge
            #   A C4 containing edge (u,v) has form u-v-x-w-u where x in N(v)\{u}, w in N(u)\{v}, x~w
            #   Actually, it's easier to think: removing (u,v) removes all C4s that contain this edge
            #   The number of C4s containing edge (u,v) is sum over pairs (x,y) where x~v, y~u, x~y, x!=u, y!=v
            #   This is complex. Let's use a different approach.
            
            # Alternative: compute C4 count before and after using the formula
            # C4 count = 1/2 * sum_{u<v} C(common_neighbors(u,v), 2)
            
            # After removing (u,v) and adding (a,b):
            # For pairs involving u or v or a or b, common neighbors change
            
            # Let's just do a full recount if the delta looks promising
            # First, estimate the change
            
            # Removing (u,v) affects common neighbor counts for pairs that include u or v
            # Adding (a,b) affects common neighbor counts for pairs that include a or b
            
            # This is getting complex. Let's use a simpler heuristic:
            # Prefer swaps where cn_uv is high (removing a high-common-neighbor edge)
            # and cn_ab is low (adding a low-common-neighbor edge)
            
            delta_est = cn_ab - cn_uv
            if delta_est < 0:
                # Try this swap
                # Remove (u,v)
                edge_set.discard((u, v))
                edge_set.discard((v, u))
                adj[u, v] = 0
                adj[v, u] = 0
                deg[u] -= 1
                deg[v] -= 1
                
                # Add (a,b)
                edge_set.add((a, b))
                adj[a, b] = 1
                adj[b, a] = 1
                deg[a] += 1
                deg[b] += 1
                
                new_c4 = _count_c4(adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edges = list(edge_set)
                    no_improve_count = 0
                    found = True
                else:
                    # Revert
                    edge_set.discard((a, b))
                    adj[a, b] = 0
                    adj[b, a] = 0
                    deg[a] -= 1
                    deg[b] -= 1
                    
                    edge_set.add((u, v))
                    adj[u, v] = 1
                    adj[v, u] = 1
                    deg[u] += 1
                    deg[v] += 1
                    no_improve_count += 1
                break
        
        if not found:
            no_improve_count += 1
    
    return best_edges, best_c4

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    # Build polarity graph
    edges, n = _build_polarity_graph(5)
    
    if len(edges) > 91:
        # Remove random edges to get to 91
        import random
        random.seed(42)
        random.shuffle(edges)
        edges = edges[:91]
    elif len(edges) < 91:
        # Add random non-edges
        import random
        random.seed(42)
        edge_set = set(edges)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        non_edges = [p for p in all_pairs if p not in edge_set]
        random.shuffle(non_edges)
        edges.extend(non_edges[:91 - len(edges)])
    
    # Local search
    edges, c4 = _local_search(edges, n, time_limit=deadline - (time.time() - start_time) - 5)
    
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
