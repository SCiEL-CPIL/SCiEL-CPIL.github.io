"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def build_luw_vectorized(q, d):
        """Build LUW(d) graph over GF(q) using vectorized operations. Returns (n, edges)."""
        n = 2 * q ** d
        qd = q ** d
        
        # Define the recurrence rule: rule[c] = (a, b) for c in 2..d
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # Generate all vertices on side 0: x = (x1, ..., xd) in GF(q)^d
        # Use meshgrid to generate all combinations
        axes = [np.arange(q) for _ in range(d)]
        grids = np.meshgrid(*axes, indexing='ij')
        x_arr = np.stack(grids, axis=-1).reshape(-1, d)  # shape: (qd, d)
        
        edges_list = []
        
        # For each vertex on side 0, and for each "first" value, compute neighbor
        # y[1] = first (0..q-1), y[c] = x[c-1] + y[a]*x[b-1] for c=2..d
        # We vectorize over all x and all first values
        
        for first in range(q):
            # y[1] = first (scalar)
            y = np.zeros((qd, d), dtype=np.int64)
            y[:, 0] = first  # y[1] in 1-indexed = y[:, 0] in 0-indexed
            
            for c in range(2, d + 1):
                a, b = rule[c]
                # y[c] = x[c-1] + y[a] * x[b-1]
                # In 0-indexed: y[:, c-1] = x_arr[:, c-1] + y[:, a-1] * x_arr[:, b-1]
                y[:, c - 1] = (x_arr[:, c - 1] + y[:, a - 1] * x_arr[:, b - 1]) % q
            
            # Compute neighbor indices
            # Convert y columns to index: idx = y[:,0]*q^(d-1) + y[:,1]*q^(d-2) + ... + y[:,d-1]
            powers = q ** np.arange(d - 1, -1, -1)
            neighbor_local = (y * powers).sum(axis=1)  # shape: (qd,)
            
            # Source vertex indices (side 0)
            powers_x = q ** np.arange(d - 1, -1, -1)
            src_local = (x_arr * powers_x).sum(axis=1)  # shape: (qd,)
            
            # Full indices: side 0 -> 0 + local, side 1 -> qd + local
            src_full = src_local  # side 0
            dst_full = qd + neighbor_local  # side 1
            
            # Add edges
            for i in range(qd):
                u = int(src_full[i])
                v = int(dst_full[i])
                if u < v:
                    edges_list.append((u, v))
                else:
                    edges_list.append((v, u))
        
        # Remove duplicates
        edges_list = list(set(edges_list))
        
        return n, edges_list

    def verify_girth_fast(n, edges, target_girth):
        """Verify girth >= target_girth using non-backtracking walks with early termination."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build directed edge list
        dir_edges = []
        dir_edge_idx = {}
        for u in range(n):
            for v in adj[u]:
                key = (u, v)
                if key not in dir_edge_idx:
                    dir_edge_idx[key] = len(dir_edges)
                    dir_edges.append((u, v))
        
        num_dir_edges = len(dir_edges)
        
        # Build non-backtracking transition
        nb_trans = [[] for _ in range(num_dir_edges)]
        for (u, v), idx in dir_edge_idx.items():
            for w in adj[v]:
                if w != u:
                    next_key = (v, w)
                    if next_key in dir_edge_idx:
                        nb_trans[idx].append(dir_edge_idx[next_key])
        
        # Check for cycles of length 2 to target_girth-1
        for L in range(2, target_girth):
            # Use iterative propagation
            # State: dict edge_idx -> set of start vertices
            edge_reachable = {}
            for (u, v), idx in dir_edge_idx.items():
                edge_reachable[idx] = {u}
            
            for step in range(L - 1):
                if time.time() > deadline - 5:
                    return False  # Timeout, assume invalid to be safe
                new_reachable = {}
                for e_idx, starts in edge_reachable.items():
                    for f_idx in nb_trans[e_idx]:
                        if f_idx not in new_reachable:
                            new_reachable[f_idx] = set()
                        new_reachable[f_idx].update(starts)
                edge_reachable = new_reachable
            
            # Check closure
            for (u, v), idx in dir_edge_idx.items():
                if idx in edge_reachable and v in edge_reachable[idx]:
                    return False
        
        return True

    # For k=5, g=9:
    # LUW(d) over GF(q) has degree q and girth >= 2d+2
    # We need degree k, so q = k = 5
    # We need girth >= g, so 2d+2 >= g, i.e., d >= (g-2)/2
    # For g=9: d >= 3.5, so d >= 4
    # LUW(4) over GF(5): n = 2*5^4 = 1250
    
    q = k
    min_d = max(2, (g - 2) // 2 + 1)  # d such that 2d+2 >= g
    
    best_result = None
    
    for d in range(min_d, min_d + 3):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw_vectorized(q, d)
            # Verify the graph is k-regular
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(deg[i] != k for i in range(n)):
                continue
            # Verify girth
            if verify_girth_fast(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception as e:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: try larger d
    for d in range(min_d + 3, min_d + 6):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw_vectorized(q, d)
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(deg[i] != k for i in range(n)):
                continue
            if verify_girth_fast(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Last resort
    d = min_d + 2
    n, edges = build_luw_vectorized(q, d)
    return n, edges
# EVOLVE-BLOCK-END
