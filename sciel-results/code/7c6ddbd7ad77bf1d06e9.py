import time
import numpy as np
import random

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix."""
    n = adj.shape[0]
    common = adj @ adj
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            k = common[u, v]
            if k >= 2:
                total += k * (k - 1) // 2
    return total

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5).
    Returns adj matrix (31x31) and list of edges.
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31
    
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    return edges, adj

def _c4s_created_by_add(adj, u, v):
    """Count C4s created by adding edge (u,v) to graph with adjacency adj."""
    # A C4 containing (u,v) has form u-a-b-v-u
    # where a is neighbor of u, b is neighbor of v, a != b, and (a,b) is an edge
    ni = np.where(adj[u] == 1)[0]
    nj = np.where(adj[v] == 1)[0]
    if len(ni) == 0 or len(nj) == 0:
        return 0
    # Count edges between N(u) and N(v)
    sub = adj[np.ix_(ni, nj)]
    return int(np.sum(sub))

def _c4s_removed_by_del(adj, u, v):
    """Count C4s destroyed by removing edge (u,v) from graph with adjacency adj.
    adj must have (u,v) present."""
    # Same formula: number of edges between N(u)\{v} and N(v)\{u}
    ni = np.where(adj[u] == 1)[0]
    nj = np.where(adj[v] == 1)[0]
    # Remove v from ni and u from nj
    ni_mask = ni != v
    nj_mask = nj != u
    ni = ni[ni_mask]
    nj = nj[nj_mask]
    if len(ni) == 0 or len(nj) == 0:
        return 0
    sub = adj[np.ix_(ni, nj)]
    return int(np.sum(sub))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # 1. Build base graph (90 edges, C4-free)
    edges, adj = _build_polarity_graph()
    n = 31
    
    # 2. Find best single edge to add (minimizes C4s created)
    best_edge = None
    best_add_c4 = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                c4s = _c4s_created_by_add(adj, i, j)
                if c4s < best_add_c4:
                    best_add_c4 = c4s
                    best_edge = [i, j]
    
    if best_edge:
        adj[best_edge[0], best_edge[1]] = 1
        adj[best_edge[1], best_edge[0]] = 1
        edges.append(best_edge)
    
    current_c4 = best_add_c4
    
    # Precompute neighbor lists for fast access
    def get_neighbors(u):
        return np.where(adj[u] == 1)[0]
    
    # 3. Local search with simulated annealing
    # 1-swap moves: remove one edge, add another
    # Use incremental C4 counting
    
    random.seed(42)
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    n_iter = 3000
    
    current_adj = adj.copy()
    current_edges = edges[:]
    best_adj = adj.copy()
    best_edges = edges[:]
    best_c4 = current_c4
    
    for it in range(n_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Temperature
        progress = it / max(n_iter - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Choose a random edge to remove
        edge_idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[edge_idx]
        
        # Compute C4s removed by deleting (u,v)
        c4s_rem = _c4s_removed_by_del(current_adj, u, v)
        
        # Remove the edge
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_edges.pop(edge_idx)
        
        # Choose a random non-edge to add
        # Sample from non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if current_adj[i, j] == 0:
                    non_edges.append((i, j))
        
        if not non_edges:
            # Restore
            current_adj[u, v] = 1
            current_adj[v, u] = 1
            current_edges.append([u, v])
            continue
        
        x, y = random.choice(non_edges)
        
        # Compute C4s created by adding (x,y)
        c4s_add = _c4s_created_by_add(current_adj, x, y)
        
        # Add the edge
        current_adj[x, y] = 1
        current_adj[y, x] = 1
        current_edges.append([x, y])
        
        # New C4 count
        new_c4 = current_c4 - c4s_rem + c4s_add
        
        # Accept or reject
        if new_c4 <= current_c4 or random.random() < np.exp(-(new_c4 - current_c4) / T):
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = current_adj.copy()
                best_edges = current_edges[:]
        else:
            # Revert
            current_adj[x, y] = 0
            current_adj[y, x] = 0
            current_edges.pop()
            current_adj[u, v] = 1
            current_adj[v, u] = 1
            current_edges.append([u, v])
    
    # 4. Final greedy polish: try all 1-swaps to improve
    polish_adj = best_adj.copy()
    polish_edges = best_edges[:]
    polish_c4 = best_c4
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if polish_adj[i,j] == 1]
        
        for e1 in edge_list:
            if time.time() > deadline - 3.0:
                break
            u, v = e1
            c4s_rem = _c4s_removed_by_del(polish_adj, u, v)
            
            polish_adj[u, v] = 0
            polish_adj[v, u] = 0
            idx = polish_edges.index([u, v]) if [u, v] in polish_edges else polish_edges.index([v, u])
            polish_edges.pop(idx)
            
            best_add = None
            best_new_c4 = polish_c4
            
            for i in range(n):
                for j in range(i + 1, n):
                    if polish_adj[i, j] == 0 and (i,j) != (u,v):
                        c4s_add = _c4s_created_by_add(polish_adj, i, j)
                        new_c4 = polish_c4 - c4s_rem + c4s_add
                        if new_c4 < best_new_c4:
                            best_new_c4 = new_c4
                            best_add = [i, j]
            
            if best_add is not None and best_new_c4 < polish_c4:
                polish_adj[best_add[0], best_add[1]] = 1
                polish_adj[best_add[1], best_add[0]] = 1
                polish_edges.append(best_add)
                polish_c4 = best_new_c4
                improved = True
                # Restart since graph changed
                break
            else:
                # Revert
                polish_adj[u, v] = 1
                polish_adj[v, u] = 1
                polish_edges.append([u, v])
    
    # Verify edge count
    assert len(polish_edges) == 91, f"Expected 91 edges, got {len(polish_edges)}"
    assert len(set(tuple(sorted(e)) for e in polish_edges)) == 91, "Duplicate edges"
    
    return polish_edges
