"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_c4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    return False
    return True

def _min_degree(n, adj):
    return int(np.min(adj.sum(axis=1))) if n > 0 else 0

def _try_add_edge(adj, i, j, n):
    """Check if adding edge (i,j) keeps graph C4-free. Returns True if safe."""
    if adj[i][j]:
        return True
    common = np.sum(adj[i] & adj[j])
    return common < 2

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s to be safe under 150s limit
    
    # Try to build a C4-free graph with high n and min degree >= n-39
    best_n = 39
    best_edges = []
    
    # Strategy: Start with a known C4-free graph structure and grow it
    # Use polarity graph of projective plane of order q as seed, then extend
    
    # For now, try direct local search for n=46
    target_n = 46
    min_deg_required = target_n - 39  # = 7
    
    # Start with a random sparse graph
    random.seed(42)
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Initialize with random edges
        adj = np.zeros((target_n, target_n), dtype=bool)
        
        # Add random edges while maintaining C4-free
        num_edges_target = int(target_n * min_deg_required / 2) + 10
        edges_added = 0
        max_tries = 10000
        tries = 0
        while edges_added < num_edges_target and tries < max_tries:
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i == j or adj[i][j]:
                continue
            if _try_add_edge(adj, i, j, target_n):
                adj[i][j] = True
                adj[j][i] = True
                edges_added += 1
            tries += 1
        
        # Check min degree
        if _min_degree(target_n, adj) >= min_deg_required:
            # Verify C4-free
            if _is_c4_free(target_n, adj):
                best_n = target_n
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i][j]:
                            edges.append([i, j])
                best_edges = edges
                break
        
        # If min degree not met, try to fix by adding edges to low-degree vertices
        degs = adj.sum(axis=1)
        for _ in range(200):
            if time.time() > deadline:
                break
            low_vertices = [v for v in range(target_n) if degs[v] < min_deg_required]
            if not low_vertices:
                break
            v = random.choice(low_vertices)
            for w in range(target_n):
                if w != v and not adj[v][w]:
                    if _try_add_edge(adj, v, w, target_n):
                        adj[v][w] = True
                        adj[w][v] = True
                        degs[v] += 1
                        degs[w] += 1
                        break
        
        if _min_degree(target_n, adj) >= min_deg_required and _is_c4_free(target_n, adj):
            best_n = target_n
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if adj[i][j]:
                        edges.append([i, j])
            best_edges = edges
            break
    
    # If we didn't get n=46, try smaller n
    if best_n < 46:
        for n_try in range(45, 38, -1):
            if time.time() > deadline:
                break
            min_deg_req = n_try - 39
            if min_deg_req < 0:
                min_deg_req = 0
            
            adj = np.zeros((n_try, n_try), dtype=bool)
            random.seed(n_try)
            np.random.seed(n_try)
            
            for attempt in range(3):
                if time.time() > deadline:
                    break
                edges_added = 0
                max_tries = 5000
                tries = 0
                while tries < max_tries:
                    i = random.randint(0, n_try - 1)
                    j = random.randint(0, n_try - 1)
                    if i == j or adj[i][j]:
                        tries += 1
                        continue
                    if _try_add_edge(adj, i, j, n_try):
                        adj[i][j] = True
                        adj[j][i] = True
                        edges_added += 1
                    tries += 1
                
                if _min_degree(n_try, adj) >= min_deg_req and _is_c4_free(n_try, adj):
                    best_n = n_try
                    edges = []
                    for i in range(n_try):
                        for j in range(i+1, n_try):
                            if adj[i][j]:
                                edges.append([i, j])
                    best_edges = edges
                    break
            
            if best_n == n_try:
                break
    
    if best_n >= 46:
        return {'n': best_n, 'edges': best_edges}
    elif best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
