"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill
_best_weights = None

def _generate_er_graph(n, p, seed):
    """Generate an Erdos-Renyi graph adjacency matrix."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < p
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return mask.astype(np.int8)

def _compute_k_core(adj):
    """Compute k-core numbers for each vertex."""
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.int32)
    k_core = np.zeros(n, dtype=np.int32)
    in_core = np.ones(n, dtype=bool)
    queue = []
    
    # Initialize queue with vertices of degree 0
    for i in range(n):
        if deg[i] == 0:
            queue.append(i)
    
    current_k = 0
    while queue:
        u = queue.pop()
        if not in_core[u]:
            continue
        if deg[u] < current_k:
            in_core[u] = False
            continue
        k_core[u] = current_k
        # Process neighbors
        neighbors = np.where(adj[u] == 1)[0]
        for v in neighbors:
            if in_core[v]:
                deg[v] -= 1
                if deg[v] < current_k:
                    queue.append(v)
        # Check if we can increase k
        # Find minimum degree among remaining core vertices
        remaining = in_core
        if np.any(remaining):
            min_deg = deg[remaining].min()
            if min_deg > current_k:
                current_k = min_deg
                # Re-add vertices that might have been processed
                # Actually, k-core algorithm is more subtle
                # Let's use a simpler iterative approach
    
    # Simpler iterative k-core computation
    k_core = np.zeros(n, dtype=np.int32)
    deg = adj.sum(axis=1).astype(np.int32)
    removed = np.zeros(n, dtype=bool)
    
    for k in range(n):
        # Remove all vertices with degree < k
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not removed[i] and deg[i] < k:
                    removed[i] = True
                    changed = True
                    neighbors = np.where(adj[i] == 1)[0]
                    for v in neighbors:
                        if not removed[v]:
                            deg[v] -= 1
        if not np.any(removed):
            break
        # Assign k-core number
        for i in range(n):
            if not removed[i] and k_core[i] == 0:
                k_core[i] = k
    
    # Vertices never removed have k-core = n (or max)
    # Actually, let's do the standard algorithm properly
    k_core = _k_core_standard(adj)
    return k_core

def _k_core_standard(adj):
    """Standard k-core computation."""
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.int32)
    k_core = np.zeros(n, dtype=np.int32)
    removed = np.zeros(n, dtype=bool)
    
    # Use a bucket-based approach for efficiency
    # But for n=200, simple iterative is fine
    max_k = 0
    remaining = np.ones(n, dtype=bool)
    
    while np.any(remaining):
        # Find minimum degree among remaining
        min_deg = deg[remaining].min() if np.any(remaining) else 0
        
        # Remove all vertices with degree < min_deg + 1? No, remove with degree < current k
        # Actually, k-core: iteratively remove vertices with degree < k
        # Start with k=0, remove degree < 0 (none), then k=1, remove degree < 1, etc.
        
        # Let's just do the standard peeling
        pass
    
    # Simple O(n^2) k-core for n=200
    k_core = np.zeros(n, dtype=np.int32)
    deg = adj.sum(axis=1).astype(np.int32)
    removed = np.zeros(n, dtype=bool)
    
    for k in range(1, n + 1):
        # Remove vertices with degree < k
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not removed[i] and deg[i] < k:
                    removed[i] = True
                    changed = True
                    nbrs = np.where(adj[i] == 1)[0]
                    for v in nbrs:
                        if not removed[v]:
                            deg[v] -= 1
        # Mark vertices that survived this round
        for i in range(n):
            if not removed[i] and k_core[i] == 0:
                k_core[i] = k
    
    return k_core

def _compute_features(adj):
    """Compute feature matrix for a given adjacency matrix."""
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.float64)
    
    # Weighted degree: sum of degrees of neighbors
    wdeg = (adj @ deg)
    
    # K-core numbers
    k_core = _k_core_standard(adj).astype(np.float64)
    
    # Clustering coefficient (local)
    clustering = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbrs = np.where(adj[i] == 1)[0]
        d = len(nbrs)
        if d >= 2:
            sub_adj = adj[np.ix_(nbrs, nbrs)]
            edges = sub_adj.sum() / 2
            clustering[i] = edges / (d * (d - 1) / 2)
    
    # Second order degree
    deg2 = (adj @ wdeg)
    
    # Normalize features to [0, 1] range for better optimization
    features = np.column_stack((deg, wdeg, k_core, clustering, deg2))
    
    # Normalize each feature to [0,1]
    for j in range(features.shape[1]):
        fmin = features[:, j].min()
        fmax = features[:, j].max()
        if fmax > fmin:
            features[:, j] = (features[:, j] - fmin) / (fmax - fmin)
    
    return features

def _greedy_is_size(adj, weights):
    """
    Runs the greedy decoder and returns the size of the independent set.
    weights: array of floats
    """
    n = adj.shape[0]
    features = _compute_features(adj)
    scores = features @ weights
    
    # Sort indices by descending score
    order = np.argsort(-scores, kind='stable')
    
    selected = np.zeros(n, dtype=bool)
    is_size = 0
    
    for u in order:
        if selected[u]:
            continue
        neighbors = np.where(adj[u] == 1)[0]
        if len(neighbors) == 0:
            selected[u] = True
            is_size += 1
        else:
            if np.any(selected[neighbors]):
                continue
            else:
                selected[u] = True
                is_size += 1
    
    return is_size

def _optimize_weights():
    """
    Optimizes the weights for the linear feature combination.
    """
    global _best_weights
    global _start_time
    
    seeds = [1, 2, 3, 4, 5]
    graphs = [_generate_er_graph(200, 0.02, s) for s in seeds]
    
    def objective(weights):
        if time.time() - _start_time > TIME_LIMIT:
            return 1e9
        total_size = 0
        for g in graphs:
            total_size += _greedy_is_size(g, weights)
        return -total_size / len(graphs)
    
    # Initial guess: prioritize low k-core (sparse regions)
    # Features: [deg, wdeg, k_core, clustering, deg2]
    # We want low degree, low k-core, low clustering
    x0 = np.array([-0.5, -0.1, -1.0, -0.2, -0.05])
    
    from scipy.optimize import minimize
    
    best_result = None
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        if time.time() - _start_time > TIME_LIMIT:
            _best_weights = x0
            return
            
        if result.success or result.fun < objective(x0):
            _best_weights = result.x
        else:
            _best_weights = x0
    except Exception:
        _best_weights = x0
    
    # Local refinement
    best_w = _best_weights.copy()
    best_val = objective(best_w)
    
    np.random.seed(42)
    for _ in range(50):
        if time.time() - _start_time > TIME_LIMIT:
            break
        w_try = _best_weights + np.random.normal(0, 0.05, size=len(_best_weights))
        val_try = objective(w_try)
        if val_try < best_val:
            best_val = val_try
            best_w = w_try
    
    _best_weights = best_w

def priority(adjacency):
    """
    Main entry point.
    """
    global _best_weights
    global _start_time
    
    if _best_weights is None:
        _optimize_weights()
    
    # Calculate features for the actual input graph
    features = _compute_features(adjacency.astype(np.int8))
    scores = features @ _best_weights
    
    return scores.astype(float)

# EVOLVE-BLOCK-END
