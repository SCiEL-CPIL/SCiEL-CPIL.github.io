import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _run_aco_fast(D, eta, n_ants=20, n_iter=30, seed=42):
    """Run ACO with given heuristic eta, return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for it in range(n_iter):
        all_paths = []
        all_lens = []
        
        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0
            
            for step in range(1, n):
                available = np.where(~visited)[0]
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()
                
                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]
                
                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)
            
            tour_len += D[current, 0]
            path.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            all_paths.append(path)
            all_lens.append(tour_len)
        
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = all_lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = all_paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k+1]
                tau[i, j] += reward
    
    return best_len

def _make_eta_topological(D, alpha=1.0, beta=1.0):
    """
    Create heuristic based on inverse distance and Fiedler vector penalty.
    eta[i,j] = (1/D[i,j])^alpha * exp(-beta * (fiedler[i] - fiedler[j])^2)
    """
    n = D.shape[0]
    # Base geometric heuristic: inverse distance
    # Avoid division by zero
    D_safe = np.where(D > 0, D, 1.0)
    eta_base = 1.0 / D_safe
    
    # Compute Fiedler vector
    # Graph Laplacian L = D - A, where A is adjacency (inverse distance or similar)
    # We use inverse distance as weights for the graph structure
    W = 1.0 / D_safe
    np.fill_diagonal(W, 0)
    
    # Degree matrix
    deg = np.sum(W, axis=1)
    D_mat = np.diag(deg)
    
    # Laplacian
    L = D_mat - W
    
    # Eigendecomposition
    # L is symmetric, so we can use eigh
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigenvalues are sorted in ascending order
        fiedler = eigenvectors[:, 1]
    except Exception:
        # Fallback if eigendecomposition fails
        fiedler = np.zeros(n)
    
    # Calculate penalty based on difference in Fiedler coordinates
    # Nodes far apart in the Fiedler space are likely to be on opposite sides of the "bottleneck"
    # or represent a "jump" that might cause crossings or long detours if connected directly
    # We want to penalize edges that connect nodes with large differences in their Fiedler coordinates
    # if that difference is large relative to the distance? 
    # Actually, the directive says: "multiplicatively dampen heuristic entries for edges that connect nodes with large differences in their Fiedler coordinates"
    
    fiedler_diff_sq = (fiedler[:, np.newaxis] - fiedler[np.newaxis, :]) ** 2
    
    # Scale the penalty. The magnitude of fiedler vector depends on normalization.
    # Typically fiedler vector is unit norm. 
    # We need a scaling factor to make the exponent meaningful.
    # Let's normalize the fiedler difference to be in a comparable range.
    # Max possible diff is sqrt(2) if unit norm? No, max diff between two components can be up to sqrt(2) if concentrated.
    # Let's just use a heuristic scale.
    # If beta is too large, we kill all long edges.
    # Let's try to normalize fiedler to [-1, 1] range roughly?
    # Or just use raw values with a tuned beta.
    # Given n=50, let's assume fiedler values are O(1/sqrt(n)).
    # So differences are O(1/sqrt(n)).
    # Let's scale by n to make the exponent O(1).
    scale = float(n)
    penalty = np.exp(-beta * scale * fiedler_diff_sq)
    
    eta = eta_base * penalty
    
    # Set diagonal to 1e-9
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure positive
    eta = np.maximum(eta, 1e-9)
    
    return eta

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    
    if time.time() > _DEADLINE:
        # Fallback: simple inverse distance
        D_safe = np.where(D > 0, D, 1.0)
        eta = 1.0 / D_safe
        np.fill_diagonal(eta, 1e-9)
        return eta
    
    # We will search for the best beta (penalty strength)
    # alpha is fixed at 1.0 for now, or we can search both.
    # Let's fix alpha=1.0 and search beta.
    
    best_beta = 1.0
    best_score = float('inf')
    
    # Candidate betas
    # beta controls how strongly we penalize Fiedler-space distance
    # Small beta -> close to pure geometric
    # Large beta -> strong topological penalty
    candidates = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    search_deadline = time.time() + 100.0
    
    for beta in candidates:
        if time.time() > search_deadline:
            break
        try:
            eta = _make_eta_topological(D, alpha=1.0, beta=beta)
            score = _run_aco_fast(D, eta, seed=42)
            if score < best_score:
                best_score = score
                best_beta = beta
        except Exception:
            continue
    
    # Refine around best_beta
    if time.time() < search_deadline - 10:
        step = 0.1
        current_beta = best_beta
        for _ in range(20):
            if time.time() > search_deadline:
                break
            improved = False
            for delta in [-step, step]:
                new_beta = current_beta + delta
                if new_beta < 0:
                    continue
                try:
                    eta = _make_eta_topological(D, alpha=1.0, beta=new_beta)
                    score = _run_aco_fast(D, eta, seed=42)
                    if score < best_score:
                        best_score = score
                        best_beta = new_beta
                        current_beta = new_beta
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                step *= 0.7
            if step < 0.001:
                break
    
    return _make_eta_topological(D, alpha=1.0, beta=best_beta)
