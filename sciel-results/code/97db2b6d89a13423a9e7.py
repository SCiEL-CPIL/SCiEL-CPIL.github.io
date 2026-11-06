"""Multi-Start Spectral + SDP-like Initialization with Vectorized SA and 2-Flip Polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses a combination of spectral (top eigenvector) and SDP-inspired (Laplacian
    Fiedler + random hyperplane) initializations, followed by vectorized
    Simulated Annealing with 2-flip moves and a final high-precision polish.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    total_edges = np.sum(adj) / 2.0
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1.0
    
    def compute_cut(labels):
        s = 2.0 * labels - 1.0
        Q = s @ adj @ s
        return (total_edges - Q) / 2.0

    def spectral_init_top(rng):
        """Initialize using the leading eigenvector of the adjacency matrix."""
        v = rng.uniform(-1, 1, n)
        v /= np.linalg.norm(v)
        for _ in range(50):
            v_new = adj @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v_new /= norm
            if np.allclose(v, v_new, atol=1e-10):
                break
            v = v_new
        labels = (v > 0).astype(np.int8)
        return labels

    def sdp_style_init(rng):
        """
        SDP-inspired initialization:
        1. Compute Laplacian L = D - A.
        2. Find Fiedler vector (eigenvector of 2nd smallest eigenvalue).
        3. Use random hyperplane rounding on a matrix formed from the top few eigenvectors
           of the Laplacian (or just the Fiedler vector with random perturbation).
        Here we approximate the SDP relaxation by using the Fiedler vector and
        random hyperplane rounding for diversity.
        """
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
            # (index 1, since index 0 is usually the constant vector for connected graphs)
            fiedler = eigenvectors[:, 1]
            
            # Random hyperplane rounding:
            # Project a random vector onto the Fiedler vector space?
            # Or simply: sign of (fiedler + small noise)?
            # Better: Use the top k eigenvectors to form a matrix X (n x k),
            # then for each random vector u in R^k, label i = 1 if X[i] @ u > 0.
            
            k = min(3, n)
            X = eigenvectors[:, 1:1+k]  # n x k
            
            # Generate random hyperplane
            u = rng.normal(0, 1, k)
            u /= np.linalg.norm(u)
            
            projections = X @ u
            labels = (projections > 0).astype(np.int8)
            
            return labels
        except Exception:
            # Fallback to Fiedler vector only
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            labels = (fiedler > 0).astype(np.int8)
            return labels

    def local_search_sa(labels, time_limit, use_2flip=True):
        """Simulated Annealing with optional 2-flip moves."""
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.005
        
        rng = np.random.default_rng()
        
        while time.time() < time_limit and T > min_T:
            s = 2.0 * current_labels - 1.0
            neighbor_sum = adj @ s
            
            if use_2flip and rng.random() < 0.3:
                # Try a 2-flip move
                v1 = rng.integers(n)
                v2 = rng.integers(n)
                if v1 == v2:
                    continue
                    
                g1 = s[v1] * neighbor_sum[v1]
                g2 = s[v2] * neighbor_sum[v2]
                cross = 2.0 * adj[v1, v2] * s[v1] * s[v2]
                gain = g1 + g2 - cross
                
                if gain >= 0:
                    current_labels[v1] = 1 - current_labels[v1]
                    current_labels[v2] = 1 - current_labels[v2]
                    current_cut += gain
                else:
                    if rng.random() < np.exp(gain / T):
                        current_labels[v1] = 1 - current_labels[v1]
                        current_labels[v2] = 1 - current_labels[v2]
                        current_cut += gain
            else:
                # 1-flip move
                v = rng.integers(n)
                gain = s[v] * neighbor_sum[v]
                
                if gain >= 0:
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                else:
                    if rng.random() < np.exp(gain / T):
                        current_labels[v] = 1 - current_labels[v]
                        current_cut += gain
            
            T *= cooling_rate
            
            if time.time() > time_limit:
                break
        
        return current_labels, current_cut

    def polish(labels, time_limit):
        """Exhaustive 1-flip and 2-flip local search until no improvement."""
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            s = 2.0 * current_labels - 1.0
            neighbor_sum = adj @ s
            
            # Try all 1-flip moves
            for v in range(n):
                gain = s[v] * neighbor_sum[v]
                if gain > 0:
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                    improved = True
                    # Recompute s and neighbor_sum since labels changed
                    s = 2.0 * current_labels - 1.0
                    neighbor_sum = adj @ s
            
            if not improved and time.time() < time_limit:
                # Try 2-flip moves
                s = 2.0 * current_labels - 1.0
                neighbor_sum = adj @ s
                for v1 in range(n):
                    for v2 in range(v1 + 1, n):
                        g1 = s[v1] * neighbor_sum[v1]
                        g2 = s[v2] * neighbor_sum[v2]
                        cross = 2.0 * adj[v1, v2] * s[v1] * s[v2]
                        gain = g1 + g2 - cross
                        if gain > 0:
                            current_labels[v1] = 1 - current_labels[v1]
                            current_labels[v2] = 1 - current_labels[v2]
                            current_cut += gain
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    # Recompute
                    s = 2.0 * current_labels - 1.0
                    neighbor_sum = adj @ s
        
        return current_labels, current_cut

    rng = np.random.default_rng()
    
    # Allocate time: 70% for SA restarts, 30% for final polish
    sa_time = 90.0
    polish_time = 40.0
    
    n_restarts = 80
    time_per_restart = sa_time / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - polish_time:
            break
        
        # Alternate between different initialization strategies
        if i % 3 == 0:
            labels = spectral_init_top(rng)
        elif i % 3 == 1:
            labels = sdp_style_init(rng)
        else:
            # Random initialization with some spectral influence
            labels = spectral_init_top(rng)
            flip_mask = rng.random(n) < 0.2
            labels[flip_mask] = 1 - labels[flip_mask]
        
        # Run SA with 2-flip moves
        labels, cut = local_search_sa(labels, time.time() + time_per_restart, use_2flip=True)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = compute_cut(best_labels)
    
    # Final high-precision polish pass
    remaining = deadline - time.time()
    if remaining > 1.0:
        polished, polished_cut = polish(best_labels, time.time() + min(remaining - 0.5, polish_time))
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
