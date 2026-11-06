"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization (Fiedler vector), Simulated Annealing, and Local Search.
    
    1. Spectral Initialization using Fiedler vector (2nd smallest eigenvector of Laplacian).
    2. Simulated Annealing with Fast Delta Updates.
    3. Local Polish.
    4. Time Guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # Helper to compute cut size
    def calc_cut(s):
        total_edges = 0.5 * np.sum(adj)
        val = s @ adj @ s
        return total_edges - 0.5 * val
    
    # 1. Spectral Initialization using Fiedler vector (2nd smallest eigenvector of Laplacian)
    try:
        # Compute Laplacian L = D - A
        degrees = adj.sum(axis=1)
        L = np.diag(degrees) - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # (index 1, since eigenvalues are sorted in ascending order)
        fiedler = eigenvectors[:, 1]
        s = (fiedler >= 0).astype(np.float64) * 2 - 1  # Convert to {-1, 1}
    except Exception:
        s = np.random.choice([-1, 1], size=n).astype(np.float64)
    
    labels = ((s + 1) / 2).astype(np.int8)
    As = adj @ s
    current_cut = calc_cut(s)
    best_cut = current_cut
    best_s = s.copy()
    
    # 2. Simulated Annealing
    # Parameters
    T0 = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    T = T0
    
    # Precompute for speed
    rng = np.random.default_rng()
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Randomly select a vertex to flip
        v = rng.integers(n)
        
        # Calculate delta for flipping v
        delta = 2 * s[v] * As[v]
        
        # Accept if improving, or with probability exp(delta/T) if worsening
        if delta >= 0:
            accept = True
        else:
            accept = rng.random() < np.exp(delta / T)
        
        if accept:
            # Apply flip
            old_s_v = s[v]
            s[v] = -old_s_v
            labels[v] = 1 - labels[v]
            
            # Update As
            As -= 2 * old_s_v * adj[v]
            
            # Update current cut
            current_cut += delta
            
            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_s = s.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Occasionally do a local polish to escape local optima
        if rng.random() < 0.01:
            # Quick local polish: flip any vertex with positive delta
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for v in range(n):
                    delta = 2 * s[v] * As[v]
                    if delta > 0:
                        old_s_v = s[v]
                        s[v] = -old_s_v
                        labels[v] = 1 - labels[v]
                        As -= 2 * old_s_v * adj[v]
                        current_cut += delta
                        improved = True
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_s = s.copy()
    
    # 3. Final local polish: greedily fix all improving moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            delta = 2 * s[v] * As[v]
            if delta > 0:
                old_s_v = s[v]
                s[v] = -old_s_v
                labels[v] = 1 - labels[v]
                As -= 2 * old_s_v * adj[v]
                current_cut += delta
                improved = True
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_s = s.copy()
    
    # Return best labels found
    best_labels = ((best_s + 1) / 2).astype(np.int8)
    return best_labels
# EVOLVE-BLOCK-END
