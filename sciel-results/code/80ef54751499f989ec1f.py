"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Find a high-quality cut in an undirected graph using spectral initialization
    followed by vectorized local search with simulated annealing and ILS.
    
    Parameters:
    -----------
    adjacency : bool matrix (n x n)
        Symmetric adjacency matrix with zeros on diagonal.
        
    Returns:
    --------
    labels : array of 0/1 per vertex
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Convert to float for numerical operations
    A = np.array(adjacency, dtype=np.float64)
    degrees = A.sum(axis=1)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.sum() == 0 or mask1.sum() == 0:
            return 0.0
        return A[mask0][:, mask1].sum()
    
    def vectorized_gains(labels):
        """
        Compute the gain from flipping each vertex.
        For vertex v with label l:
          same_count = number of neighbors with same label
          diff_count = number of neighbors with different label
          gain = same_count - diff_count
        If gain > 0, flipping improves the cut.
        """
        # For each vertex, compute edges to same-label and different-label neighbors
        # Using matrix multiplication for speed
        # edges_to_same[i] = sum_j A[i,j] * (labels[j] == labels[i])
        # edges_to_diff[i] = sum_j A[i,j] * (labels[j] != labels[i])
        # gain[i] = edges_to_same[i] - edges_to_diff[i]
        
        # Efficient computation:
        # Let s = labels (0/1)
        # For vertex i with label s[i]:
        #   edges_to_same[i] = sum_j A[i,j] * (s[j] == s[i])
        #   edges_to_diff[i] = sum_j A[i,j] * (s[j] != s[i])
        #   gain[i] = edges_to_same[i] - edges_to_diff[i]
        #            = sum_j A[i,j] * (2*(s[j]==s[i]) - 1)
        #            = sum_j A[i,j] * (2*s[j] - 1) * (2*s[i] - 1) ... no that's not right
        
        # Let's use a different approach:
        # Define x = 2*labels - 1, so x[i] is +1 or -1
        # An edge (i,j) crosses if x[i] != x[j], i.e., x[i]*x[j] = -1
        # Cut size = sum_{i<j} A[i,j] * (1 - x[i]*x[j]) / 2
        #          = (sum_{i,j} A[i,j] - sum_{i,j} A[i,j]*x[i]*x[j]) / 4
        #          = (2*E - x^T A x) / 4  where E is number of edges
        # 
        # To maximize cut, we want to minimize x^T A x
        # 
        # The gradient of x^T A x w.r.t. x[i] (treating x as continuous) is 2*(A x)[i]
        # But since x[i] is binary, flipping x[i] changes x^T A x by:
        #   delta = -2 * x[i] * (A x)[i]
        # So the gain in cut from flipping vertex i is:
        #   gain[i] = -delta / 4 = x[i] * (A x)[i] / 2
        # Wait, let me re-derive:
        #   cut = (2E - x^T A x) / 4
        #   Flipping x[i] -> -x[i] changes x^T A x by:
        #     new_x^T A new_x = x^T A x - 4*x[i]*(A x)[i]
        #     (because the cross terms involving x[i] change sign)
        #   So delta_cut = (2E - (x^T A x - 4*x[i]*(A x)[i]))/4 - (2E - x^T A x)/4
        #                = 4*x[i]*(A x)[i]/4
        #                = x[i]*(A x)[i]
        # 
        # So gain[i] = x[i] * (A x)[i]
        
        x = 2.0 * labels - 1.0  # +1 or -1
        Ax = A @ x  # (A x)[i] for all i
        gains = x * Ax  # gain from flipping vertex i
        
        return gains
    
    def greedy_polish(labels, max_rounds=20):
        """
        Perform greedy local search: flip all vertices with positive gain,
        repeat until no improvement.
        """
        labels = labels.copy()
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            gains = vectorized_gains(labels)
            # Find vertices with positive gain
            flip_mask = gains > 1e-9
            if not flip_mask.any():
                break
            labels[flip_mask] = 1 - labels[flip_mask]
        return labels
    
    def simulated_annealing(labels, max_iters=2000, T_init=5.0, T_min=0.1, cooling=0.999):
        """
        Simulated annealing with vectorized gain computation.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T = T_init
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Compute gains
            gains = vectorized_gains(labels)
            
            # Randomly pick a vertex to flip
            v = np.random.randint(n)
            gain = gains[v]
            
            # Accept if gain > 0, or with probability exp(gain/T) if gain < 0
            if gain > 0 or np.random.rand() < np.exp(gain / max(T, 1e-9)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= cooling
            if T < T_min:
                break
        
        return best_labels, best_cut
    
    def ils(labels, max_restarts=30):
        """
        Iterated Local Search: perturb the best solution and re-optimize.
        """
        best_labels, best_cut = greedy_polish(labels)
        
        for _ in range(max_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb: flip a random subset of vertices
            num_flips = max(1, n // 10)
            flip_indices = np.random.choice(n, num_flips, replace=False)
            perturbed = best_labels.copy()
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            
            # Re-optimize
            polished = greedy_polish(perturbed)
            cut_val = cut_size(polished)
            
            if cut_val > best_cut:
                best_cut = cut_val
                best_labels = polished
        
        return best_labels, best_cut
    
    # Spectral initialization
    D = np.diag(degrees)
    L = D - A
    
    try:
        from scipy.linalg import eigh
        # Get the Fiedler vector (second smallest eigenvalue)
        vals, vecs = eigh(L, subset_by_index=[1, 1])
        fiedler = vecs[:, 0]
    except Exception:
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(n)
    
    # Initialize labels based on sign of Fiedler vector
    labels_spec = (fiedler > 0).astype(int)
    
    # Ensure we don't have all zeros or all ones
    if labels_spec.sum() == 0 or labels_spec.sum() == n:
        labels_spec = np.random.randint(0, 2, n)
    
    # Run greedy polish from spectral initialization
    best_labels, best_cut = greedy_polish(labels_spec)
    
    # Multi-start with random initializations
    num_restarts = 15
    for _ in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        labels_rand = np.random.randint(0, 2, n)
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
            labels_rand[0] = 0
            labels_rand[1] = 1
        
        polished, cut_val = greedy_polish(labels_rand)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = polished
    
    # ILS phase
    if time.time() < deadline - 25.0:
        ils_labels, ils_cut = ils(best_labels, max_restarts=20)
        if ils_cut > best_cut:
            best_cut = ils_cut
            best_labels = ils_labels
    
    # Final simulated annealing polish
    if time.time() < deadline - 15.0:
        sa_labels, sa_cut = simulated_annealing(best_labels, max_iters=3000, T_init=3.0, T_min=0.05, cooling=0.9995)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final greedy polish
    if time.time() < deadline - 5.0:
        best_labels = greedy_polish(best_labels, max_rounds=30)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
