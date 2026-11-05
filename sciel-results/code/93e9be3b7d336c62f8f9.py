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
        Compute flip gain for every vertex in a vectorized way.
        gain[v] = same_side_neighbors[v] - other_side_neighbors[v]
        """
        # For vertex v with label l_v:
        #   same_side = number of neighbors w with labels[w] == l_v
        #   other_side = number of neighbors w with labels[w] != l_v
        #   gain = same_side - other_side
        
        # adj @ labels gives, for each v, sum of labels of neighbors of v
        # = number of neighbors of v with label 1
        neighbor_labels_sum = A @ labels
        
        # For vertex v with label 0:
        #   same_side = number of neighbors with label 0 = degree[v] - neighbor_labels_sum[v]
        #   other_side = neighbor_labels_sum[v]
        #   gain = (degree[v] - neighbor_labels_sum[v]) - neighbor_labels_sum[v] = degree[v] - 2*neighbor_labels_sum[v]
        
        # For vertex v with label 1:
        #   same_side = neighbor_labels_sum[v]
        #   other_side = degree[v] - neighbor_labels_sum[v]
        #   gain = neighbor_labels_sum[v] - (degree[v] - neighbor_labels_sum[v]) = 2*neighbor_labels_sum[v] - degree[v]
        
        gains = np.where(labels == 0, 
                         degrees - 2 * neighbor_labels_sum,
                         2 * neighbor_labels_sum - degrees)
        return gains
    
    def greedy_polish(labels, max_iters=100):
        """
        Perform vectorized greedy local search by flipping vertices with positive gain.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
                
            gains = vectorized_gains(labels)
            # Find vertices with positive gain
            positive_mask = gains > 1e-9
            if not positive_mask.any():
                break
                
            # Flip all vertices with positive gain simultaneously
            # This is an approximation; for exact greedy, we'd flip one at a time
            # But flipping multiple at once can be faster and still effective
            labels[positive_mask] = 1 - labels[positive_mask]
            current_cut = cut_size(labels)
            improved = True
            
        return labels, current_cut
    
    def simulated_annealing(labels, max_iters=200, initial_temp=1.0, cooling=0.95):
        """
        Perform simulated annealing local search.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            gains = vectorized_gains(labels)
            
            # Choose a random vertex to flip
            v = np.random.randint(0, n)
            delta = gains[v]
            
            # Accept if improvement or with probability exp(-delta/temp) for worsening
            if delta > 0 or (temp > 0 and np.random.random() < np.exp(-delta / temp)):
                labels[v] = 1 - labels[v]
                current_cut += delta
            
            temp *= cooling
            if temp < 1e-6:
                break
        
        return labels, current_cut
    
    def ils(labels, max_iters=50, perturbation_size=5):
        """
        Iterated Local Search: perturb the best solution and re-optimize.
        """
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Perturb: flip a random subset of vertices
            candidate = best_labels.copy()
            num_flips = np.random.randint(1, perturbation_size + 1)
            flip_indices = np.random.choice(n, num_flips, replace=False)
            candidate[flip_indices] = 1 - candidate[flip_indices]
            
            # Polish the perturbed solution
            polished, cut = greedy_polish(candidate)
            
            if cut > best_cut:
                best_labels = polished
                best_cut = cut
            else:
                # With some probability, keep the perturbed version to escape local optima
                if np.random.random() < 0.1:
                    best_labels = candidate
        
        return best_labels, best_cut
    
    # Spectral initialization
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        from scipy.linalg import eigh
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
    
    if labels_spec.sum() == 0 or labels_spec.sum() == n:
        labels_spec = np.random.randint(0, 2, n)
    
    # Run greedy polish from spectral initialization
    best_labels, best_cut = greedy_polish(labels_spec)
    
    # Multi-start with random initializations
    num_restarts = 15
    restart_count = 0
    while restart_count < num_restarts:
        if time.time() > deadline - 15.0:
            break
        
        labels_rand = np.random.randint(0, 2, n)
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
            labels_rand[0] = 0
            labels_rand[1] = 1
        
        # Run greedy polish
        labels_ls, cut_ls = greedy_polish(labels_rand)
        
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls
        
        restart_count += 1
    
    # Apply simulated annealing to the best solution
    if time.time() < deadline - 20.0:
        sa_labels, sa_cut = simulated_annealing(best_labels, max_iters=100, initial_temp=0.5)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Apply ILS to escape deeper local optima
    if time.time() < deadline - 10.0:
        ils_labels, ils_cut = ils(best_labels, max_iters=30, perturbation_size=5)
        if ils_cut > best_cut:
            best_cut = ils_cut
            best_labels = ils_labels
    
    # Final polish
    if time.time() < deadline - 5.0:
        best_labels, best_cut = greedy_polish(best_labels)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
