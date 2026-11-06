"""Maximum Cut solver using Spectral Initialization and Iterated Local Search."""
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Spectral initialization: Use the Fiedler vector (eigenvector of the second smallest eigenvalue
       of the Laplacian) to get a high-quality initial partition.
    2. Fast single-flip local search (hill climbing) to reach a local optimum.
    3. Iterated Local Search (ILS): Perturb the solution by flipping k random vertices,
       then re-run local search. Keep the best solution found.
    
    Returns: 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 145.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array of floats
    adj = np.array(adjacency, dtype=np.float32)
    # Ensure symmetry (just in case)
    adj = (adj + adj.T) / 2.0
    # Zero out diagonal
    np.fill_diagonal(adj, 0.0)
    
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the cut size."""
        l = labels.astype(np.float32)
        # Cut size = 0.5 * sum_i sum_j (l_i - l_j)^2 * A_ij
        # = sum_i l_i * (sum_j A_ij) - sum_i l_i * (sum_j A_ij l_j)
        # = l @ degrees - l @ (adj @ l)
        return float(l @ degrees - l @ (adj @ l))
    
    def local_search(labels):
        """
        Fast single-flip hill climbing.
        For each vertex u, the gain from flipping u is:
        gain[u] = 2 * (neighbors_in_other_set[u] - neighbors_in_same_set[u])
                 = 2 * (neighbors_in_other_set[u] - (degrees[u] - neighbors_in_other_set[u]))
                 = 4 * neighbors_in_other_set[u] - 2 * degrees[u]
        
        We repeatedly flip the vertex with the maximum positive gain until no improvement.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        # Precompute adj @ labels for fast gain calculation
        # neighbors_in_other_set[u] = degrees[u] - (adj @ labels)[u] if labels[u] == 0
        # Actually, let's define:
        # same_set_neighbors[u] = (adj @ labels)[u] if labels[u] == 1 else (adj @ (1-labels))[u]
        # other_set_neighbors[u] = degrees[u] - same_set_neighbors[u]
        # gain[u] = 2 * (other_set_neighbors[u] - same_set_neighbors[u])
        
        # Let's use a simpler formulation:
        # Let s[u] = 1 if labels[u]==1, -1 if labels[u]==0
        # Then cut = 0.25 * sum_i sum_j (1 - s_i s_j) A_ij
        # But flipping s[u] to -s[u] changes cut by:
        # delta = 2 * sum_{v != u} A[u,v] * s[v] * (-s[u]) ... wait, let's derive properly.
        
        # Alternative: gain from flipping u:
        # If u is in set 0, flipping to set 1:
        #   Edges from u to set 0: were NOT cut, become cut (+1 each)
        #   Edges from u to set 1: were cut, become NOT cut (-1 each)
        #   Net gain = (neighbors in set 0) - (neighbors in set 1)
        # If u is in set 1, flipping to set 0:
        #   Edges from u to set 1: were NOT cut, become cut (+1 each)
        #   Edges from u to set 0: were cut, become NOT cut (-1 each)
        #   Net gain = (neighbors in set 1) - (neighbors in set 0)
        
        # Let's compute:
        # in_set_1 = (labels == 1).astype(np.float32)
        # in_set_0 = 1 - in_set_1
        # neighbors_in_set_1[u] = (adj @ in_set_1)[u]
        # neighbors_in_set_0[u] = degrees[u] - neighbors_in_set_1[u]
        # gain[u] = if labels[u]==0: neighbors_in_set_0[u] - neighbors_in_set_1[u]
        #          if labels[u]==1: neighbors_in_set_1[u] - neighbors_in_set_0[u]
        #          = (2*labels[u] - 1) * (neighbors_in_set_1[u] - neighbors_in_set_0[u])
        #          = (2*labels[u] - 1) * (2*neighbors_in_set_1[u] - degrees[u])
        
        while True:
            if time.time() > DEADLINE - 2:
                break
            
            in_set_1 = (labels == 1).astype(np.float32)
            neighbors_in_set_1 = adj @ in_set_1
            # For u in set 0: gain = (deg - n1) - n1 = deg - 2*n1
            # For u in set 1: gain = n1 - (deg - n1) = 2*n1 - deg
            # So gain = (2*labels[u] - 1) * (2*n1[u] - deg[u])
            sgn = 2.0 * labels.astype(np.float32) - 1.0
            gains = sgn * (2.0 * neighbors_in_set_1 - degrees)
            
            max_gain = np.max(gains)
            if max_gain <= 1e-6:
                break
            
            # Flip the vertex with max gain
            idx = np.argmax(gains)
            labels[idx] = 1 - labels[idx]
            current_cut += max_gain
        
        return labels, current_cut
    
    def spectral_init():
        """
        Spectral initialization using the Fiedler vector.
        Compute the Laplacian L = D - A, find the eigenvector corresponding
        to the second smallest eigenvalue, and partition by the median.
        """
        D = np.diag(degrees)
        L = D - adj
        # Use numpy's eigh for symmetric matrices
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector with the second smallest eigenvalue
            fiedler = eigenvectors[:, 1]
            # Partition by median
            median_val = np.median(fiedler)
            labels = (fiedler >= median_val).astype(int)
            return labels
        except:
            # Fallback to random
            return np.random.randint(0, 2, size=n)
    
    def random_init():
        return np.random.randint(0, 2, size=n)
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    # First, try spectral initialization
    labels = spectral_init()
    labels, cut = local_search(labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Iterated Local Search
    num_iter = 0
    max_iter = 5000  # We'll break on time
    
    while time.time() < DEADLINE - 3 and num_iter < max_iter:
        # Perturb the best solution
        k = rng.integers(2, 6)  # Flip 2 to 5 random vertices
        labels = best_labels.copy()
        if n > 0:
            flip_indices = rng.choice(n, size=min(k, n), replace=False)
            labels[flip_indices] = 1 - labels[flip_indices]
        
        # Local search
        labels, cut = local_search(labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        num_iter += 1
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
