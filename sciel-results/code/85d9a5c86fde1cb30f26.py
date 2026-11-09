"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solve Maximum Cut using a combination of spectral embedding (Goemans-Williamson style),
    randomized hyperplane projections, and local search with simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # labels is a 0/1 array
        # An edge (i,j) crosses if labels[i] != labels[j]
        # We can compute this as: sum over all edges where labels differ
        # Using matrix operations:
        # Create a matrix where entry (i,j) is 1 if labels[i] != labels[j]
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * diff) // 2)
    
    def local_search(labels, max_rounds=50):
        """
        Greedy local search: flip a vertex if it improves the cut.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        best_cut = cut_size(labels)
        
        # Precompute neighbor lists for efficiency
        # For each vertex, track which neighbors are in same/different partition
        # Use a more efficient approach: maintain gain for each vertex
        
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            # Compute gain for flipping each vertex
            # gain[v] = (number of neighbors in opposite partition) - (number of neighbors in same partition)
            # If gain > 0, flipping improves the cut
            for v in range(n):
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                gain = other - same
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    best_cut += gain
                    improved = True
            
            if not improved:
                break
        
        return labels, best_cut
    
    def simulated_annealing(labels, max_iters=2000):
        """
        Simulated annealing to escape local optima.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        T = 10.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if T < 0.01:
                break
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            new_cut = cut_size(labels)
            delta = new_cut - current_cut
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_cut = new_cut
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = labels.copy()
            else:
                labels[v] = old_label
            
            T *= cooling
        
        return best_labels, best_cut
    
    # Step 1: Spectral embedding using the Laplacian
    # Compute the Laplacian L = D - A
    degree = adj.sum(axis=1).astype(float)
    D = np.diag(degree)
    L = D - adj.astype(float)
    
    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    
    # Use the second eigenvector (Fiedler vector) for initial partition
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    fiedler = eigenvectors[:, 1] if n > 1 else eigenvectors[:, 0]
    
    # Initial partition based on sign of Fiedler vector
    labels_fiedler = (fiedler > 0).astype(int)
    
    # Step 2: Generate multiple initial partitions using random hyperplane projections
    # in the low-dimensional subspace spanned by the first few eigenvectors
    # Use top k eigenvectors (excluding the trivial one)
    k = min(5, n - 1)
    V = eigenvectors[:, 1:k+1]  # Shape: (n, k)
    
    best_labels = None
    best_cut = -1
    
    # Try the Fiedler-based partition
    labels_l, cut_l = local_search(labels_fiedler, max_rounds=100)
    if cut_l > best_cut:
        best_cut = cut_l
        best_labels = labels_l.copy()
    
    # Try random hyperplane projections
    num_projections = 20
    for _ in range(num_projections):
        if time.time() > deadline - 20:
            break
        
        # Random unit vector in R^k
        u = np.random.randn(k)
        u /= np.linalg.norm(u)
        
        # Project vertices onto u
        projections = V @ u
        labels_proj = (projections > 0).astype(int)
        
        # Local search from this starting point
        labels_l, cut_l = local_search(labels_proj, max_rounds=100)
        if cut_l > best_cut:
            best_cut = cut_l
            best_labels = labels_l.copy()
    
    # Step 3: Simulated annealing from the best partition found
    if time.time() < deadline - 15:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=3000)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
    
    # Step 4: Final local search polish
    if time.time() < deadline - 5:
        labels_final, cut_final = local_search(best_labels, max_rounds=200)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
