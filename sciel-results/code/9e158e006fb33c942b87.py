"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: 150s kill, use 145s as safety margin
    deadline = time.time() + 145.0
    
    # Convert to int for faster operations
    adj_int = adj.copy()
    
    def cut_size(labels):
        # Count edges crossing the partition
        # labels is a 0/1 array
        return int(adj_int[labels == 0, labels == 1].sum())
    
    def local_search(labels, max_iterations=1000):
        """Greedy local search: flip vertex if it increases cut."""
        labels = labels.copy()
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            for v in range(n):
                # Compute gain from flipping v
                # Current cut contribution from v: edges to opposite side
                # After flip: edges to same side become cut edges
                same_side = (labels == labels[v]).astype(np.int8)
                # Number of neighbors on same side (currently NOT cut)
                same_neighbors = int(adj_int[v, same_side].sum())
                # Number of neighbors on opposite side (currently cut)
                diff_neighbors = int(adj_int[v, 1 - same_side].sum())
                # If we flip, we gain same_neighbors - diff_neighbors
                gain = same_neighbors - diff_neighbors
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    # Spectral initialization using the Fiedler vector
    deg = adj_int.sum(axis=1)
    L = np.diag(deg) - adj_int
    
    # Compute eigenvectors of the Laplacian
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the second smallest eigenvector
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
    except Exception:
        fiedler = np.random.randn(n)
    
    # Generate multiple random hyperplane projections from the spectral embedding
    # Use the top few eigenvectors for a richer embedding
    try:
        # Use eigenvectors 1 through min(10, n-1) for embedding
        k = min(10, n - 1)
        embedding = eigvecs[:, idx[1:k+1]]
    except Exception:
        embedding = np.random.randn(n, 5)
    
    best_labels = None
    best_cut = -1
    
    # Try multiple random hyperplane cuts on the embedding
    num_hyperplanes = 50
    for i in range(num_hyperplanes):
        if time.time() > deadline - 10:
            break
        
        # Random hyperplane
        if embedding.shape[1] > 0:
            random_vec = np.random.randn(embedding.shape[1])
            projections = embedding @ random_vec
            # Sign-based partition
            labels = (projections > 0).astype(np.int8)
        else:
            labels = np.random.randint(0, 2, n)
        
        # Local search to refine
        refined = local_search(labels, max_iterations=500)
        c = cut_size(refined)
        
        if c > best_cut:
            best_cut = c
            best_labels = refined.copy()
    
    # Also try alternating start and random starts
    if time.time() < deadline - 5:
        # Alternating
        labels_alt = np.arange(n) % 2
        refined_alt = local_search(labels_alt, max_iterations=500)
        c_alt = cut_size(refined_alt)
        if c_alt > best_cut:
            best_cut = c_alt
            best_labels = refined_alt.copy()
    
    # Try a few random starts
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        labels_rand = np.random.randint(0, 2, n)
        refined_rand = local_search(labels_rand, max_iterations=300)
        c_rand = cut_size(refined_rand)
        if c_rand > best_cut:
            best_cut = c_rand
            best_labels = refined_rand.copy()
    
    # Final local search with more iterations on the best solution
    if best_labels is not None and time.time() < deadline - 5:
        final = local_search(best_labels, max_iterations=1000)
        best_labels = final
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
