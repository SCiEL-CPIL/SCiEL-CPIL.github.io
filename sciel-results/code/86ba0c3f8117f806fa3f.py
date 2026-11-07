import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix)
    n = D.shape[0]
    
    # 1. Build a k-Nearest Neighbor graph to capture local structure
    # For 50 points, k=5 is a good balance.
    k = 5
    # Mask out self-loops
    D_masked = D.copy()
    np.fill_diagonal(D_masked, np.inf)
    
    # Find k nearest neighbors for each node
    # argsort returns indices of the k smallest values
    nn_indices = np.argpartition(D_masked, k, axis=1)[:, :k]
    
    # Create a symmetric adjacency matrix W
    # W[i, j] = 1/d(i,j) if j is in k-NN of i OR i is in k-NN of j
    W = np.zeros((n, n))
    for i in range(n):
        for j in nn_indices[i]:
            # Symmetrize: if i is close to j, mark both
            # We use inverse distance as weight to emphasize close edges
            weight = 1.0 / D[i, j]
            W[i, j] = weight
            W[j, i] = weight
            
    # Ensure no self loops in W
    np.fill_diagonal(W, 0)
    
    # 2. Compute the Laplacian of this weighted graph
    # L = D_deg - W
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    
    # 3. Compute eigenvectors
    # We want the first few non-trivial eigenvectors (Fiedler vectors)
    # eigh returns eigenvalues in ascending order.
    # The first eigenvector is constant (all 1s), corresponding to eigenvalue 0.
    # We take the next 2 eigenvectors to approximate 2D coordinates.
    try:
        eigenvalues, eigenvectors = eigh(L)
        # eigenvectors[:, 0] is the trivial one.
        # Take vectors 1 and 2 (indices 1 and 2)
        # If n is small, ensure we have enough. n=50 is fine.
        emb_vecs = eigenvectors[:, 1:3]
        
        # Normalize the embedding to avoid scaling issues
        # Standard deviation normalization per coordinate
        stds = np.std(emb_vecs, axis=0)
        stds[stds < 1e-9] = 1e-9 # Avoid division by zero
        emb_vecs = emb_vecs / stds
        
    except Exception:
        # Fallback to simple inverse distance if spectral fails
        return 1.0 / (D + 1e-9)
        
    # 4. Compute distances in the embedded space
    # emb_vecs is (n, 2)
    # Calculate pairwise Euclidean distances
    diff = emb_vecs[:, np.newaxis, :] - emb_vecs[np.newaxis, :, :]
    D_emb = np.sqrt(np.sum(diff**2, axis=2))
    
    # 5. Construct Heuristic
    # Use inverse distance in the embedded space
    # Add a small epsilon to avoid division by zero for i==j
    H = 1.0 / (D_emb + 1e-9)
    
    # The diagonal should be low or irrelevant, but let's keep it consistent
    # In ACO, diagonal is usually not used, but we return the full matrix.
    # The problem statement says "Heuristic entries that are not positive become 1e-9".
    # 1/(0+1e-9) is 1e9, which is positive. This is fine.
    # However, typically we don't want to select self. The ACO logic "unvisited node" handles this.
    
    # Ensure all values are finite and positive
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
