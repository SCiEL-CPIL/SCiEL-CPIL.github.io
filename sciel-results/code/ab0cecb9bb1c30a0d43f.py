import numpy as np
from scipy.linalg import eigh

N = 13

def construct():
    # Spectral Graph Theory Approach
    # We model the points as nodes in a cycle graph C_13.
    # The positions are derived from the first non-trivial eigenvectors of the graph Laplacian.
    # This structurally enforces the points to lie on a smooth convex manifold (an ellipse/circle).
    
    # 1. Construct the Laplacian matrix of the cycle graph.
    # The Laplacian L is symmetric and positive semi-definite.
    # For a cycle graph with uniform edge weights, L is a circulant matrix.
    # We add a small diagonal term to make it strictly positive definite (SPD) as per the spectral method requirements.
    
    L = np.zeros((N, N))
    
    # Set edge weights (uniform for regular polygon)
    # In a learnable context, these would be parameters.
    w = 1.0 
    
    for i in range(N):
        # Diagonal: sum of incident weights
        L[i, i] = 2 * w
        
        # Off-diagonal: negative weights to neighbors
        L[i, (i + 1) % N] = -w
        L[i, (i - 1) % N] = -w
        
    # Regularization to ensure SPD (eigenvalues > 0)
    L += 1e-6 * np.eye(N)
    
    # 2. Compute eigendecomposition
    # eigh is efficient for symmetric/Hermitian matrices
    eigenvalues, eigenvectors = eigh(L)
    
    # 3. Extract the first non-trivial eigenvectors
    # The first eigenvector (index 0) is the constant vector (trivial mode).
    # The next two eigenvectors (indices 1 and 2) correspond to the first harmonic (k=1).
    # These span the 2D subspace that determines the shape of the convex hull.
    
    v1 = eigenvectors[:, 1]
    v2 = eigenvectors[:, 2]
    
    # 4. Construct 2D points
    # Each point i is (v1[i], v2[i])
    points = np.column_stack((v1, v2))
    
    return points
