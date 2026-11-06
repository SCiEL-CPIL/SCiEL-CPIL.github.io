import numpy as np
import time

def priority(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Ensure symmetry
    A = (A + A.T) * 0.5
    
    degree = A.sum(axis=1)
    
    # Build Laplacian L = D - A
    # L = diag(degree) - A
    # We need the second smallest eigenvalue's eigenvector (Fiedler vector)
    
    # Use scipy for sparse or dense eigenvalue computation
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        # Build sparse Laplacian
        D = diags(degree, 0, shape=(n, n), format='csr')
        L = D - csr_matrix(A)
        
        # Find smallest eigenvalues (k=2 to get the Fiedler vector)
        # eigsh finds interior eigenvalues; we want the 2nd smallest
        # Use which='SA' for smallest algebraic
        try:
            eigenvalues, eigenvectors = eigsh(L, k=2, which='SA', maxiter=n*10)
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            fiedler_vec = eigenvectors[:, idx[1]]
        except Exception:
            # Fallback: use dense
            L_dense = np.diag(degree) - A
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            idx = np.argsort(eigenvalues)
            fiedler_vec = eigenvectors[:, idx[1]]
    except ImportError:
        # Dense fallback
        L_dense = np.diag(degree) - A
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        idx = np.argsort(eigenvalues)
        fiedler_vec = eigenvectors[:, idx[1]]
    
    # Fiedler vector magnitude: vertices with large |fiedler| are at the periphery
    # High magnitude = less central to connectivity = safer to pick
    fiedler_mag = np.abs(fiedler_vec)
    
    # Combine with degree: prefer low degree AND high Fiedler magnitude
    # Normalize both to [0, 1] range
    deg_min, deg_max = degree.min(), degree.max()
    if deg_max > deg_min:
        deg_norm = (degree - deg_min) / (deg_max - deg_min)
    else:
        deg_norm = np.zeros(n)
    
    fied_min, fied_max = fiedler_mag.min(), fiedler_mag.max()
    if fied_max > fied_min:
        fied_norm = (fiedler_mag - fied_min) / (fied_max - fied_min)
    else:
        fied_norm = np.zeros(n)
    
    # Score: prefer high Fiedler magnitude (peripheral) and low degree
    score = fied_norm - 0.5 * deg_norm
    
    return score
