"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def _fiedler_vector(adjacency, n):
    """
    Compute the Fiedler vector (eigenvector of the second-smallest eigenvalue
    of the graph Laplacian). Returns an array of length n with the Fiedler
    values, or None on failure.
    """
    if n < 3:
        return np.zeros(n)
    
    degree = adjacency.sum(axis=1).astype(np.float64)
    A = csr_matrix(adjacency.astype(np.float64))
    D = diags(degree)
    L = D - A
    
    try:
        k = min(2, n - 1)
        if k < 2:
            return np.zeros(n)
        
        # Use 'SA' (smallest algebraic) for better stability with Laplacians
        vals, vecs = eigsh(L, k=k, which='SA', maxiter=n*10)
        
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
            # Remove mean to ensure orthogonality to all-ones
            fiedler = fiedler - fiedler.mean()
            norm = np.linalg.norm(fiedler)
            if norm > 1e-12:
                fiedler = fiedler / norm
            return fiedler
        else:
            return np.zeros(n)
            
    except Exception:
        # Fallback to 'SM' if 'SA' fails
        try:
            k = min(2, n - 1)
            if k < 2:
                return np.zeros(n)
            vals, vecs = eigsh(L, k=k, which='SM', maxiter=n*10)
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            if len(vals) >= 2:
                fiedler = vecs[:, 1]
                fiedler = fiedler - fiedler.mean()
                norm = np.linalg.norm(fiedler)
                if norm > 1e-12:
                    fiedler = fiedler / norm
                return fiedler
        except Exception:
            pass
        return None


def priority(adjacency):
    """
    Compute priority scores for each vertex using the Fiedler vector of the
    graph Laplacian. Vertices with larger absolute Fiedler values are prioritized,
    as they lie at natural graph cuts/periphery and are less entangled in dense cores.
    """
    n = adjacency.shape[0]
    adjacency = np.asarray(adjacency, dtype=bool)
    
    degree = adjacency.sum(axis=1).astype(float)
    base_score = -degree
    
    fiedler = _fiedler_vector(adjacency, n)
    
    if fiedler is None:
        return base_score
    
    if _is_deadline_exceeded():
        return base_score
    
    fiedler_score = np.abs(fiedler)
    
    # Normalize Fiedler to [0, 1]
    f_min, f_max = fiedler_score.min(), fiedler_score.max()
    if f_max - f_min > 1e-12:
        fiedler_norm = (fiedler_score - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    # Normalize degree to [0, 1]
    d_min, d_max = degree.min(), degree.max()
    if d_max - d_min > 1e-12:
        degree_norm = (degree - d_min) / (d_max - d_min)
    else:
        degree_norm = np.zeros(n)
    
    # Use a stronger weight on Fiedler and include a second-order term
    # The square of the normalized Fiedler value emphasizes extreme peripheral vertices
    fiedler_sq = fiedler_norm ** 2
    
    # Combined score: emphasize high Fiedler magnitude (peripheral) and low degree
    # Using a convex combination that rewards both traits
    combined = 0.7 * fiedler_sq + 0.3 * fiedler_norm - 0.15 * degree_norm
    
    return combined
# EVOLVE-BLOCK-END
