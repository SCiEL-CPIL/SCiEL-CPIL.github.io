"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    """
    Maximum Cut via Semidefinite Relaxation (SDR) + Local Search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # 1. SDR Phase: Maximize Tr(X * L) where L = D - A
    # X is an n x n matrix. We want X ~ v v^T for unit vectors v.
    # Objective: sum_{i<j} A_ij * (1 - x_i . x_j) / 2 = 1/2 * (sum A_ij - sum A_ij x_i . x_j)
    # Maximizing cut is equivalent to minimizing sum A_ij x_i . x_j.
    # Let L = D - A. Then sum A_ij x_i . x_j = 1/2 * Tr(X L).
    # So we want to minimize Tr(X L) subject to X_ii = 1, X >= 0.
    
    # Compute Laplacian L
    # L = D - A
    # Since A is symmetric, D is diag(sum A)
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Initialize X
    # Random orthonormal-ish start? Or identity?
    # Identity is a valid PSD matrix with unit diagonal.
    # But it corresponds to orthogonal vectors, which is a bad cut (all same side or opposite? No, orthogonal means 0 correlation, which is good for cut? 
    # Wait. Cut size = 1/2 sum A_ij (1 - x_i . x_j).
    # If x_i . x_j = 0 (orthogonal), cut contribution is 0.5 A_ij.
    # If x_i . x_j = -1 (opposite), cut contribution is 1.0 A_ij.
    # If x_i . x_j = 1 (same), cut contribution is 0.
    # We want x_i . x_j to be -1 for edges.
    # So we want to minimize sum A_ij x_i . x_j.
    # Identity matrix X has X_ij = 0 for i!=j. So x_i . x_j = 0.
    # This is a neutral start.
    
    X = np.eye(n)
    
    # Gradient of Tr(X L) w.r.t X is L + L^T = 2L (since L is symmetric).
    # We want to MINIMIZE Tr(X L).
    # So we do Gradient Descent on Tr(X L).
    # Update: X <- X - eta * 2L
    # But we must project back to the constraint set: PSD, unit diagonal.
    
    # Projection onto PSD with unit diagonal is non-trivial.
    # Simple projection: 
    # 1. Symmetrize
    # 2. Eigen-decompose, clip negative eigenvalues to 0 (PSD projection)
    # 3. Normalize rows/columns to have unit diagonal? 
    #    Actually, the constraint is X_ii = 1.
    #    A common heuristic is to project onto PSD, then normalize rows to unit norm.
    #    If X = V V^T, then X_ii = ||v_i||^2. To make it 1, we scale v_i.
    #    So: Eigen-decompose X = U D U^T. 
    #    X_psd = U * np.diag(np.maximum(D, 0)) * U^T.
    #    Then, normalize rows: X_psd[i,j] /= sqrt(X_psd[i,i] * X_psd[j,j])? No, that changes PSD-ness.
    #    Better: Reconstruct from eigenvectors.
    #    Let V be the matrix of eigenvectors scaled by sqrt(eigenvalues).
    #    X = V V^T.
    #    Scale rows of V so that ||v_i|| = 1.
    #    Then X_new = V_scaled V_scaled^T. This has unit diagonal.
    
    def project_psd_unit_diag(M):
        # Symmetrize
        M = (M + M.T) / 2.0
        # Eigen decomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            return np.eye(n)
        
        # Clip negative eigenvalues
        eigvals = np.maximum(eigvals, 0.0)
        
        # Reconstruct V such that M = V V^T
        # V = eigvecs * sqrt(eigvals)
        sqrt_eigvals = np.sqrt(eigvals)
        V = eigvecs * sqrt_eigvals
        
        # Normalize rows of V to unit norm
        norms = np.linalg.norm(V, axis=1)
        norms[norms == 0] = 1.0 # Avoid division by zero
        V = V / norms[:, np.newaxis]
        
        # Reconstruct M
        M_new = V @ V.T
        return M_new

    # Gradient Descent Loop
    # We need to be careful with step size.
    # L has eigenvalues in [0, 2*max_deg].
    # Let's use a fixed step size or adaptive.
    # max_eig of L is approx 2 * max degree.
    max_deg = deg.max()
    if max_deg == 0:
        return np.zeros(n, dtype=int)
    
    step_size = 1.0 / (2.0 * max_deg)
    
    # Run SDR for a few iterations
    # Since n=100, eigendecomposition is fast (O(n^3) ~ 10^6 ops).
    # We can afford ~100-500 iterations.
    for i in range(200):
        if time.time() > deadline:
            break
            
        # Gradient: 2L
        grad = 2.0 * L
        # Update
        X = X - step_size * grad
        # Project
        X = project_psd_unit_diag(X)
        
        # Optional: Early stop if convergence
        # Not strictly necessary for 100 nodes, but good practice.
        
    # 2. Rounding Phase
    # Goemans-Williamson: Random hyperplane.
    # Or, simpler: Use the rows of V (from the last projection) as vectors.
    # We already have V from the last projection step? 
    # Let's re-extract V from the final X to be safe.
    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = np.maximum(eigvals, 0.0)
    V = eigvecs * np.sqrt(eigvals)
    norms = np.linalg.norm(V, axis=1)
    norms[norms == 0] = 1.0
    V = V / norms[:, np.newaxis]
    
    # Random hyperplane rounding
    # Generate random vector r
    # label = 1 if v_i . r > 0 else 0
    
    # To improve quality, we can try multiple random hyperplanes or use the vector itself?
    # Standard GW uses 1 random hyperplane.
    # But we can also try to align with the "dominant" direction?
    # Actually, for Max Cut, the random hyperplane is the theoretical guarantee.
    # However, for small n, we can do better by trying a few or using local search.
    
    # Let's initialize labels using the random hyperplane
    r = np.random.randn(n)
    labels = (V @ r > 0).astype(int)
    
    # 3. Local Search Phase (Polish)
    # Vectorized 1-flip local search.
    # This is crucial to break out of local optima of the rounding.
    
    # We will perform several passes.
    # In each pass, for each vertex, we check if flipping it improves the cut.
    # Gain of flipping v: 
    # Current cut contribution of v: sum_{u in diff} A_vu
    # New cut contribution of v: sum_{u in same} A_vu
    # Change = sum_{u in same} A_vu - sum_{u in diff} A_vu
    # If Change > 0, flip.
    
    # Vectorized implementation:
    # For a given label vector l:
    # same_mask = (l == l[v])
    # gain = adj[v, same_mask].sum() - adj[v, ~same_mask].sum()
    
    # We can vectorize over all vertices simultaneously?
    # Let S be the set of vertices with label 1, T with label 0.
    # For v in S:
    #   gain_v = sum_{u in T} A_vu - sum_{u in S} A_vu
    #   = (A @ (1-l))[v] - (A @ l)[v]  (assuming l is 0/1)
    #   Wait, A @ l gives sum of weights to neighbors in S.
    #   A @ (1-l) gives sum of weights to neighbors in T.
    #   So gain_v = (A @ (1-l))[v] - (A @ l)[v]
    #   If gain_v > 0, flip v (move to T).
    
    # For v in T:
    #   gain_v = sum_{u in S} A_vu - sum_{u in T} A_vu
    #   = (A @ l)[v] - (A @ (1-l))[v]
    #   If gain_v > 0, flip v (move to S).
    
    # Let's define a function to do one pass of local search.
    def local_search_pass(labels, adj):
        # labels: 0/1 array
        # Returns updated labels
        
        # Compute contributions
        # sum_to_1 = adj @ labels
        # sum_to_0 = adj @ (1 - labels)
        
        sum_to_1 = adj @ labels
        sum_to_0 = adj @ (1 - labels)
        
        # For v in 1 (label=1):
        # gain = sum_to_0[v] - sum_to_1[v]
        # If gain > 0, flip to 0.
        
        # For v in 0 (label=0):
        # gain = sum_to_1[v] - sum_to_0[v]
        # If gain > 0, flip to 1.
        
        # We can compute a "desirability" score for flipping each vertex.
        # flip_gain[v] = (sum_to_0[v] - sum_to_1[v]) if label[v]==1 else (sum_to_1[v] - sum_to_0[v])
        
        # Let's use a trick:
        # Let diff = sum_to_0 - sum_to_1
        # If label[v] == 1, we want to flip if diff[v] > 0.
        # If label[v] == 0, we want to flip if -diff[v] > 0 => diff[v] < 0.
        
        # So we want to flip v if sign(diff[v]) != label[v] ? 
        # No.
        # If label=1, flip if diff > 0.
        # If label=0, flip if diff < 0.
        
        # This is equivalent to: flip if diff[v] * (2*label[v] - 1) > 0?
        # If label=1, term is diff. Flip if diff > 0. Correct.
        # If label=0, term is -diff. Flip if -diff > 0 => diff < 0. Correct.
        
        diff = sum_to_0 - sum_to_1
        # sign_factor = 1 if label==1, -1 if label==0
        sign_factor = 2.0 * labels - 1.0
        # Condition: diff * sign_factor > 0
        to_flip = (diff * sign_factor) > 0
        
        # Update labels
        labels[to_flip] = 1 - labels[to_flip]
        
        return labels

    # Run local search until no improvement or time limit
    # We do passes until stable or time runs out.
    # Since n=100, each pass is O(n^2) matrix-vector mult. Very fast.
    # We can run many passes.
    
    prev_cut = -1
    for i in range(100):
        if time.time() > deadline:
            break
            
        labels = local_search_pass(labels, adj)
        
        # Check for convergence
        # Calculate cut size
        # Cut = 0.5 * (sum(A) - sum(A * labels * labels.T)) ? No.
        # Cut = sum_{i<j, l_i != l_j} A_ij
        # = 0.5 * sum_{i,j} A_ij (1 - (2l_i-1)(2l_j-1)) ?
        # Let s_i = 2l_i - 1 (spin).
        # Edge (i,j) is cut if s_i != s_j.
        # Indicator = (1 - s_i s_j) / 2.
        # Cut = 0.5 * sum_{i,j} A_ij (1 - s_i s_j) = 0.5 * (sum(A) - s^T A s)
        
        s = 2.0 * labels - 1.0
        cut_val = 0.5 * (adj.sum() - s @ adj @ s)
        
        if cut_val == prev_cut:
            break
        prev_cut = cut_val
        
    return labels

# EVOLVE-BLOCK-END
