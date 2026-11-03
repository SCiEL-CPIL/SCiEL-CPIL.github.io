import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    Returns: {'U': list, 'V': list, 'W': list}
    Where U has shape (9, rank), V has shape (21, rank), W has shape (21, rank)
    with entries that are integers or half-integers.
    
    The decomposition satisfies:
    Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T_{i,j,k}
    where T_{i,j,k} = 1 for all valid (i,j,k) and 0 otherwise.
    
    For 3x3x7 matrix multiplication, the schoolbook algorithm uses rank 63.
    We attempt to find a lower rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 63
    
    # Build the target tensor
    # T[ui, vj, wk] = 1 where:
    # ui = i*m + j (9 values)
    # vj = j*p + k (21 values)
    # wk = k*n + i (21 values)
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                target[ui, vj, wk] = 1.0
    
    # Initialize with schoolbook (rank 63)
    best_rank = base_rank
    best_U = np.zeros((n*m, base_rank), dtype=np.float64)
    best_V = np.zeros((m*p, base_rank), dtype=np.float64)
    best_W = np.zeros((p*n, base_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                best_U[ui, col] = 1.0
                best_V[vj, col] = 1.0
                best_W[wk, col] = 1.0
                col += 1
    
    # Helper: verify a decomposition
    def verify_decomp(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the target tensor."""
        R = U.shape[1]
        if V.shape[1] != R or W.shape[1] != R:
            return False
        # Compute T_approx = sum_c U[:,c] * V[:,c] * W[:,c]
        # Use einsum for efficiency
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.max(np.abs(T_approx - target))
        return err < tol
    
    # Helper: snap to half-integers and re-verify
    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        if verify_decomp(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap, True
        return U, V, W, False
    
    # Try to find lower rank using various strategies
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Strategy 1: Random sparse initialization with ALS-like updates
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random sparse binary matrices
            np.random.seed(hash((R, attempt)) % (2**32))
            
            density = 0.15
            U = np.random.choice([-1.0, 1.0], size=(n*m, R)) * (np.random.rand(n*m, R) < density)
            V = np.random.choice([-1.0, 1.0], size=(m*p, R)) * (np.random.rand(m*p, R) < density)
            W = np.random.choice([-1.0, 1.0], size=(p*n, R)) * (np.random.rand(p*n, R) < density)
            
            # Add some structure: ensure each column has at least one non-zero in each matrix
            for c in range(R):
                if np.all(U[:, c] == 0):
                    U[np.random.randint(n*m), c] = 1.0
                if np.all(V[:, c] == 0):
                    V[np.random.randint(m*p), c] = 1.0
                if np.all(W[:, c] == 0):
                    W[np.random.randint(p*n), c] = 1.0
            
            # Iterative refinement: coordinate descent on columns
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # For each column c, optimize U[:,c], V[:,c], W[:,c]
                # This is a rank-1 tensor approximation problem
                # Fixed V and W, optimal U[:,c] = (T - sum_{c'!=c} U_{c'}*V_{c'}*W_{c'}) dot (V_c * W_c) / ||V_c * W_c||^2
                # Similarly for V and W.
                
                # Compute current approximation
                T_current = np.einsum('ic,jc,kc->ijk', U, V, W)
                residual = target - T_current
                
                # Update each coordinate
                for c in range(R):
                    # Optimal U[:,c] given V[:,c] and W[:,c]
                    vw = V[:, c] * W[:, c]  # shape (m*p, p*n)
                    if np.sum(vw**2) > 1e-12:
                        # Project residual onto the subspace spanned by outer products with V_c and W_c
                        # This requires contracting residual with V_c and W_c
                        # U_opt[i] = sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c] / sum_{j,k} V[j,c]^2 * W[k,c]^2
                        num = np.einsum('ijk,j,k->i', residual, V[:, c], W[:, c])
                        den = np.sum(V[:, c]**2) * np.sum(W[:, c]**2)
                        if den > 1e-12:
                            U[:, c] = num / den
                    
                    # Recompute residual
                    T_current = np.einsum('ic,jc,kc->ijk', U, V, W)
                    residual = target - T_current
                    
                    # Optimal V[:,c] given U[:,c] and W[:,c]
                    uw = U[:, c] * W[:, c]
                    if np.sum(uw**2) > 1e-12:
                        num = np.einsum('ijk,i,k->j', residual, U[:, c], W[:, c])
                        den = np.sum(U[:, c]**2) * np.sum(W[:, c]**2)
                        if den > 1e-12:
                            V[:, c] = num / den
                    
                    T_current = np.einsum('ic,jc,kc->ijk', U, V, W)
                    residual = target - T_current
                    
                    # Optimal W[:,c] given U[:,c] and V[:,c]
                    uv = U[:, c] * V[:, c]
                    if np.sum(uv**2) > 1e-12:
                        num = np.einsum('ijk,i,j->k', residual, U[:, c], V[:, c])
                        den = np.sum(U[:, c]**2) * np.sum(V[:, c]**2)
                        if den > 1e-12:
                            W[:, c] = num / den
                
                # Check convergence
                if verify_decomp(U, V, W, tol=1e-8):
                    found = True
                    break
            
            if found:
                U_s, V_s, W_s, success = snap_and_verify(U, V, W)
                if success:
                    best_rank = R
                    best_U = U_s
                    best_V = V_s
                    best_W = W_s
                    break
        
        if found:
            break
    
    # If no better rank found, use the schoolbook decomposition
    def to_list(mat):
        # Round to nearest 0.5
        snapped = np.round(mat * 2) / 2.0
        return snapped.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
