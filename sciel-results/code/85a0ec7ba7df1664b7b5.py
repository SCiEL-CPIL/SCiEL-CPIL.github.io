"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+l, :], V_arr[l*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_54(n, m, p):
    """Build the rank-54 Strassen decomposition."""
    rank = 54
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    return U, V, W, rank

def _try_remove_term(U, V, W, n, m, p, rank, term_idx):
    """Try to express term term_idx as a linear combination of other terms.
    Returns (new_U, new_V, new_W, new_rank) if successful, else None."""
    # We want to find coefficients c_r (for r != term_idx) such that:
    # U[:, term_idx] = sum_{r != term_idx} c_r * U[:, r]
    # V[:, term_idx] = sum_{r != term_idx} c_r * V[:, r]  
    # W[:, term_idx] = sum_{r != term_idx} c_r * W[:, r]
    
    # This is a system: for each row in U, V, W:
    # U[i, term_idx] = sum_{r != term_idx} c_r * U[i, r]
    # V[j, term_idx] = sum_{r != term_idx} c_r * V[j, r]
    # W[k, term_idx] = sum_{r != term_idx} c_r * W[k, r]
    
    # Build the linear system
    other_terms = [r for r in range(rank) if r != term_idx]
    num_equations = n*m + m*p + p*n
    num_unknowns = len(other_terms)
    
    A = np.zeros((num_equations, num_unknowns))
    b = np.zeros(num_equations)
    
    for idx, r in enumerate(other_terms):
        # U part
        A[:n*m, idx] = np.array([U[i][r] for i in range(n*m)])
        # V part
        A[n*m:n*m+m*p, idx] = np.array([V[j][r] for j in range(m*p)])
        # W part
        A[n*m+m*p:, idx] = np.array([W[k][r] for k in range(p*n)])
    
    b[:n*m] = np.array([U[i][term_idx] for i in range(n*m)])
    b[n*m:n*m+m*p] = np.array([V[j][term_idx] for j in range(m*p)])
    b[n*m+m*p:] = np.array([W[k][term_idx] for k in range(p*n)])
    
    # Solve least squares
    try:
        coeffs, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    # Check if the solution is good
    recon = A @ coeffs
    if np.max(np.abs(recon - b)) > 1e-8:
        return None
    
    # Also verify the tensor identity still holds after removal
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n*m)]
    new_V = [[0] * new_rank for _ in range(m*p)]
    new_W = [[0] * new_rank for _ in range(p*n)]
    
    for r in range(rank):
        if r == term_idx:
            continue
        new_idx = r if r < term_idx else r - 1
        for i in range(n*m):
            new_U[i][new_idx] = U[i][r] + coeffs[r - (1 if r > term_idx else 0)] * 0  # Wait, this is wrong
            # Actually we need to add c_r * (term r) to all other terms
            # No wait: we're expressing term_term_idx as sum c_r * term_r
            # So we remove term_term_idx and add c_r to each term_r
            # But that changes the coefficients of the other terms!
            # This approach is flawed. Let me reconsider.
        pass
    
    # The above approach is incorrect. Removing a term and redistributing 
    # changes the other terms. We need a different approach.
    return None

def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """Try to reduce the rank by finding algebraic dependencies."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try each term as a candidate for removal
    for term_idx in range(rank):
        if time.time() > deadline:
            break
        
        # Check if the term can be expressed as a combination of others
        # by solving the linear system
        other_terms = [r for r in range(rank) if r != term_idx]
        num_equations = n*m + m*p + p*n
        num_unknowns = len(other_terms)
        
        A = np.zeros((num_equations, num_unknowns))
        b = np.zeros(num_equations)
        
        for idx, r in enumerate(other_terms):
            A[:n*m, idx] = U_arr[:, r]
            A[n*m:n*m+m*p, idx] = V_arr[:, r]
            A[n*m+m*p:, idx] = W_arr[:, r]
        
        b[:n*m] = U_arr[:, term_idx]
        b[n*m:n*m+m*p] = V_arr[:, term_idx]
        b[n*m+m*p:] = W_arr[:, term_idx]
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        recon = A @ coeffs
        if np.max(np.abs(recon - b)) > 1e-8:
            continue
        
        # If we can express this term, we can remove it
        # New decomposition: remove term_idx, and the other terms remain unchanged
        # because we're just saying term_idx = sum c_r * term_r
        # But wait, that means the tensor identity becomes:
        # sum_{r != term_idx} U_r V_r W_r + (sum c_r U_r) (sum c_r V_r) (sum c_r W_r)
        # This is NOT equal to sum_{r != term_idx} U_r V_r W_r in general!
        # The product of sums is not the sum of products.
        
        # So this approach doesn't work directly. We need a different strategy.
        # Let's try a numerical optimization approach instead.
        break
    
    # Fallback: try numerical optimization for rank reduction
    for target_rank in range(rank - 1, max(40, rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Initialize with the current decomposition, padded/truncated to target_rank
        if target_rank <= rank:
            U_init = U_arr[:, :target_rank].copy()
            V_init = V_arr[:, :target_rank].copy()
            W_init = W_arr[:, :target_rank].copy()
        else:
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            U_init[:, :rank] = U_arr
            V_init[:, :rank] = V_arr
            W_init[:, :rank] = W_arr
        
        # Add small random perturbation
        rng = np.random.default_rng(42 + target_rank)
        U_init = U_init + rng.normal(0, 0.05, size=U_init.shape)
        V_init = V_init + rng.normal(0, 0.05, size=V_init.shape)
        W_init = W_init + rng.normal(0, 0.05, size=W_init.shape)
        
        # Use scipy.optimize.minimize to minimize the tensor residual
        from scipy.optimize import minimize
        
        def objective(x):
            U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_x = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_x = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            res = 0.0
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        s = np.dot(U_x[i*m+l, :], V_x[l*p+k, :] * W_x[k*n+i, :])
                        res += (s - 1.0) ** 2
            return res
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < 1e-8:
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, target_rank):
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_sol.tolist()
                    best_rank = target_rank
                    break
        except Exception:
            continue
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130  # Safety margin
    
    # Build the rank-54 Strassen decomposition
    U54, V54, W54, rank54 = _build_strassen_54(n, m, p)
    
    # Verify
    if not _verify_decomp(U54, V54, W54, n, m, p, rank54):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U54, V54, W54
    best_rank = rank54
    
    # Try to reduce the rank
    if time.time() < _deadline:
        best_U, best_V, best_W, best_rank = _try_rank_reduction(
            U54, V54, W54, n, m, p, rank54, _deadline
        )
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
