"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1.0
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1.0
                V[b*p + c, col] = 1.0
                W[c*n + a, col] = 1.0
                col += 1
    return U, V, W

def _verify_fast(U, V, W, T_target, tol=1e-9):
    """
    Verify tensor decomposition quickly.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    T_target: (nm, mp, pn)
    """
    # Check half-integer constraint
    def is_half_int(mat):
        scaled = mat * 2
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)
    
    if not (is_half_int(U) and is_half_int(V) and is_half_int(W)):
        return False
        
    # Compute tensor
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Use einsum for speed
    T_computed = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
    
    if np.max(np.abs(T_computed - T_target)) > tol:
        return False
        
    return True

def _try_eliminate_column(U, V, W, T_target, col_idx, tol=1e-9):
    """
    Try to eliminate column col_idx.
    Fix U and V (without col_idx), solve for W.
    """
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    # Remove column
    mask = [i for i in range(rank) if i != col_idx]
    U_new = U[:, mask]
    V_new = V[:, mask]
    
    # We want to find W_new (pn, rank-1) such that:
    # sum_{r in mask} U_new[i, r] V_new[j, r] W_new[k, r] = T_target[i, j, k]
    
    # Let's fix U_new and V_new.
    # For each k, we have a linear system?
    # Actually, for fixed U_new, V_new, the equation is linear in W_new.
    # T_target[i, j, k] = sum_r (U_new[i, r] * V_new[j, r]) * W_new[k, r]
    
    # Let's reshape T_target to (pn, nm*mp)
    T_flat = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    # Construct design matrix B (nm*mp, rank-1)
    # B[(i, j), r] = U_new[i, r] * V_new[j, r]
    B = np.zeros((nm * mp, rank - 1))
    for r_idx, r in enumerate(mask):
        B[:, r_idx] = (U_new[:, r_idx] * V_new[:, r_idx]).flatten()
        
    # Solve B^T W_new^T = T_flat^T ?
    # T_flat[k, (i,j)] = sum_r B[(i,j), r] * W_new[k, r]
    # This is a system for each k: T_flat[k, :] = W_new[k, :] @ B
    # So W_new[k, :] is the solution to X @ B = T_flat[k, :]
    # Which is equivalent to B^T X^T = T_flat[k, :]^T
    
    try:
        # We can solve this for all k at once using lstsq on the transposed system
        # But B is (nm*mp, rank-1). T_flat is (pn, nm*mp).
        # We want W_new (pn, rank-1).
        # W_new @ B = T_flat
        # This is a linear system for each row of W_new.
        # Let's use np.linalg.lstsq
        # B.T (rank-1, nm*mp)
        # T_flat.T (nm*mp, pn)
        # We want X (rank-1, pn) such that B.T @ X = T_flat.T
        # X will be W_new.T
        
        B_T = B.T
        T_flat_T = T_flat.T
        
        # lstsq solves A x = b
        # Here A = B_T, b = T_flat_T
        # x will be (rank-1, pn)
        W_new_T, residuals, rank_sol, sv = np.linalg.lstsq(B_T, T_flat_T, rcond=1e-12)
        W_new = W_new_T.T
    except:
        return None
        
    # Quantize to half-integers
    W_new_q = np.round(W_new * 2) / 2.0
    U_new_q = np.round(U_new * 2) / 2.0
    V_new_q = np.round(V_new * 2) / 2.0
    
    # Verify
    if _verify_fast(U_new_q, V_new_q, W_new_q, T_target):
        return U_new_q, V_new_q, W_new_q
        
    return None

def construct():
    deadline = time.time() + 130.0  # 20s safety margin
    n, m, p = DIMENSIONS
    
    # 1. Get Schoolbook Solution (Rank 125)
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    T_target = _build_target(n, m, p)
    
    best_rank = n * m * p
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # 2. Try to reduce rank by 1 using column elimination
    # Iterate through columns, try to remove one
    # If successful, we have rank 124.
    # We can then try to remove another from the new 124-rank solution.
    
    current_U, current_V, current_W = U_sb, V_sb, W_sb
    current_rank = n * m * p
    
    # Attempt iterative elimination
    # Limit iterations to avoid timeout
    max_eliminations = 5 # Try to find rank 120
    
    for _ in range(max_eliminations):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try each column
        for col in range(current_rank):
            if time.time() > deadline - 10:
                break
                
            result = _try_eliminate_column(current_U, current_V, current_W, T_target, col)
            if result is not None:
                U_new, V_new, W_new = result
                current_U, current_V, current_W = U_new, V_new, W_new
                current_rank -= 1
                best_rank = current_rank
                best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                print(f"Reduced rank to {best_rank}")
                found = True
                break # Restart loop with new rank
                
        if not found:
            break # No more columns can be eliminated easily
            
    # 3. If we have time, try random search for further reduction?
    # Given the difficulty, sticking to elimination is safer.
    
    return best_result
# EVOLVE-BLOCK-END
