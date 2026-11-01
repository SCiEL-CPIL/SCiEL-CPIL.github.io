"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.linalg import lstsq, svd
from collections import defaultdict

DIMENSIONS = [2, 5, 6]

DEADLINE_SECONDS = 145.0  # safety margin before 150s hard limit
START_TIME = time.time()


def _remaining():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _check_rank(U, V, W, n, m, p, tol=1e-9):
    """
    Verify exact decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    must equal 1 for all i,j,k.
    """
    r = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each r, the rank-1 tensor is outer(U_col, V_col, W_col)
    # We need sum over r of outer products = all-ones tensor
    total = np.zeros((n, m, p), dtype=np.float64)
    for r_idx in range(r):
        uc = U_arr[:, r_idx].reshape(n, m)
        vc = V_arr[:, r_idx].reshape(m, p)
        wc = W_arr[:, r_idx].reshape(p, n)
        total += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    
    # Check if total is all 1s
    if not np.allclose(total, 1.0, atol=tol):
        return False
    return True


def _build_rank1_matrices(i, j, k, n, m, p):
    """Build U, V, W matrices for a single rank-1 term (i,j,k)."""
    U = [[0] for _ in range(n * m)]
    V = [[0] for _ in range(m * p)]
    W = [[0] for _ in range(p * n)]
    U[i * m + j][0] = 1
    V[j * p + k][0] = 1
    W[k * n + i][0] = 1
    return U, V, W


def _add_term(U, V, W, i, j, k, n, m, p, col_idx, sign=1.0):
    """Add a rank-1 term with sign to existing matrices."""
    U[i * m + j][col_idx] = sign
    V[j * p + k][col_idx] = sign
    W[k * n + i][col_idx] = sign


def _try_lasvegas_random(trial_budget, n, m, p, rank_target):
    """
    Try random initialization with iterative refinement.
    This is a heuristic that may find lower-rank decompositions.
    """
    best_result = None
    best_rank = rank_target + 1
    
    for trial in range(trial_budget):
        if _remaining() < 5.0:
            break
        if time.time() - START_TIME > DEADLINE_SECONDS - 10:
            break
            
        # Random initialization for the target rank
        U = np.random.randn(n * m, rank_target) * 0.1
        V = np.random.randn(m * p, rank_target) * 0.1
        W = np.random.randn(p * n, rank_target) * 0.1
        
        # Iterative refinement: alternate fixing two factors and solving for the third
        # This is a simplified tensor decomposition approach
        for it in range(50):  # limited iterations
            if _remaining() < 3.0:
                break
                
            # Solve for U with V, W fixed
            # For each (i,j), we need sum_r U[i*m+j,r] * (V_col_r * W_col_r) = 1
            # This is a linear system: U_row @ diag(V_col * W_col) = 1
            # But V and W depend on r, so this is coupled.
            
            # Simpler: just do gradient-free random search with occasional checks
            # For this small problem, try random perturbations
            U[np.random.randint(0, n*m)] += np.random.randn() * 0.01
            V[np.random.randint(0, m*p)] += np.random.randn() * 0.01
            W[np.random.randint(0, p*n)] += np.random.randn() * 0.01
        
        # Check if we got close to a solution
        U_int = np.rint(U).astype(int)
        V_int = np.rint(V).astype(int)
        W_int = np.rint(W).astype(int)
        
        if _check_rank(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p, tol=1e-6):
            best_result = {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
            best_rank = rank_target
            break
    
    return best_result, best_rank


def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Start with schoolbook solution as fallback
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition
    # Known bounds: for 2x5x6, the schoolbook is 60.
    # Try to see if we can do better with some combinatorial search.
    
    # Strategy: try to remove some columns by expressing one basis vector as a combination of others
    # This is equivalent to finding if any of the rank-1 tensors are linearly dependent
    # in the tensor space, but since they form a basis, they are independent.
    
    # For small cases, try rank = rank - 1, rank - 2, etc.
    # But first, let's verify the schoolbook works
    if not _check_rank(U, V, W, n, m, p):
        # Fallback: return the schoolbook anyway (it should work)
        pass
    
    # Try a few lower ranks
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for target_rank in range(rank - 1, 39, -1):  # Try down to 40
        if _remaining() < 5.0:
            break
        if time.time() - START_TIME > DEADLINE_SECONDS - 15:
            break
            
        result, found_rank = _try_lasvegas_random(5, n, m, p, target_rank)
        if result is not None:
            best_rank = found_rank
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            break
    
    # Final check
    if not _check_rank(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Return schoolbook if anything went wrong
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
