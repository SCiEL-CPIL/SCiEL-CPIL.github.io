"""Matrix multiplication 2,2,10 - Strassen-based decomposition with rank-reduction search."""
import time
import numpy as np
from scipy.optimize import minimize

_DIMS = (2, 2, 10)
_DEADLINE = time.time() + 140


def _verify(U, V, W, n=2, m=2, p=10, tol=1e-9):
    """Verify exact decomposition: T_{i,j,k} = sum_r U[i,r]*V[j,r]*W[k,r]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    # True tensor: T[i,j,k] = 1 if i==k (A row i, B col k, via m) else 0
    # Actually for matmul C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T_{i,j,k} where i indexes A rows, j indexes A cols/B rows, k indexes B cols
    # T[i,j,k] = 1 if j is the shared index, so T[i,j,k] = delta_{i,?} ... 
    # Standard: T_{i,j,k} = 1 if i==k is NOT right.
    # The multiplication tensor M_{i,j,k} = 1 if the (i,j) entry of A multiplies (j,k) entry of B
    # to contribute to (i,k) entry of C. So M[i,j,k] = 1 for all valid i,j,k? No.
    # M[i,j,k] = 1 iff j == j (always true) -- wait, that's not right either.
    # The tensor is: M[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # That means M is the all-ones tensor of shape (n,m,p)? No!
    # M[i,j,k] = 1 if and only if the indices are consistent: A has indices (i,j), B has (j,k).
    # So M[i,j,k] = 1 for all valid i,j,k. That's just an all-ones tensor.
    # Wait, that can't be right. Let me reconsider.
    # The matrix multiplication tensor T is defined by T[i,j,k] = 1 if i==k? No.
    # T[i,j,k] = delta_{i,k}? No, that's for trace.
    # For matrix multiplication C = AB: C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has T[i,j,k] = 1 for all i,j,k (since every triple contributes).
    # Hmm, but then the rank would just be n*m*p = 40 for trivial decomposition.
    # Actually no. The tensor T[i,j,k] = 1 for all i,j,k IS the all-ones tensor.
    # But wait, that's not the standard convention. Let me re-read the task.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is B's column and i is A's row (C's row/col).
    # U is indexed by (i, j) where i is A's row, j is A's col / B's row.
    # V is indexed by (j, k) where j is B's row, k is B's col.
    # The decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j?
    # No: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j]*B[j,k] for each j? No.
    # The tensor equation is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    # Where T[i,j,k] = 1 is the multiplication tensor (every entry is 1).
    # Wait no. Let me think again. The multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # That's correct! Every combination of (A's row i, shared index j, B's column k) contributes.
    # So the tensor is all-ones of shape (2,2,10). Rank of all-ones tensor (2,2,10) is... 
    # The trivial rank is 2*2*10 = 40. Strassen gives 35.
    # So verification: check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k.
    # With indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_strassen_35(n=2, m=2, p=10):
    """Build the rank-35 Strassen block-diagonal decomposition."""
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    sA = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sB = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    wC = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for blk in range(5):
        for r in range(7):
            c = blk * 7 + r
            for i in range(4):
                U[i][c] = sA[r][i]
            k0 = 2 * blk
            k1 = 2 * blk + 1
            V[k0][c] = sB[r][0]
            V[k1][c] = sB[r][1]
            V[p + k0][c] = sB[r][2]
            V[p + k1][c] = sB[r][3]
            W[k0 * n][c] = wC[r][0]
            W[k1 * n][c] = wC[r][1]
            W[k0 * n + 1][c] = wC[r][2]
            W[k1 * n + 1][c] = wC[r][3]

    return U, V, W


def _try_rank_reduction(n=2, m=2, p=10, target_rank=34, time_budget=60.0):
    """Try to find a lower-rank decomposition by perturbing the Strassen solution.
    
    Uses a smooth epigraph formulation: minimize max over (i,j,k) of |sum_r U_i V_j W_k - 1|
    with the constraint that the effective rank is at most target_rank.
    
    Strategy: Start from rank-35 Strassen, try to zero out one column by optimization.
    """
    U0, V0, W0 = _build_strassen_35(n, m, p)
    U0 = np.array(U0, dtype=float)
    V0 = np.array(V0, dtype=float)
    W0 = np.array(W0, dtype=float)
    
    n_params = (n*m + m*p + p*n) * target_rank
    start_time = time.time()
    
    # Try each column to eliminate
    for elim_col in range(35):
        if time.time() - start_time > time_budget:
            break
        
        # Remove column elim_col, keep others
        keep = [c for c in range(35) if c != elim_col]
        U_init = U0[:, keep].flatten()
        V_init = V0[:, keep].flatten()
        W_init = W0[:, keep].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        def objective(x):
            U = x[:n*m*target_rank].reshape(n*m, target_rank)
            V = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            # Compute all tensor entries
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        errors.append(val - 1.0)
            errors = np.array(errors)
            # Smooth max via log-sum-exp
            max_err = np.log(np.sum(np.exp(20 * errors)) / len(errors)) / 20
            # Also add L2 regularization to keep values small
            reg = 0.001 * np.sum(x**2)
            return max_err + reg
        
        def gradient(x):
            U = x[:n*m*target_rank].reshape(n*m, target_rank)
            V = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Compute errors
            errors = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        errors[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                        idx += 1
            
            # Soft-max weights
            weights = np.exp(20 * errors)
            weights /= np.sum(weights)
            
            # Gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w = weights[idx] * 20
                        # d/dU[i*m+j, r] = w * V[j*p+k, r] * W[k*n+i, r]
                        grad_U[i*m+j] += w * V[j*p+k] * W[k*n+i]
                        # d/dV[j*p+k, r] = w * U[i*m+j, r] * W[k*n+i, r]
                        grad_V[j*p+k] += w * U[i*m+j] * W[k*n+i]
                        # d/dW[k*n+i, r] = w * U[i*m+j, r] * V[j*p+k, r]
                        grad_W[k*n+i] += w * U[i*m+j] * V[j*p+k]
                        idx += 1
            
            # Add regularization gradient
            grad_U += 0.002 * U
            grad_V += 0.002 * V
            grad_W += 0.002 * W
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < 1e-6:
            U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Verify
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                # Convert to list of lists
                U_list = [[float(U_opt[i, r]) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V_opt[i, r]) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W_opt[i, r]) for r in range(target_rank)] for i in range(p*n)]
                return U_list, V_list, W_list
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication."""
    n, m, p = _DIMS
    
    # First try rank reduction search with limited time
    time_start = time.time()
    search_budget = min(80.0, _DEADLINE - time_start - 20.0)  # Leave 20s safety margin
    
    if search_budget > 10.0:
        result = _try_rank_reduction(n, m, p, target_rank=34, time_budget=search_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Check if we're close to deadline
    elapsed = time.time() - time_start
    if elapsed > _DEADLINE - time_start - 10.0:
        # Not enough time, return the known rank-35 solution
        U, V, W = _build_strassen_35(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try rank 33 if we have more time
    remaining = _DEADLINE - time.time() - 15.0
    if remaining > 10.0:
        result = _try_rank_reduction(n, m, p, target_rank=33, time_budget=remaining)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-35 Strassen
    U, V, W = _build_strassen_35(n, m, p)
    return {"U": U, "V": V, "W": W}
