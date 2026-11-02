"""Program for matrix multiplication 2x3 by 3x4 using numerical optimization."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Uses numerical optimization (L-BFGS-B style via scipy) to search for low-rank
    decompositions. Tries ranks from theoretical lower bound upward.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But we need to check if rank 12 is achievable.
    for rank in range(12, 25):
        if time.time() - start_time > _DEADLINE - 10:
            break
        result = _solve_rank(rank, n, m, p, start_time)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}


def _solve_rank(rank, n, m, p, start_time):
    """
    Try to find a rank-r decomposition using numerical optimization.
    
    We minimize ||sum_c u_c ⊗ v_c ⊗ w_c - T||_F^2 where T[i,j,k]=1 for all valid indices.
    The tensor T is the all-ones tensor of shape (n, m, p).
    
    We parameterize U, V, W as free variables and use scipy.optimize.minimize.
    """
    from scipy.optimize import minimize
    
    # Number of variables: rank * (n*m + m*p + p*n)
    num_um = n * m  # 6
    num_vm = m * p  # 12
    num_wm = p * n  # 8
    total_vars = rank * (num_um + num_vm + num_wm)
    
    # Target tensor: all ones
    target = np.ones((n, m, p))
    
    def _unpack(params):
        """Unpack flat parameter vector into U, V, W."""
        U = params[:rank * num_um].reshape(rank, num_um).T  # (num_um, rank)
        V = params[rank * num_um:rank * (num_um + num_vm)].reshape(rank, num_vm).T
        W = params[rank * (num_um + num_vm):].reshape(rank, num_wm).T
        return U, V, W
    
    def _objective(params):
        """Compute ||tensor - target||_F^2."""
        U, V, W = _unpack(params)
        # Compute the tensor: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # Reshape to (n,m,p)
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            # Outer product: u_col[i,j] * v_col[j,k] * w_col[k,i]
            # We need: for each (i,j,k), add u_col[i,j]*v_col[j,k]*w_col[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        diff = tensor - target
        return np.sum(diff * diff)
    
    def _gradient(params):
        """Compute gradient of objective."""
        U, V, W = _unpack(params)
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        diff = tensor - target
        
        grad = np.zeros_like(params)
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i,j,c] = 2 * diff[i,j,k] * v_col[j,k] * w_col[k,i]
                        grad[c * num_um + i * m + j] += 2 * diff[i, j, k] * v_col[j, k] * w_col[k, i]
                        # d/dV[j,k,c] = 2 * diff[i,j,k] * u_col[i,j] * w_col[k,i]
                        grad[rank * num_um + c * num_vm + j * p + k] += 2 * diff[i, j, k] * u_col[i, j] * w_col[k, i]
                        # d/dW[k,i,c] = 2 * diff[i,j,k] * u_col[i,j] * v_col[j,k]
                        grad[rank * (num_um + num_vm) + c * num_wm + k * n + i] += 2 * diff[i, j, k] * u_col[i, j] * v_col[j, k]
        return grad
    
    # Try multiple random initializations
    np.random.seed(42)
    best_result = None
    best_obj = np.inf
    
    max_starts = 50
    for start_idx in range(max_starts):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Random initialization
        params0 = np.random.randn(total_vars) * 0.5
        
        try:
            res = minimize(
                _objective,
                params0,
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                if res.fun < 1e-10:
                    # Found a valid decomposition
                    U, V, W = _unpack(res.x)
                    # Round to nearest half-integer or integer for exactness
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    
                    # Verify
                    if _verify(U, V, W, n, m, p):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            continue
    
    return None


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True


# EVOLVE-BLOCK-END
