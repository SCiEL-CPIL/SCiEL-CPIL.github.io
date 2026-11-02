"""Search for a rank-minimal exact tensor decomposition for 2x3 * 3x4."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
# Deadline guard: evaluator kills after 150s; use a safety margin.
DEADLINE = 135.0

def _tensor_target(n, m, p):
    """Return the target matrix-multiplication tensor T[i, j, k] = 1 iff the (i,k) entry of A@B uses A[i,j]*B[j,k].
    Actually the tensor T_{i,j,k} is 1 for all i,j,k (it's the identity tensor for matmul in this indexing).
    The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    # For matrix multiplication, the tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because (A@B)[i,k] = sum_j A[i,j] B[j,k], the coefficient is always 1.
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build the tensor product check
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # We can vectorize by iterating over k or using einsum-like approach
    # For small sizes, a loop is fine
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.asarray(arr, dtype=float)
    snapped = np.round(arr * 2) / 2.0
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    return arr

def _try_random_decomposition(n, m, p, rank, seed, max_iters=5000):
    """Try to find a rank-`rank` decomposition via random search + local refinement."""
    rng = np.random.RandomState(seed)
    best_error = float('inf')
    best_solution = None
    
    for it in range(max_iters):
        # Random initialization
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Simple gradient-free local improvement: perturb and keep if error decreases
        error = _compute_error(U, V, W, n, m, p)
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
            if error < 1e-9:
                break
        
        # Local refinement: coordinate descent with random perturbations
        for _ in range(20):
            # Perturb one matrix
            mat_idx = rng.randint(0, 3)
            if mat_idx == 0:
                dU = rng.randn(n * m, rank) * 0.1
                new_U = U + dU
                new_err = _compute_error(new_U, V, W, n, m, p)
                if new_err < error:
                    U, error = new_U, new_err
            elif mat_idx == 1:
                dV = rng.randn(m * p, rank) * 0.1
                new_V = V + dV
                new_err = _compute_error(U, new_V, W, n, m, p)
                if new_err < error:
                    V, error = new_V, new_err
            else:
                dW = rng.randn(p * n, rank) * 0.1
                new_W = W + dW
                new_err = _compute_error(U, V, new_W, n, m, p)
                if new_err < error:
                    W, error = new_W, new_err
        
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
            if error < 1e-9:
                break
    
    if best_solution is not None and best_error < 1e-6:
        U, V, W = best_solution
        U = _snap_to_half_int(U)
        V = _snap_to_half_int(V)
        W = _snap_to_half_int(W)
        if _verify_fast(U, V, W, n, m, p, tol=1e-9):
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _compute_error(U, V, W, n, m, p):
    """Compute the max absolute error of the decomposition."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                s = np.dot(u_val, v_val * w_val)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition (rank = n*m*p = 24)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower-rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # For 2x3 * 3x4, the tensor rank is known to be 12 (or possibly less?)
    # Let's search from rank 12 upward
    for rank in range(12, rank_schoolbook):
        if time.time() > DEADLINE - 10:
            break
        # Try several random seeds
        for seed in range(5):
            if time.time() > DEADLINE - 5:
                break
            result = _try_random_decomposition(n, m, p, rank, seed, max_iters=2000)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition at this rank, move to next rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
