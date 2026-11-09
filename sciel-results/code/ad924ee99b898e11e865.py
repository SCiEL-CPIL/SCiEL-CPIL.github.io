"""Search for a rank-minimal exact tensor decomposition of 2x5 x 5x5 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

# Deadline guard: the evaluator kills at 150s; keep a safety margin.
_DEADLINE = time.time() + 140.0
_RNG = random.Random(12345)


def _tensor_indices():
    # Returns list of (u, v, w) indices where the multiplication tensor is 1.
    # U index: i*m + j ; V index: j*p + k ; W index: k*n + i  (trace convention)
    idx = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                idx.append((i * M + j, j * P + k, k * N + i))
    return idx


_TENSOR = _tensor_indices()
_TENSOR_SET = set(_TENSOR)


def _verify(U, V, W, tol=1e-6):
    """Check that U @ diag(1) ... reconstructs the tensor exactly."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for (u, v, w) in _TENSOR:
        s = np.dot(U[u, :], V[v, :] * W[w, :])
        if abs(s - 1.0) > tol:
            return False
    # Check all other entries are 0
    for u in range(N * M):
        for v in range(M * P):
            for w in range(P * N):
                if (u, v, w) in _TENSOR_SET:
                    continue
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(s) > tol:
                    return False
    return True


def _reconstruction_error(U, V, W):
    """Compute the total reconstruction error across all tensor entries."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    error = 0.0
    for (u, v, w) in _TENSOR:
        s = np.dot(U[u, :], V[v, :] * W[w, :])
        error += (s - 1.0) ** 2
    for u in range(N * M):
        for v in range(M * P):
            for w in range(P * N):
                if (u, v, w) in _TENSOR_SET:
                    continue
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                error += s ** 2
    return error


def _smooth_loss(params, rank):
    """
    Smooth surrogate loss for tensor decomposition.
    params: flattened array of U, V, W matrices
    Returns: sum of squared errors (smooth, differentiable)
    """
    U_size = N * M * rank
    V_size = M * P * rank
    W_size = P * N * rank
    
    U = params[:U_size].reshape(N * M, rank)
    V = params[U_size:U_size+V_size].reshape(M * P, rank)
    W = params[U_size+V_size:].reshape(P * N, rank)
    
    return _reconstruction_error(U, V, W)


def _smooth_loss_grad(params, rank):
    """
    Gradient of the smooth loss function using finite differences.
    """
    U_size = N * M * rank
    V_size = M * P * rank
    W_size = P * N * rank
    
    U = params[:U_size].reshape(N * M, rank)
    V = params[U_size:U_size+V_size].reshape(M * P, rank)
    W = params[U_size+V_size:].reshape(P * N, rank)
    
    # Compute gradients using automatic differentiation via finite differences
    # For efficiency, use a small step size
    eps = 1e-7
    grad = np.zeros_like(params)
    
    # Sample a subset of indices for gradient computation to save time
    # Use all tensor indices and a random sample of non-tensor indices
    sample_non_tensor = []
    all_non_tensor = [(u, v, w) for u in range(N * M) for v in range(M * P) for w in range(P * N) 
                      if (u, v, w) not in _TENSOR_SET]
    if len(all_non_tensor) > 500:
        sample_non_tensor = _RNG.sample(all_non_tensor, 500)
    else:
        sample_non_tensor = all_non_tensor
    
    indices_to_check = _TENSOR + sample_non_tensor
    
    for idx in range(len(params)):
        orig_val = params[idx]
        params[idx] = orig_val + eps
        loss_plus = _smooth_loss(params, rank)
        params[idx] = orig_val - eps
        loss_minus = _smooth_loss(params, rank)
        params[idx] = orig_val
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    
    return grad


def _optimize_rank(rank, init_U=None, init_V=None, init_W=None, max_iter=2000, tol=1e-10):
    """
    Optimize factor matrices for a given rank using gradient descent.
    """
    U_size = N * M * rank
    V_size = M * P * rank
    W_size = P * N * rank
    total_size = U_size + V_size + W_size
    
    if init_U is not None:
        U = np.array(init_U, dtype=float)
        V = np.array(init_V, dtype=float)
        W = np.array(init_W, dtype=float)
    else:
        # Initialize with small random values
        rng = np.random.RandomState(_RNG.randint(0, 1000000))
        U = rng.randn(N * M, rank) * 0.1
        V = rng.randn(M * P, rank) * 0.1
        W = rng.randn(P * N, rank) * 0.1
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use L-BFGS-B optimizer which can handle larger problems
    result = minimize(
        _smooth_loss,
        params,
        args=(rank,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'disp': False}
    )
    
    optimal_params = result.x
    U_opt = optimal_params[:U_size].reshape(N * M, rank)
    V_opt = optimal_params[U_size:U_size+V_size].reshape(M * P, rank)
    W_opt = optimal_params[U_size+V_size:].reshape(P * N, rank)
    
    return U_opt, V_opt, W_opt, result.fun, result.success


def _try_reduce_rank(current_rank, U, V, W):
    """
    Try to find a valid decomposition at current_rank - 1.
    """
    target_rank = current_rank - 1
    if target_rank < 1:
        return None
    
    # Try multiple random initializations
    best_result = None
    best_error = float('inf')
    
    for attempt in range(3):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Initialize by projecting current solution to lower rank
        # Use SVD to get best rank-k approximation
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        # Create a combined tensor and use HOSVD-like approach
        # For simplicity, use random initialization with scaling
        rng = np.random.RandomState(_RNG.randint(0, 1000000))
        
        # Scale initialization based on current solution norm
        scale = np.linalg.norm(U) / np.sqrt(current_rank)
        
        U_init = rng.randn(N * M, target_rank) * scale
        V_init = rng.randn(M * P, target_rank) * scale
        W_init = rng.randn(P * N, target_rank) * scale
        
        U_opt, V_opt, W_opt, loss, success = _optimize_rank(
            target_rank, U_init, V_init, W_init, max_iter=1500, tol=1e-12
        )
        
        if loss < best_error:
            best_error = loss
            best_result = (U_opt, V_opt, W_opt)
        
        # Check if we found an exact solution
        if _verify(U_opt, V_opt, W_opt, tol=1e-6):
            return (U_opt, V_opt, W_opt)
    
    # If we didn't find exact solution but error is very small, try to polish
    if best_result is not None and best_error < 1e-4:
        U_opt, V_opt, W_opt = best_result
        # Try a final polish with tighter tolerance
        U_polished, V_polished, W_polished, loss2, _ = _optimize_rank(
            target_rank, U_opt, V_opt, W_opt, max_iter=1000, tol=1e-14
        )
        if _verify(U_polished, V_polished, W_polished, tol=1e-6):
            return (U_polished, V_polished, W_polished)
    
    return None


def _build_structured(rank, seed=0):
    """
    Build a structured decomposition using block Strassen-like ideas.
    """
    # Start with schoolbook
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W


def _local_search(U, V, W, rank, max_iters=5000, tol=1e-6):
    """
    Try to reduce rank by removing columns that are linearly dependent or redundant.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = rank

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        # Try to remove a random column
        c = _RNG.randint(0, current_rank - 1)
        U2 = np.delete(U, c, axis=1)
        V2 = np.delete(V, c, axis=1)
        W2 = np.delete(W, c, axis=1)
        if _verify(U2, V2, W2, tol):
            U, V, W = U2, V2, W2
            current_rank -= 1
        else:
            pass

    return U, V, W, current_rank


def construct():
    start = time.time()
    # Start with schoolbook rank = 2*5*5 = 50
    base_rank = N * M * P
    U, V, W = _build_structured(base_rank)

    # Verify base
    assert _verify(U, V, W, 1e-6), "Schoolbook decomposition failed verification"

    best_rank = base_rank
    best = (U, V, W)

    # Try local search to reduce rank
    U2, V2, W2, r2 = _local_search(U, V, W, base_rank, max_iters=10000)
    if r2 < best_rank:
        best_rank = r2
        best = (U2, V2, W2)

    # Try gradient-based rank reduction
    current_U, current_V, current_W = best
    current_rank = best_rank
    
    while current_rank > 1:
        if time.time() > _DEADLINE - 20.0:
            break
        
        result = _try_reduce_rank(current_rank, current_U, current_V, current_W)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank -= 1
            best_rank = current_rank
            best = (current_U, current_V, current_W)
        else:
            # Try a few more times with different random seeds before giving up
            success = False
            for _ in range(2):
                if time.time() > _DEADLINE - 15.0:
                    break
                result = _try_reduce_rank(current_rank, current_U, current_V, current_W)
                if result is not None:
                    current_U, current_V, current_W = result
                    current_rank -= 1
                    best_rank = current_rank
                    best = (current_U, current_V, current_W)
                    success = True
                    break
            if not success:
                break

    U_final, V_final, W_final = best

    # Convert to lists with integer/half-integer entries
    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
