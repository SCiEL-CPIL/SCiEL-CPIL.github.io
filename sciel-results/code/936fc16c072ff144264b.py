"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard: kill at 130s to stay within 150s wall-clock limit
DEADLINE = time.time() + 130.0

def _time_left():
    return DEADLINE - time.time()

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k).
    Actually, the tensor for matrix multiplication is:
    T[i, j, k] = 1 if and only if the (i,k) entry of C = A*B involves A[i,j]*B[j,k].
    The standard formulation: T is a 3-tensor of size n x m x p where
    T[i,j,k] = 1 for all i,j,k (since each product C[i,k] = sum_j A[i,j] B[j,k]).
    
    Wait, let me reconsider. The matrix multiplication tensor has entries:
    T[i, j, k] = 1 for all i in [n], j in [m], k in [p].
    The tensor contraction: C[i,k] = sum_j A[i,j] * B[j,k]
    This corresponds to a rank-1 decomposition where each term is:
    u[i] * v[j] * w[k] with appropriate indexing.
    
    The tensor T of shape (n, m, p) has T[i,j,k] = 1 for all i,j,k.
    We want to decompose T as sum_r u_r[i] * v_r[j] * w_r[k].
    
    The schoolbook decomposition has rank n*m*p where each term picks one (i,j,k).
    """
    # T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    correctly decompose the matrix multiplication tensor.
    
    The convention: W[k*n + i, r] for the output factor.
    U[i*m + j, r] for input A factor.
    V[j*p + k, r] for input B factor.
    
    The tensor element T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Reshape to 3D
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # Compute tensor: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # W3 is indexed as [k, i, r], so we need W3[k, i, r]
    # Let's compute efficiently
    T_computed = np.zeros((n, m, p))
    for r in range(rank):
        T_computed += U3[:, :, r][:, :, np.newaxis] * V3[:, :, r][np.newaxis, :, :] * W3[:, :, r].T[np.newaxis, np.newaxis, :]
        # Wait, W3[k, i, r] means for fixed r, W3[:, :, r] is shape (p, n)
        # We need W3[k, i, r] for each (k, i)
        # So the term is U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Let me redo this more carefully
    
    # Actually let me just do it directly for small sizes
    T_computed = np.zeros((n, m, p))
    U_mat = np.array(U)  # (n*m, rank)
    V_mat = np.array(V)  # (m*p, rank)
    W_mat = np.array(W)  # (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_mat[i*m + j, r] * V_mat[j*p + k, r] * W_mat[k*n + i, r]
                T_computed[i, j, k] = val
    
    # Check if all entries are approximately 1
    return np.allclose(T_computed, 1.0, atol=tol)

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random decomposition and try to optimize it."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Normalize to avoid scaling issues
    for r in range(rank):
        norm = max(1e-10, np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r]))
        if norm > 0:
            U[:, r] /= norm
            V[:, r] /= norm
            W[:, r] /= norm
            # Scale back to maintain product
            scale = norm ** 3
            U[:, r] *= scale ** (1/3)
            V[:, r] *= scale ** (1/3)
            W[:, r] *= scale ** (1/3)
    
    return U, V, W

def _optimize_decomposition(n, m, p, rank, U0, V0, W0, max_iters=500, tol=1e-10):
    """Optimize a decomposition using scipy minimize."""
    if _time_left() < 5:
        return None, None, None, float('inf')
    
    U0 = np.array(U0, dtype=float)
    V0 = np.array(V0, dtype=float)
    W0 = np.array(W0, dtype=float)
    
    # Flatten all parameters
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    nU = n * m * rank
    nV = m * p * rank
    nW = p * n * rank
    
    def objective(x):
        U = x[:nU].reshape(n*m, rank)
        V = x[nU:nU+nV].reshape(m*p, rank)
        W = x[nU+nV:].reshape(p*n, rank)
        
        # Compute tensor and compare to ones
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:]) * 0  # placeholder
                    # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    T[i,j,k] = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
        
        # We want T[i,j,k] = 1 for all i,j,k
        diff = T - 1.0
        return np.sum(diff**2)
    
    # Use L-BFGS-B for efficiency
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol})
    except Exception:
        return None, None, None, float('inf')
    
    if _time_left() < 5:
        return None, None, None, float('inf')
    
    x = result.x
    U = x[:nU].reshape(n*m, rank)
    V = x[nU:nU+nV].reshape(m*p, rank)
    W = x[nU+nV:].reshape(p*n, rank)
    
    return U, V, W, result.fun

def _try_rank(n, m, p, rank, n_restarts=10, seed_base=0):
    """Try to find a valid decomposition of given rank."""
    best_U, best_V, best_W = None, None, None
    best_cost = float('inf')
    
    for restart in range(n_restarts):
        if _time_left() < 10:
            break
        
        seed = seed_base + restart * 1000
        U0, V0, W0 = _random_decomposition(n, m, p, rank, seed=seed)
        
        U, V, W, cost = _optimize_decomposition(n, m, p, rank, U0, V0, W0, max_iters=300)
        
        if U is not None and cost < best_cost:
            best_U, best_V, best_W = U, V, W
            best_cost = cost
        
        # Check if we found an exact solution
        if best_cost < 1e-10:
            # Verify
            if _check_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
                return best_U, best_V, best_W
    
    return best_U, best_V, best_W

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication."""
    n, m, p = N, M, P
    
    # Start with schoolbook and try to reduce rank
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # For 2x5x6, the Strassen-like bounds suggest rank around 30-40
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 30
    
    # Try a few candidate ranks
    candidate_ranks = [lower_bound, lower_bound + 1, lower_bound + 2, lower_bound + 3, 
                       lower_bound + 5, lower_bound + 8, lower_bound + 10]
    
    # Also include schoolbook as fallback
    best_result = None
    best_rank = RANK_SCHOOLBOOK
    
    for rank in sorted(candidate_ranks):
        if _time_left() < 15:
            break
        
        U, V, W = _try_rank(n, m, p, rank, n_restarts=5, seed_base=rank)
        
        if U is not None:
            if _check_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_result = (U, V, W)
                best_rank = rank
                break  # Found minimal rank, stop
    
    # Fallback: schoolbook decomposition
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
        best_rank = RANK_SCHOOLBOOK
    else:
        U, V, W = best_result
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # The problem says "integer/half-integer entries" - let's try to round
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Try to snap to nearest half-integer
    def snap_to_half(x):
        return round(2*x) / 2.0
    
    U_list = [[snap_to_half(U_list[i][j]) for j in range(len(U_list[i]))] for i in range(len(U_list))]
    V_list = [[snap_to_half(V_list[i][j]) for j in range(len(V_list[i]))] for i in range(len(V_list))]
    W_list = [[snap_to_half(W_list[i][j]) for j in range(len(W_list[i]))] for i in range(len(W_list))]
    
    # Verify after snapping
    if not _check_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
        # Snapping broke it, use original
        U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W
