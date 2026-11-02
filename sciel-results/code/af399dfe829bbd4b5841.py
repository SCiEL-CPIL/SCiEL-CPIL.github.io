"""Rewritten program: search for a lower-rank exact tensor decomposition of 4x4 * 4x7."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# Known lower bounds / trivial upper bound
# n=4, m=4, p=7 -> schoolbook rank = 112
# We try to find a decomposition with rank < 112 using structured ansatzes.

def _verify(U, V, W, n=4, m=4, p=7, tol=1e-6):
    """Check if U,V,W exactly decompose the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    which is 1 if (i,j,k) is a valid matmul term, else 0.
    Actually the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the identity map on the tensor product).
    Wait: C = A @ B where A is n x m, B is m x p.
    C[i,k] = sum_j A[i,j] B[j,k].
    The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build the tensor contraction
    # For efficiency, loop over c and accumulate
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # This is an outer product of three vectors indexed by (i,j), (j,k), (k,i)
        # We can do: for i,j,k: T[i,j,k] += uc[i,j]*vc[j,k]*wc[k,i]
        # Vectorized: 
        # Let's use einsum: uc[i,j] * vc[j,k] * wc[k,i] -> sum over i,j,k gives a scalar, but we want the full tensor.
        # Actually, we can compute this as:
        # M1 = uc @ vc  -> shape (n, p)
        # Then T[i,j,k] += M1[i,k] * wc[k,i] ... no, that's not right.
        # T[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # Let's just do a triple loop for small sizes, or use broadcasting.
        # n=4, m=4, p=7 -> 112 elements per c. rank up to ~100. Total ~11200 ops. Fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank(rank, n=4, m=4, p=7, max_iter=500, seed=0):
    """Try to find a decomposition of given rank using random initialization + optimization."""
    np.random.seed(seed)
    random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iter):
        # Compute gradient of loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        err = T - 1.0
        loss = np.sum(err ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if loss < 1e-10:
            return best_U, best_V, best_W, loss
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = err[i, j, k]
                        dU[i * m + j, c] += 2 * e * vc[j, k] * wc[k, i]
                        dV[j * p + k, c] += 2 * e * uc[i, j] * wc[k, i]
                        dW[k * n + i, c] += 2 * e * uc[i, j] * vc[j, k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        # Decay learning rate
        lr *= 0.995
    
    return best_U, best_V, best_W, best_loss

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _construct_schoolbook()
    
    # Try to find lower rank decompositions
    # Known results: for 4x4*4x7, the minimal rank is unknown but likely < 112.
    # Let's try ranks from 100 down to 80.
    for rank in range(100, 75, -1):
        if time.time() > deadline - 30:
            break
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            U, V, W, loss = _try_rank(rank, n, m, p, max_iter=200, seed=seed)
            if loss < 1e-8:
                # Verify with higher precision
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better found, return schoolbook
    U_fb_list = U_fb.tolist()
    V_fb_list = V_fb.tolist()
    W_fb_list = W_fb.tolist()
    return {"U": U_fb_list, "V": V_fb_list, "W": W_fb_list}
