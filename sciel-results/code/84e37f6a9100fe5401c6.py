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
    which is 1 for all i,j,k (the matmul tensor is all ones).
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build the tensor contraction vectorized
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # wc[k,i] -> wc.T[i,k], so wc.T has shape (n, p)
        T += uc[:, :, None] * vc[None, :, :] * wc.T[:, None, :]
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
        # Compute T vectorized
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            T += uc[:, :, None] * vc[None, :, :] * wc.T[:, None, :]
        err = T - 1.0
        loss = np.sum(err ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if loss < 1e-10:
            return best_U, best_V, best_W, loss
        
        # Compute gradients vectorized
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # dU[i*m+j, c] += 2 * err[i,j,k] * vc[j,k] * wc[k,i] summed over k
            # dV[j*p+k, c] += 2 * err[i,j,k] * uc[i,j] * wc[k,i] summed over i
            # dW[k*n+i, c] += 2 * err[i,j,k] * uc[i,j] * vc[j,k] summed over j
            
            # dU: for each (i,j), sum over k: 2*err[i,j,k]*vc[j,k]*wc[k,i]
            # err[i,j,k] * vc[j,k] -> shape (n,m,p) * (m,p) broadcast
            # Let's compute step by step
            # tmp1[i,j,k] = err[i,j,k] * vc[j,k]  -> (n,m,p)
            tmp1 = err * vc[None, :, :]  # (n,m,p) * (1,m,p) -> (n,m,p)
            # dU[i,j,c] = sum_k 2*tmp1[i,j,k]*wc[k,i]
            # wc[k,i] -> wc.T[i,k], shape (n,p)
            dU_c = 2 * np.einsum('ijk,ik->ij', tmp1, wc.T)  # (n,m)
            dU[:, c] = dU_c.flatten()
            
            # dV: for each (j,k), sum over i: 2*err[i,j,k]*uc[i,j]*wc[k,i]
            # tmp2[i,j,k] = err[i,j,k] * wc[k,i] -> need wc[k,i] broadcast to (n,m,p)
            # wc.T[i,k] shape (n,p), broadcast to (n,m,p) as wc.T[:,None,:]
            tmp2 = err * wc.T[:, None, :]  # (n,m,p) * (n,1,p) -> (n,m,p)
            # dV[j,k,c] = sum_i 2*tmp2[i,j,k]*uc[i,j]
            dV_c = 2 * np.einsum('ijk,ij->jk', tmp2, uc)  # (m,p)
            dV[:, c] = dV_c.flatten()
            
            # dW: for each (k,i), sum over j: 2*err[i,j,k]*uc[i,j]*vc[j,k]
            # tmp3[i,j,k] = err[i,j,k] * uc[i,j] -> (n,m,p) * (n,m) broadcast
            tmp3 = err * uc[:, :, None]  # (n,m,p) * (n,m,1) -> (n,m,p)
            # dW[k,i,c] = sum_j 2*tmp3[i,j,k]*vc[j,k]
            # vc[j,k] shape (m,p), we need to sum over j for each (i,k)
            dW_c = 2 * np.einsum('ijk,jk->ik', tmp3, vc)  # (n,p) -> this is dW.T
            dW[:, c] = dW_c.T.flatten()
        
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
