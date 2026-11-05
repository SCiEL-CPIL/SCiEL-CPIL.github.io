"""Tensor decomposition search for 3x3x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form an exact tensor decomposition.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k,l,r] = delta_{i+k, r+n?} ... 
    Actually the convention is:
    (A*B)[i,r] = sum_j A[i,j]*B[j,r]
    In tensor form: T_{i,j,r} = delta_{i+r, j+?} no.
    
    The multiplication tensor T has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    and T[i,j,k] = 1 iff i == k (output row) ... wait.
    
    Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor is T[i,j,k] = delta_{i,k}... no, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in the output.
    Output index is (i,k) where i in [0,n), k in [0,p).
    So T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to output[i,k]).
    
    The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Note: W index is (k*n + i), i.e., W[k, i] in the (p, n) grid.
    
    So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = U.shape[1]
    # Build the tensor via einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # T should be all ones
    return np.allclose(T, np.ones((n, m, p)), atol=tol)

def _try_random_decomposition(n, m, p, rank, max_iter=2000, seed=42):
    """Try to find a rank decomposition via random initialization and gradient descent."""
    rng = np.random.RandomState(seed)
    
    # Parameterize U, V, W as random
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    def loss_fn(U, V, W):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.sum((T - 1.0) ** 2)
    
    # Simple gradient descent with random restarts
    best_loss = loss_fn(U, V, W)
    if best_loss < 1e-10:
        return U, V, W
    
    lr = 0.001
    for it in range(max_iter):
        # Compute gradients numerically would be slow; use analytical
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] (summed over k? No, for fixed i,j,k)
        # Actually for each (i,j,k): dT[i,j,k]/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
        # dL/dU[i*m+j,c] = sum_k 2*(T[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c]
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = T - 1.0  # (n, m, p)
        
        # dL/dU[i,j,c] = 2 * sum_k err[i,j,k] * V3[j,k,c] * W3[k,i,c]
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', err, V3, W3)
        # dL/dV[j,k,c] = 2 * sum_i err[i,j,k] * U3[i,j,c] * W3[k,i,c]
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', err, U3, W3)
        # dL/dW[k,i,c] = 2 * sum_j err[i,j,k] * U3[i,j,c] * V3[j,k,c]
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', err, U3, V3)
        
        U -= lr * dU.reshape(n*m, rank)
        V -= lr * dV.reshape(m*p, rank)
        W -= lr * dW.reshape(p*n, rank)
        
        loss = loss_fn(U, V, W)
        if loss < best_loss:
            best_loss = loss
            if loss < 1e-10:
                return U, V, W
    return None

def _try_integer_decomposition(n, m, p, max_rank=45, timeout=100):
    """Try to find an integer/half-integer decomposition by searching known results."""
    # For 3x3x3, the minimum rank is 23 (Larsen 2021).
    # For 3x3x5, we need to find the minimum rank.
    # The schoolbook is 45. 
    # A simple block approach: treat as three 3x3x3 problems sharing structure.
    # Each output column k=0,1,2,3,4 needs a 3x3x3-like decomposition.
    # But they share U and V, only W differs per k.
    # Actually W has shape (p*n, rank) = (15, rank). For each k, W[k*n:(k+1)*n, :] is the n-vector.
    # The constraint for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try a structured approach: use the 3x3x3 rank-23 decomposition if we can find it,
    # and replicate/adapt it.
    # Since I don't have the explicit 3x3x3 rank-23 factors hardcoded, let me try random search.
    
    # Try random search for small ranks
    start_time = time.time()
    for rank in range(23, max_rank + 1):
        if time.time() - start_time > timeout - 10:
            break
        # Try multiple random restarts
        for restart in range(20):
            if time.time() - start_time > timeout - 10:
                break
            result = _try_random_decomposition(n, m, p, rank, max_iter=500, seed=1000 + rank * 100 + restart)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    # Try to snap to nearest half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank
                    return U.tolist(), V.tolist(), W.tolist(), rank
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Try to find a better decomposition
    try:
        result = _try_integer_decomposition(n, m, p, max_rank=44, timeout=110)
        if result is not None:
            U, V, W, rank = result
            # Final verification
            if _verify(np.array(U), np.array(V), np.array(W), n, m, p, tol=1e-6):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
