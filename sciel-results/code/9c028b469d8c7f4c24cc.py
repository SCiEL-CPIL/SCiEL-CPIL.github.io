"""Matrix multiplication 2x3 x 3x3: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 18

# Precompute the multiplication tensor T[i, j, k] = 1 for all i,j,k
# The tensor we need to decompose: C[i,k] = sum_j A[i,j] B[j,k]
# Tensor T[i, j, k] where T[i,j,k] = 1 (all ones) for the standard MM tensor
# Actually, the MM tensor has T[i,j,k] = delta structure.
# For MM: C[i,k] = sum_j A[i,j]*B[j,k]
# The tensor is T[i,j,k] which equals 1 if we're computing the (i,k) entry from (i,j) and (j,k)
# In fact, the MM tensor for (n,m,p) has T[i,j,k] = 1 for all valid i,j,k.
# Wait, no. The MM tensor T has T[i,j,k] = 1 if the index structure matches.
# For standard matrix multiplication A(n x m) * B(m x p) = C(n x p):
# C[i,k] = sum_j A[i,j] B[j,k]
# The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
# This is the all-ones tensor of shape (n, m, p).

def _build_tensor():
    """Build the MM tensor as a 3D array T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=float)

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W give an exact decomposition of the MM tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Compute the reconstructed tensor: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Using einsum
    T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_true = _build_tensor()
    return np.allclose(T_recon, T_true, atol=tol)

def _schoolbook():
    """Return the schoolbook rank-18 decomposition."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half_int(mat):
    """Snap values to nearest half-integer and return as list of lists."""
    snapped = np.round(np.array(mat, dtype=float) * 2) / 2.0
    return snapped.tolist()

def _try_rank(rank, max_iters=5000, seed=42):
    """Try to find a rank-rank decomposition using gradient descent on the tensor."""
    rng = np.random.RandomState(seed)
    T = _build_tensor()
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        loss = np.sum((T_recon - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Compute gradients
        # dL/dU3[i,j,r] = 2 * (T_recon[i,j,k] - T[i,j,k]) * V3[j,k,r] * W3[k,i,r] summed over k
        # dL/dV3[j,k,r] = 2 * (T_recon[i,j,k] - T[i,j,k]) * U3[i,j,r] * W3[k,i,r] summed over i
        # dL/dW3[k,i,r] = 2 * (T_recon[i,j,k] - T[i,j,k]) * U3[i,j,r] * V3[j,k,r] summed over j
        
        diff = T_recon - T  # shape (n, m, p)
        
        dU3 = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V3, W3)
        dV3 = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U3, W3)
        dW3 = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U3, V3)
        
        dU = dU3.reshape(n * m, rank)
        dV = dV3.reshape(m * p, rank)
        dW = dW3.reshape(p * n, rank)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if it % 100 == 0:
            if loss < best_loss * 0.9:
                lr *= 0.9
            else:
                lr = max(0.001, lr * 0.5)
    
    # Check if we found a valid decomposition
    if best_loss < 1e-9:
        # Snap to half-integers
        U_snap = _snap_to_half_int(best_U)
        V_snap = _snap_to_half_int(best_V)
        W_snap = _snap_to_half_int(best_W)
        if _verify(U_snap, V_snap, W_snap):
            return {"U": U_snap, "V": V_snap, "W": W_snap}
        # Also check without snapping
        if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist()):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    # Start with schoolbook (rank 18)
    result = _schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 2x3 x 3x3, the rank is at least... 
    # The tensor rank of the 2x3x3 all-ones tensor.
    # Let's try ranks from 17 down to some lower bound.
    # Actually, let's be more strategic. Try a few ranks.
    
    for rank in [17, 16, 15]:
        if time.time() > deadline - 10.0:
            break
        res = _try_rank(rank, max_iters=2000, seed=rank * 1000 + 42)
        if res is not None:
            result = res
            break
    
    return result
