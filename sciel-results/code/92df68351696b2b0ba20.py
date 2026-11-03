import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    # The target tensor is 1 for all i, j, k
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _schoolbook():
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return U_sb, V_sb, W_sb

def _try_optimization_search(deadline):
    # Try to find a rank 23 decomposition
    target_rank = 23
    best_rank = float('inf')
    best_result = None
    
    # Initialize with random values
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Simple gradient descent on Frobenius norm
        # Error = || T_reconstructed - 1 ||^2
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
                
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            
            T_rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            error = T_rec - 1.0
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                # Check half-integer constraint after rounding
                U_r = np.round(2 * U) / 2
                V_r = np.round(2 * V) / 2
                W_r = np.round(2 * W) / 2
                if _verify(U_r, V_r, W_r):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = (U_r, V_r, W_r)
                break
            
            # Gradients
            dU = np.einsum('ijk,jkr->ijr', error, W_3d) # Wait, V is jkr? No, V is jkr in einsum?
            # T = U_ijr V_jkr W_kir
            # dT/dU_ijr = V_jkr W_kir
            # dT/dV_jkr = U_ijr W_kir
            # dT/dW_kir = U_ijr V_jkr
            
            dU = np.einsum('ijk,jkr->ijr', error, W_3d) # This is wrong. W is kir.
            # Need to align indices.
            # dU_ijr = sum_{j,k} error_ijk * V_jkr * W_kir? No.
            # T_ijk = sum_r U_ijr V_jkr W_kir
            # dT_ijk/dU_ijr = V_jkr W_kir
            # dU_ijr = 2 * error_ijk * V_jkr * W_kir (summed over j,k? No, j,k are fixed for T_ijk)
            # Wait, U_ijr depends on i,j,r.
            # dL/dU_ijr = 2 * sum_{k} error_ijk * V_jkr * W_kir
            # Let's use np.einsum carefully.
            
            # dU: i, j, r
            # error: i, j, k
            # V: j, k, r
            # W: k, i, r
            # dU_ijr = 2 * sum_k error_ijk * V_jkr * W_kir
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', error, V_3d, W_3d)
            
            # dV: j, k, r
            # dV_jkr = 2 * sum_i error_ijk * U_ijr * W_kir
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', error, U_3d, W_3d)
            
            # dW: k, i, r
            # dW_kir = 2 * sum_j error_ijk * U_ijr * V_jkr
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', error, U_3d, V_3d)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Normalize to prevent divergence
            # U /= np.linalg.norm(U)
            
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try optimization
    result = _try_optimization_search(deadline)
    
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
