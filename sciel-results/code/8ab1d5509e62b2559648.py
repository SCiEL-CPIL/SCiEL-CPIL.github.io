import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_time = 120
    start_time = time.time()
    
    # Fallback: Schoolbook
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Target rank to try
    target_rank = 30
    
    def verify(U, V, W):
        # Check if U, V, W define the tensor exactly
        # Tensor T[i,j,k] = 1 if j is middle index
        # Reconstruction: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ?
        # No, the decomposition is T = sum U_c \otimes V_c \otimes W_c
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 if i,j,k correspond to a valid multiplication term?
        # The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
        # has entries T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # Actually, the tensor is defined such that <T, A \otimes B> = C.
        # T_{i,j,k} = 1 if j is the shared index.
        # So T[i, j, k] = 1 for all i,j,k? No.
        # T is a 3-tensor. T[i, j, k] = 1 if the multiplication involves A_{i,j} and B_{j,k}.
        # Yes, T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Wait, the standard tensor for matrix multiplication has T_{i,j,k} = 1 if j is the middle index.
        # So T[i, j, k] = 1 for all valid i,j,k.
        # Let's verify.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # This is a trilinear form.
        # The tensor T has T_{i,j,k} = 1.
        
        # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
        
        # Efficient check:
        # Reshape U, V, W?
        # This check is expensive. O(n*m*p*rank).
        # 3*4*6*30 = 2160 ops. Fast.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half(x):
        return np.round(x * 2) / 2.0

    def local_search(rank):
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for _ in range(100):
            # Update U
            # T = sum V_c \otimes W_c
            # We want U to minimize || T - sum U_c \otimes V_c \otimes W_c ||
            # This is linear in U.
            # For each c, U[:, c] is a variable.
            # Let's just do gradient descent on the error.
            # Error E = sum_{i,j,k} (T_{i,j,k} - sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c})^2
            
            # This is complex to implement efficiently in pure python/numpy without precomputation.
            # Given time, maybe just random search with snapping?
            
            if time.time() - start_time > target_time - 5:
                return None
            
            # Random perturbation
            U += np.random.randn(n * m, rank) * 0.01
            V += np.random.randn(m * p, rank) * 0.01
            W += np.random.randn(p * n, rank) * 0.01
            
            # Snap
            U = snap_to_half(U)
            V = snap_to_half(V)
            W = snap_to_half(W)
            
            if verify(U, V, W):
                return U, V, W
                
        return None

    # Try a few random restarts for rank 30
    for _ in range(5):
        if time.time() - start_time > target_time - 10:
            break
        res = local_search(target_rank)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return get_schoolbook()
# EVOLVE-BLOCK-END
