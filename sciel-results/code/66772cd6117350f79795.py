"""Program for 4x4x8 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-40 decomposition for 4x4x8 matrix multiplication.
    
    The 4x4x8 multiplication can be seen as two independent 4x4x4 multiplications
    (B has two 4x4 blocks). We use a rank-20 decomposition for each 4x4x4 block,
    giving total rank 40.
    
    For 4x4x4 matrix multiplication (A * B where A, B are 4x4):
    - U has shape (4*4, 20) = (16, 20)
    - V has shape (4*4, 20) = (16, 20)  
    - W has shape (4*4, 20) = (16, 20)
    
    For 4x4x8:
    - U has shape (4*4, 40) = (16, 40)
    - V has shape (4*8, 40) = (32, 40)
    - W has shape (8*4, 40) = (32, 40)
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Find a rank-20 decomposition for 4x4x4 matrix multiplication
    # Using numerical optimization
    n4 = 4
    rank4 = 20
    
    # The tensor T[i,j,k] = 1 for all i,j,k in 0..3
    # We need: sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] = 1 for all i,j,k
    
    # Initialize with random values
    np.random.seed(42)
    
    best_err = float('inf')
    best_U4 = None
    best_V4 = None
    best_W4 = None
    
    # Try multiple random initializations
    for attempt in range(50):
        U4 = np.random.randn(n4 * n4, rank4)
        V4 = np.random.randn(n4 * n4, rank4)
        W4 = np.random.randn(n4 * n4, rank4)
        
        # Gradient descent to minimize error
        lr = 0.01
        for step in range(2000):
            # Compute current tensor values
            T_current = np.zeros((n4, n4, n4))
            for i in range(n4):
                for j in range(n4):
                    for k in range(n4):
                        T_current[i, j, k] = np.dot(
                            U4[i * n4 + j, :], 
                            V4[j * n4 + k, :] * W4[k * n4 + i, :]
                        )
            
            # Compute error
            error = T_current - 1.0
            max_err = np.max(np.abs(error))
            
            if max_err < 1e-10:
                break
            
            # Compute gradients
            grad_U4 = np.zeros_like(U4)
            grad_V4 = np.zeros_like(V4)
            grad_W4 = np.zeros_like(W4)
            
            for i in range(n4):
                for j in range(n4):
                    for k in range(n4):
                        err_ijk = error[i, j, k]
                        for c in range(rank4):
                            grad_U4[i * n4 + j, c] += err_ijk * V4[j * n4 + k, c] * W4[k * n4 + i, c]
                            grad_V4[j * n4 + k, c] += err_ijk * U4[i * n4 + j, c] * W4[k * n4 + i, c]
                            grad_W4[k * n4 + i, c] += err_ijk * U4[i * n4 + j, c] * V4[j * n4 + k, c]
            
            # Update
            U4 -= lr * grad_U4
            V4 -= lr * grad_V4
            W4 -= lr * grad_W4
        
        # Check final error
        T_final = np.zeros((n4, n4, n4))
        for i in range(n4):
            for j in range(n4):
                for k in range(n4):
                    T_final[i, j, k] = np.dot(
                        U4[i * n4 + j, :], 
                        V4[j * n4 + k, :] * W4[k * n4 + i, :]
                    )
        
        final_err = np.max(np.abs(T_final - 1.0))
        
        if final_err < best_err:
            best_err = final_err
            best_U4, best_V4, best_W4 = U4.copy(), V4.copy(), W4.copy()
        
        if best_err < 1e-10:
            break
    
    U4, V4, W4 = best_U4, best_V4, best_W4
    
    # Try rounding to half-integers and verify
    U4r = np.round(U4 * 2) / 2.0
    V4r = np.round(V4 * 2) / 2.0
    W4r = np.round(W4 * 2) / 2.0
    
    max_err_r = 0.0
    for i in range(n4):
        for j in range(n4):
            for k in range(n4):
                s = np.dot(U4r[i * n4 + j, :], V4r[j * n4 + k, :] * W4r[k * n4 + i, :])
                max_err_r = max(max_err_r, abs(s - 1.0))
    
    if max_err_r < 1e-10:
        U4, V4, W4 = U4r, V4r, W4r
    
    # Construct 4x4x8 decomposition by concatenating two 4x4x4 blocks
    rank_total = 2 * rank4
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Block 1: k = 0,1,2,3 (first 4 columns of B)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for c in range(rank4):
                    U[i * 4 + j, c] = U4[i * 4 + j, c]
                    V[j * 4 + k, c] = V4[j * 4 + k, c]
                    W[k * 4 + i, c] = W4[k * 4 + i, c]
    
    # Block 2: k = 4,5,6,7 (last 4 columns of B)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for c in range(rank4):
                    U[i * 4 + j, rank4 + c] = U4[i * 4 + j, c]
                    V[j * 4 + (k + 4), rank4 + c] = V4[j * 4 + k, c]
                    W[(k + 4) * 4 + i, rank4 + c] = W4[k * 4 + i, c]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
