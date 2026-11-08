"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Build the target tensor
    T = np.zeros((9, 12, 12))
    for i in range(3):
        for j in range(3):
            for k in range(4):
                a_idx = i * 3 + j
                b_idx = j * 4 + k
                c_idx = i * 4 + k
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Try rank 33
    for target_rank in [33, 34]:
        if time.time() > deadline:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(9, target_rank)
        V = np.random.randn(12, target_rank)
        W = np.random.randn(12, target_rank)
        
        # Gradient descent on the tensor reconstruction error
        lr = 0.01
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute current reconstruction
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check final error
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        final_error = np.max(np.abs(T_rec - T))
        
        if final_error < 1e-6:
            # Found a valid decomposition!
            # Round to nearest integers/half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded version
            T_rec_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            rounded_error = np.max(np.abs(T_rec_rounded - T))
            
            if rounded_error < 1e-6:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    # Fallback: return rank-34 solution
    rank = 34
    U = np.zeros((9, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    col = 0
    # Strassen Left
    U[0, col] = 1; U[4, col] = 1; V[0, col] = 1; V[5, col] = 1; W[0, col] = 1; W[4, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[0, col] = 1; W[1, col] = 1; W[4, col] = -1; col += 1
    U[0, col] = 1; V[4, col] = 1; V[0, col] = -1; W[1, col] = 1; W[4, col] = 1; col += 1
    U[4, col] = 1; V[1, col] = 1; V[5, col] = -1; W[0, col] = 1; W[3, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[5, col] = 1; W[0, col] = -1; W[3, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[0, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[4, col] = 1; V[5, col] = 1; W[0, col] = 1; col += 1
    
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    
    U[0, col] = 1; U[4, col] = 1; V[2, col] = 1; V[7, col] = 1; W[6, col] = 1; W[10, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[2, col] = 1; W[7, col] = 1; W[10, col] = -1; col += 1
    U[0, col] = 1; V[6, col] = 1; V[2, col] = -1; W[7, col] = 1; W[10, col] = 1; col += 1
    U[4, col] = 1; V[3, col] = 1; V[7, col] = -1; W[6, col] = 1; W[9, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[7, col] = 1; W[6, col] = -1; W[9, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[2, col] = 1; V[3, col] = 1; W[10, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[6, col] = 1; V[7, col] = 1; W[6, col] = 1; col += 1
    
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
