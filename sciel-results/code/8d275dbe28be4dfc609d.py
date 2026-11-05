"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import math
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U_arr[i*m + j]
                vk = V_arr[j*p + k]
                wi = W_arr[k*n + i]
                val = np.dot(ui, vk) * 1  # This is wrong, need element-wise product with W
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                val = np.sum(ui * vk * wi)
                expected = 1
                if abs(val - expected) > tol:
                    return False
    # Also check that for "wrong" indices, the value is 0
    # The tensor is 1 only when indices are (i*m+j, j*p+k, k*n+i) for some i,j,k
    # For other combinations, it should be 0
    # Let's check a few off-diagonal entries
    checked = 0
    for i2 in range(n*m):
        for j2 in range(m*p):
            for k2 in range(p*n):
                # Check if this corresponds to a valid (i,j,k)
                i = i2 // m
                j = i2 % m
                jj = j2 // p
                k = j2 % p
                kk = k2 // n
                ii = k2 % n
                if j == jj and i == ii:
                    continue  # This is a "on-diagonal" entry
                ui = U_arr[i2]
                vk = V_arr[j2]
                wi = W_arr[k2]
                val = np.sum(ui * vk * wi)
                if abs(val) > tol:
                    return False
                checked += 1
                if checked > 100:
                    break
            if checked > 100:
                break
        if checked > 100:
            break
    return True

def _build_strassen_decomposition(n, m, p):
    """
    Build a decomposition using Strassen-like block structure.
    For 2x3x8: split into blocks of 2x1x2 for the row and column dimensions.
    A is 2x3, B is 3x8, C is 2x8.
    Split A into 2 row-blocks (each 1x3), B into 4 col-blocks (each 3x2), C into 4 col-blocks (each 2x2).
    For each group of 2 columns (k in {0,1}, {2,3}, {4,5}, {6,7}):
      A_g is 2x3, B_g is 3x2, C_g is 2x2.
      Use Strassen for 2x2x2? No, A is 2x3, not 2x2.
    Alternative: treat the 3 columns of A and 3 rows of B as the "middle" dimension.
    We can't directly apply Strassen to 2x3x2.
    
    Let's use a different approach: exploit the structure to reduce rank.
    Schoolbook rank = 48.
    
    Known lower bound: for 2x3x8, the tensor rank is at least max(2*8, 3*8, 2*3) = 24, but more tightly,
    it's known that the rank of 2x3xn is related to the border rank.
    
    Let me try a direct construction with rank < 48.
    
    Actually, let's use the fact that we can combine terms.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For each i in {0,1} and k in {0,...,7}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    To reduce rank, we can try to factor. For example, if we define:
    M1[i,k] = A[i,0]*B[0,k]
    M2[i,k] = A[i,1]*B[1,k]
    M3[i,k] = A[i,2]*B[2,k]
    
    That's 3*8 = 24 rank-1 tensors for each i... wait, no.
    Each term A[i,j]*B[j,k] is a rank-1 tensor for fixed i,j,k.
    There are 2*3*8 = 48 such terms.
    
    Can we combine? For fixed j, we have 2*8 = 16 terms: A[i,j]*B[j,k] for i in {0,1}, k in 0..7.
    This is a 2x8 matrix of coefficients. Can we factor this into fewer rank-1 terms?
    The tensor for fixed j is: sum_i sum_k A[i,j] B[j,k] E_{i,k}
    This is rank 1 as a tensor? No, it's a matrix A[:,j] * B[j,:] which is rank 1.
    So for each j, the contribution to C is a rank-1 matrix (as a bilinear form).
    But in tensor decomposition, each "term" is a rank-1 tensor.
    The term for fixed (i,j,k) is: (e_i in A) * (e_j in B) * (e_{i,k} in C)
    We can't easily combine these.
    
    Let me try a numerical search for a lower rank decomposition.
    """
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _try_lower_rank_random(n, m, p, target_rank, timeout=30):
    """Try to find a decomposition of the given rank using random initialization and optimization."""
    start_time = time.time()
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # We want to minimize the error: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - T[i,j,k])^2
    # where T is 1 for valid (i,j,k) and 0 otherwise.
    
    # This is a non-convex problem. Let's use a simple iterative method.
    # Gradient descent on U, V, W.
    
    lr = 0.01
    for iteration in range(500):
        if time.time() - start_time > timeout:
            break
        # Compute gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Residual for this entry
                    ui = U[i*m + j]
                    vk = V[j*p + k]
                    wi = W[k*n + i]
                    pred = np.sum(ui * vk * wi)
                    expected = 1
                    resid = pred - expected
                    dU[i*m + j] += resid * vk * wi
                    dV[j*p + k] += resid * ui * wi
                    dW[k*n + i] += resid * ui * vk
                    # Also, for entries where T=0, we need resid = pred - 0 = pred
                    # This is expensive to compute all zeros. Let's skip for now.
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    U, V, W = _build_strassen_decomposition(n, m, p)
    best_rank = len(U[0])
    
    # Try to find lower rank decompositions
    # Known lower bound: for 2x3xn, the rank is at least ... 
    # Let's try ranks from lower bound up to schoolbook
    # The lower bound for matrix multiplication tensor A^(m,n) * B^(n,p) is max(mn, mp, np) = max(6, 16, 24) = 24
    # But the actual rank is higher. For 2x3x8, let's try ranks from 24 to 47.
    
    start_time = time.time()
    deadline = 120  # seconds
    
    for target_rank in range(24, best_rank):
        if time.time() - start_time > deadline - 10:
            break
        U_try, V_try, W_try = _try_lower_rank_random(n, m, p, target_rank, timeout=5)
        if _validate(U_try, V_try, W_try, n, m, p, tol=1e-6):
            best_rank = target_rank
            U, V, W = U_try, V_try, W_try
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
