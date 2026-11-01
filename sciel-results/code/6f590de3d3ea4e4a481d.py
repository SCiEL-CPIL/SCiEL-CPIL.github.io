import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # Try to find lower rank using Winograd-like structure
    # For 2x3 * 3x3, try rank 12, 11, 10, ...
    
    # The tensor T has entries T[i,j,k] = 1 if they form a valid multiplication
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # Actually the tensor is defined as: T[a,b,c] = delta where a indexes U rows, b indexes V rows, c indexes W cols
    
    # Let's try a structured approach: use block Winograd
    # For the 3x3 multiplication part, use 4-point Winograd transform
    # This gives us a decomposition with rank = 2 * 4 = 8? Let's verify.
    
    # Winograd F_3,3 transform matrices
    # Input transform G (4x3):
    G = np.array([
        [1, 0, 0],
        [1, 1, 1],
        [1, -1, 1],
        [0, 1, -1]
    ], dtype=float)
    
    # Output transform A^T (3x4):
    AT = np.array([
        [1, 1, 1, 0],
        [0, 1, -1, -1],
        [1, 0, -1, 1]
    ], dtype=float)
    
    # Verify: AT @ G should give the 3x3 convolution kernel
    # The convolution matrix for 3-point conv is:
    # [[b0, b1, b2],
    #  [b0, b1, b2],
    #  [b0, b1, b2]] -- no, that's not right
    
    # Actually, for matrix multiplication C = AB, each row of C is a linear combination
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's try: for each row i of A, compute y_i = G * A[i,:] (4 values)
    # Then for each column k of B, compute v_k = (P^T) * B[:,k] where P^T is related to AT
    # Actually, we need: C[i,:] = A[i,:] @ B
    # Using Winograd: m_i = G @ A[i,:] (4 values)
    # v_k = P @ B[:,k] where P is 4x3
    # u = m_i * v (pointwise, 4 values)
    # C[i,:] = AT @ u
    
    # So we need G and P such that: AT @ (G @ a) * (P @ b) gives a @ b for all 3-vectors a, b
    # This means: AT @ diag(G @ a) @ P @ b = a @ b^T (element-wise for each output)
    
    # Actually, the condition is: for all a, b in R^3:
    # (AT @ ((G @ a) * (P @ b))) = a @ b (convolution)
    
    # Let's find P. We need: AT @ diag(G@a) @ P @ b = sum_j a_j b_j for the appropriate outputs
    
    # This is getting complex. Let me just try rank values and use a simple verification.
    
    # Try structured Winograd decomposition
    # For each of the 2 rows of A, use 4-point Winograd
    # Total rank: 2 * 4 = 8
    
    # But we need to handle the B matrix properly.
    # Let's define:
    # U[i*3 + t, r] for i in {0,1}, t in {0,1,2,3} (4 Winograd points)
    # V[j*3 + t, r] for j in {0,1,2}, t in {0,1,2,3}
    # W[k*2 + i, r] for k in {0,1,2}, i in {0,1}
    
    # The decomposition: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = delta_{j} (for the j index)
    # Wait, the tensor is: T[i,j,k] where i indexes U row (i*m+j), j indexes V row (j*p+k), k indexes W col (k*n+i)
    
    # Actually, the tensor for matrix multiplication M_{n,m,p} has:
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i,j,k
    # Otherwise T[a,b,c] = 0
    
    # The decomposition requires: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Let me just try to verify the schoolbook and then attempt lower ranks with random search
    
    # Verification function
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        
        # Compute reconstructed tensor
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    # Try Winograd-based structured decomposition
    # For 2x3 * 3x3, let's try a block approach
    
    # Method: For each row i of A, use Winograd F_3,3
    # U[i*3 + j, r] = G[r, j] for r in 0..3, j in 0..2
    # But we need to handle B and W properly
    
    # Let's try: rank = 8 (2 rows * 4 Winograd points)
    # U shape: (6, 8), V shape: (9, 8), W shape: (6, 8)
    
    # For row i of A (i=0,1):
    #   U[i*3 + j, 4*i + t] = G[t, j] for t=0..3, j=0..2
    #   V[j*3 + k, 4*i + t] = P[t, j, k]... this doesn't work directly
    
    # The issue is that B is 3x3, so each column of B is a 3-vector
    # For Winograd on B columns: v_k = P @ B[:,k]
    # So V[j*3 + k, r] should be P[r, j] for the appropriate r
    
    # Let's define:
    # For i in {0,1}, t in {0,1,2,3}:
    #   r = 4*i + t
    #   U[i*3 + j, r] = G[t, j]
    #   V[j*3 + k, r] = P[t, j]  (P is 4x3, same as G^T or related)
    #   W[k*2 + i, r] = AT[t, k]  (AT is 3x4, so AT[k, t])
    
    # Then: sum_r U[i*3+j, r] V[j*3+k, r] W[k*2+i, r]
    # = sum_t G[t,j] * P[t,j] * AT[k,t]  (summing over r=4i+t, but i is fixed)
    # Wait, r = 4*i + t, so for fixed i, we sum over t
    
    # = sum_t G[t,j] * P[t,j] * AT[k,t]
    
    # We need this to equal 1 if the indices are consistent, else 0
    # But this sum doesn't depend on i or k properly...
    
    # This approach has issues. Let me just do a random search for lower ranks.
    
    # Random search for rank reduction
    for target_rank in [17, 16, 15, 14, 13, 12, 11, 10, 9, 8]:
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        max_attempts = 50 if target_rank >= 12 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + target_rank * 1000 + attempt)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Simple verification
            if verify(U, V, W):
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank == target_rank:
            break
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
