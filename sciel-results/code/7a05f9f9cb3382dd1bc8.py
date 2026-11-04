import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Target Rank 33
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> Rank 21 (3 Strassen 2x2x2 blocks)
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Strassen for 2x2 A_sub * 2x6 B_sub
    # A_sub = A[:, 1:3] (2x2), B_sub = B[1:3, :] (2x6)
    
    # Indices in U: i*m + j. Here i in {0,1}, j in {1,2} (cols 1,2 of A)
    # Indices in V: j*p + k. Here j in {1,2} (rows 1,2 of B), k in {0..5}
    # Indices in W: k*n + i. Here k in {0..5}, i in {0,1}
    
    # Apply Strassen-like decomposition to the 2x2x6 subproblem
    # We can use 7 multiplications per 2x2x2 block, but B has 6 columns
    # Group B columns into 3 pairs: (0,1), (2,3), (4,5)
    # For each pair, use 7 Strassen multiplications
    # Total: 3 * 7 = 21 multiplications for Part 2
    
    for t in range(3):
        # Block t uses B cols 2*t, 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3 = A00(B10 - B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4 = A11(B01 - B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify correctness with random test
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct C from decomposition
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_reconstructed += u_vec @ v_vec * w_vec.T
    
    # Check if decomposition is correct
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback to schoolbook if Strassen fails
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition using random search
    # This is a heuristic approach - try random sparse matrices and check rank
    
    # For now, return the rank-33 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
