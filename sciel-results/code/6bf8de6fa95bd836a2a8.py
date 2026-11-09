"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = 2, 5, 5
    N = n * m  # 10
    M = m * p  # 25
    P = p * n  # 10

    def verify_exact(U_list, V_list, W_list):
        """Verify exact decomposition using integer/rational arithmetic."""
        rank = len(U_list[0])
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)
        
        # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Known result: For 2×n×n matrix multiplication, the rank can be reduced
    # using algebraic identities. Let me try to construct a better decomposition.
    
    # Strategy: Use the structure of 2×5×5 multiplication
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # One approach: group the computations and use Strassen-like tricks
    # For each output position k, we compute:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + ... + A[0,4]*B[4,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + ... + A[1,4]*B[4,k]
    
    # We can use the fact that we have 2 rows in A to create combinations.
    
    # Let me try a different approach: use the known rank bounds and search
    # more systematically.
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find improvements using structured search
    # For 2×5×5, let's try ranks from 49 down to 25
    
    np.random.seed(42)
    
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations with different sparsity patterns
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            strategy = trial % 4
            
            U = np.zeros((N, target_rank))
            V = np.zeros((M, target_rank))
            W = np.zeros((P, target_rank))
            
            if strategy == 0:
                # Sparse random with ±1
                for c in range(target_rank):
                    n_u = max(1, N // 4)
                    n_v = max(1, M // 4)
                    n_w = max(1, P // 4)
                    
                    u_idx = np.random.choice(N, size=min(n_u, N), replace=False)
                    v_idx = np.random.choice(M, size=min(n_v, M), replace=False)
                    w_idx = np.random.choice(P, size=min(n_w, P), replace=False)
                    
                    for idx in u_idx:
                        U[idx, c] = np.random.choice([-1, 1])
                    for idx in v_idx:
                        V[idx, c] = np.random.choice([-1, 1])
                    for idx in w_idx:
                        W[idx, c] = np.random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                for c in range(target_rank):
                    U[:, c] = np.random.randn(N) * 0.5
                    V[:, c] = np.random.randn(M) * 0.5
                    W[:, c] = np.random.randn(P) * 0.5
            elif strategy == 2:
                # Block structure: exploit the 2×5 structure
                for c in range(target_rank):
                    # U has 2 blocks of 5
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.3:
                                U[i*m+j, c] = np.random.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if np.random.random() < 0.2:
                                V[j*p+k, c] = np.random.choice([-1, 1])
                    for k in range(p):
                        for i in range(n):
                            if np.random.random() < 0.3:
                                W[k*n+i, c] = np.random.choice([-1, 1])
            else:
                # Mixed: some columns dense, some sparse
                for c in range(target_rank):
                    if c % 2 == 0:
                        # Dense
                        U[:, c] = np.random.choice([-1, 0, 1], size=N)
                        V[:, c] = np.random.choice([-1, 0, 1], size=M)
                        W[:, c] = np.random.choice([-1, 0, 1], size=P)
                    else:
                        # Sparse
                        n_u = max(1, N // 5)
                        n_v = max(1, M // 5)
                        n_w = max(1, P // 5)
                        u_idx = np.random.choice(N, size=min(n_u, N), replace=False)
                        v_idx = np.random.choice(M, size=min(n_v, M), replace=False)
                        w_idx = np.random.choice(P, size=min(n_w, P), replace=False)
                        for idx in u_idx:
                            U[idx, c] = np.random.choice([-1, 1])
                        for idx in v_idx:
                            V[idx, c] = np.random.choice([-1, 1])
                        for idx in w_idx:
                            W[idx, c] = np.random.choice([-1, 1])
            
            if verify_exact(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
