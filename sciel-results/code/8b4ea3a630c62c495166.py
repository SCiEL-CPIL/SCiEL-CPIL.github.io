"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a rank-minimal decomposition of 3x3 matrix multiplication."""
    deadline = time.time() + 115
    n, m, p = 3, 3, 3
    
    # Schoolbook fallback (rank 27)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find rank-23 decomposition via alternating least squares
    rank = 23
    np.random.seed(42)
    
    best_result = None
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with sparse half-integer entries
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with random sparse patterns
        for idx in range(n * m):
            nnz = np.random.randint(2, 5)
            cols = np.random.choice(rank, nnz, replace=False)
            U[idx, cols] = np.random.choice([0.5, 1.0, 1.5, 2.0], nnz)
        for idx in range(m * p):
            nnz = np.random.randint(2, 5)
            cols = np.random.choice(rank, nnz, replace=False)
            V[idx, cols] = np.random.choice([0.5, 1.0, 1.5, 2.0], nnz)
        for idx in range(p * n):
            nnz = np.random.randint(2, 5)
            cols = np.random.choice(rank, nnz, replace=False)
            W[idx, cols] = np.random.choice([0.5, 1.0, 1.5, 2.0], nnz)
        
        # Alternating least squares
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Solve for W given U, V
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w_sol
                    except:
                        pass
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = v_sol
                    except:
                        pass
            
            # Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    C = np.zeros((p, rank))
                    for k in range(p):
                        C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(C, np.ones(p), rcond=None)
                        U[i * m + j, :] = u_sol
                    except:
                        pass
            
            # Check convergence and validity
            T_computed = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
            
            if np.allclose(T_computed, 1.0, atol=1e-4):
                # Check for half-integer entries
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                if np.allclose(U_half, U, atol=1e-6) and np.allclose(V_half, V, atol=1e-6) and np.allclose(W_half, W, atol=1e-6):
                    # Verify with snapped values
                    T_check = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check[i, j, k] = np.dot(U_half[i * m + j, :], V_half[j * p + k, :] * W_half[k * n + i, :])
                    if np.allclose(T_check, 1.0, atol=1e-6):
                        return {
                            "U": U_half.tolist(),
                            "V": V_half.tolist(),
                            "W": W_half.tolist()
                        }
        
        # If we got close but not exact, try to snap and verify
        T_computed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
        
        if np.allclose(T_computed, 1.0, atol=1e-3):
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i, j, k] = np.dot(U_half[i * m + j, :], V_half[j * p + k, :] * W_half[k * n + i, :])
            if np.allclose(T_check, 1.0, atol=1e-6):
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }
    
    return fallback

# EVOLVE-BLOCK-END
