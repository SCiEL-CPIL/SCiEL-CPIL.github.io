import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Try rank 20 first
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 50, 60, 80]:
        if time.time() > deadline - 20:
            break
        
        result = _find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank, deadline):
    """Find a rank decomposition using iterative optimization."""
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Use alternating minimization
    # Fix V and W, solve for U (linear)
    # Fix U and W, solve for V (linear)
    # Fix U and V, solve for W (linear)
    
    random.seed(123)
    np.random.seed(123)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            return None
        
        # Initialize
        U = np.random.randint(-2, 3, size=(nm, rank)).astype(float)
        V = np.random.randint(-2, 3, size=(mp, rank)).astype(float)
        W = np.random.randint(-2, 3, size=(pn, rank)).astype(float)
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: fix V, W
            # For each (i,j), we need: sum_k sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is: sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
            # Let A_ijc = sum_k V[j*p+k, c] * W[k*n+i, c]
            # Then: sum_c U[i*m+j, c] * A_ijc = 1
            # This is a linear system in U[i*m+j, :]
            
            U_new = np.zeros((nm, rank))
            for i in range(n):
                for j in range(m):
                    # A[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros(rank)
                    for k in range(p):
                        A += V[j * p + k, :] * W[k * n + i, :]
                    # Solve: U[i*m+j, :] @ A = 1
                    # This is underdetermined if rank > 1, use least squares
                    if np.linalg.norm(A) < 1e-10:
                        U_new[i * m + j, :] = 0
                    else:
                        # Least squares solution
                        U_new[i * m + j, :] = np.linalg.lstsq(A.reshape(1, -1), np.array([1.0]), rcond=None)[0]
            
            # Solve for V: fix U, W
            V_new = np.zeros((mp, rank))
            for j in range(m):
                for k in range(p):
                    # B[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
                    B = np.zeros(rank)
                    for i in range(n):
                        B += U[i * m + j, :] * W[k * n + i, :]
                    if np.linalg.norm(B) < 1e-10:
                        V_new[j * p + k, :] = 0
                    else:
                        V_new[j * p + k, :] = np.linalg.lstsq(B.reshape(1, -1), np.array([1.0]), rcond=None)[0]
            
            # Solve for W: fix U, V
            W_new = np.zeros((pn, rank))
            for k in range(p):
                for i in range(n):
                    # C[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                    C = np.zeros(rank)
                    for j in range(m):
                        C += U[i * m + j, :] * V[j * p + k, :]
                    if np.linalg.norm(C) < 1e-10:
                        W_new[k * n + i, :] = 0
                    else:
                        W_new[k * n + i, :] = np.linalg.lstsq(C.reshape(1, -1), np.array([1.0]), rcond=None)[0]
            
            U, V, W = U_new, V_new, W_new
        
        # Verify
        if _verify_numpy(n, m, p, U, V, W):
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def _verify_numpy(n, m, p, U, V, W):
    """Verify decomposition using numpy."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True
# EVOLVE-BLOCK-END
