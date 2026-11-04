"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all (i,j,k)
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _find_lower_rank_decomposition(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using numerical methods.
    """
    np.random.seed(123)
    schoolbook_rank = n * m * p
    
    # Try different target ranks
    for target_rank in range(25, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for restart in range(3):
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for iteration in range(50):
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((target_rank, p))
                        for c in range(target_rank):
                            for k in range(p):
                                A[c, k] = V[j * p + k, c] * W[k * n + i, c]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((target_rank, n))
                        for c in range(target_rank):
                            for i in range(n):
                                A[c, i] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((target_rank, m))
                        for c in range(target_rank):
                            for j in range(m):
                                A[c, j] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check if we found a valid decomposition
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Try to find a lower-rank decomposition
    if time.time() < deadline - 5.0:
        result = _find_lower_rank_decomposition(n, m, p, deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = len(result["U"][0])
    
    # 3. Return the best valid decomposition found
    return best
# EVOLVE-BLOCK-END
