"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False, (i, j, k, total)
    return True, None


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Strategy: Use 3x3x3 schoolbook core (rank 27) + boundary terms
    # Core: indices i,j,k in {0,1,2}
    # Boundary:
    #   i in {0,1,2}, k in {3,4}: 3*2=6 terms, each sum over j in {0..4}: 6*5=30
    #   i in {3,4}, k in {0,1,2}: 2*3=6 terms, each sum over j in {0..4}: 6*5=30
    #   i in {3,4}, k in {3,4}: 2*2=4 terms, each sum over j in {0..4}: 4*5=20
    # Total: 27 + 30 + 30 + 20 = 107
    
    total_rank = 107
    U5 = [[0] * total_rank for _ in range(25)]
    V5 = [[0] * total_rank for _ in range(25)]
    W5 = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # 1. 3x3x3 schoolbook core
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U5[i * 5 + j][col] = 1
                V5[j * 5 + k][col] = 1
                W5[k * 5 + i][col] = 1
                col += 1
    
    # 2. i in {0,1,2}, k in {3,4}: C[i,k] = sum_j A[i,j]*B[j,k]
    for i in range(3):
        for k in [3, 4]:
            for j in range(5):
                U5[i * 5 + j][col] = 1
                V5[j * 5 + k][col] = 1
                W5[k * 5 + i][col] = 1
                col += 1
    
    # 3. i in {3,4}, k in {0,1,2}: C[i,k] = sum_j A[i,j]*B[j,k]
    for i in [3, 4]:
        for k in range(3):
            for j in range(5):
                U5[i * 5 + j][col] = 1
                V5[j * 5 + k][col] = 1
                W5[k * 5 + i][col] = 1
                col += 1
    
    # 4. i in {3,4}, k in {3,4}: C[i,k] = sum_j A[i,j]*B[j,k]
    for i in [3, 4]:
        for k in [3, 4]:
            for j in range(5):
                U5[i * 5 + j][col] = 1
                V5[j * 5 + k][col] = 1
                W5[k * 5 + i][col] = 1
                col += 1
    
    # Verify
    ok, info = _verify_decomposition(U5, V5, W5, 5, 5, 5)
    
    if not ok:
        # Fallback to schoolbook
        rank = 125
        U5 = [[0] * rank for _ in range(25)]
        V5 = [[0] * rank for _ in range(25)]
        W5 = [[0] * rank for _ in range(25)]
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U5[i * 5 + j][col] = 1
                    V5[j * 5 + k][col] = 1
                    W5[k * 5 + i][col] = 1
                    col += 1
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END
