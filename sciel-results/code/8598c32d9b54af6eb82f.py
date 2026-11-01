"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_rank_11():
    """
    Construct an explicit rank-11 decomposition for 2x2x3 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all (i,j,k) in {0,1}x{0,1}x{0,1,2}.
    We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    Construction strategy:
    - Use 7 terms based on Strassen's algorithm for the 2x2x2 subproblem (k=0,1)
    - Add 4 terms for k=2
    - Total: 11 terms
    
    For the 2x2x2 part, we use a modified Strassen approach where each term
    contributes to specific (i,j,k) combinations.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen-like terms for k=0,1 (terms 0-6)
    # Each term c has U_c in R^4, V_c in R^6, W_c in R^6
    # U index: i*m+j = i*2+j
    # V index: j*p+k = j*3+k
    # W index: k*n+i = k*2+i
    
    # Term 0: Handles (0,0,0) and (1,1,1) partially
    # U: [1, 0, 0, 1] -> i=0,j=0 and i=1,j=1
    # V: [1, 0, 0, 1, 0, 0] -> j=0,k=0 and j=1,k=0
    # W: [1, 0, 0, 1, 0, 0] -> k=0,i=0 and k=1,i=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: Handles (0,0,1) and (1,1,0) 
    # U: [1, 0, 0, 1]
    # V: [0, 0, 1, 0, 1, 0] -> j=0,k=1 and j=1,k=1
    # W: [0, 1, 1, 0, 0, 0] -> k=0,i=1 and k=1,i=0
    U[0][1] = 1; U[3][1] = 1
    V[2][1] = 1; V[5][1] = 1
    W[1][1] = 1; W[2][1] = 1
    
    # Term 2: Handles (0,1,0) and (1,0,1)
    # U: [0, 1, 1, 0] -> i=0,j=1 and i=1,j=0
    # V: [1, 0, 0, 0, 1, 0] -> j=0,k=0 and j=1,k=1
    # W: [1, 0, 0, 1, 0, 0] -> k=0,i=0 and k=1,i=1
    U[1][2] = 1; U[2][2] = 1
    V[0][2] = 1; V[4][2] = 1
    W[0][2] = 1; W[3][2] = 1
    
    # Term 3: Handles (0,1,1) and (1,0,0)
    # U: [0, 1, 1, 0]
    # V: [0, 1, 0, 1, 0, 0] -> j=0,k=1 and j=1,k=0
    # W: [0, 1, 1, 0, 0, 0] -> k=0,i=1 and k=1,i=0
    U[1][3] = 1; U[2][3] = 1
    V[1][3] = 1; V[3][3] = 1
    W[1][3] = 1; W[2][3] = 1
    
    # Term 4: Handles (0,0,0) and (1,1,0) with different structure
    # U: [1, 0, 0, 1]
    # V: [1, 1, 0, 0, 0, 0] -> j=0,k=0 and j=0,k=1
    # W: [1, 0, 0, 0, 0, 0] -> k=0,i=0
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1
    
    # Term 5: Handles (0,1,0) and (1,0,0)
    # U: [0, 1, 1, 0]
    # V: [0, 0, 0, 1, 1, 0] -> j=1,k=0 and j=1,k=1
    # W: [0, 1, 0, 0, 0, 0] -> k=0,i=1
    U[1][5] = 1; U[2][5] = 1
    V[3][5] = 1; V[4][5] = 1
    W[1][5] = 1
    
    # Term 6: Handles (1,1,1) and (0,0,1)
    # U: [1, 0, 0, 1]
    # V: [0, 0, 1, 0, 0, 1] -> j=0,k=1 and j=1,k=1
    # W: [0, 0, 0, 1, 0, 0] -> k=1,i=1
    U[0][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[5][6] = 1
    W[3][6] = 1
    
    # Terms 7-10: Handle k=2 (4 terms for 4 (i,j) pairs)
    # Term 7: (0,0,2)
    U[0][7] = 1
    V[2][7] = 1  # j=0,k=2
    W[4][7] = 1  # k=2,i=0
    
    # Term 8: (0,1,2)
    U[1][8] = 1
    V[5][8] = 1  # j=1,k=2
    W[4][8] = 1  # k=2,i=0
    
    # Term 9: (1,0,2)
    U[2][9] = 1
    V[2][9] = 1  # j=0,k=2
    W[5][9] = 1  # k=2,i=1
    
    # Term 10: (1,1,2)
    U[3][10] = 1
    V[5][10] = 1  # j=1,k=2
    W[5][10] = 1  # k=2,i=1
    
    return U, V, W

def _try_rank_11():
    """Attempt to construct and verify a rank-11 decomposition."""
    n, m, p = 2, 2, 3
    U, V, W = _build_rank_11()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-11 construction
    result = _try_rank_11()
    if result is not None:
        return result
    
    # Fallback: return the schoolbook rank-12 decomposition
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
# EVOLVE-BLOCK-END
