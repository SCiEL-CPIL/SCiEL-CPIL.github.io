"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    
    Uses 5 independent Strassen blocks (rank 7 each).
    The B matrix (2x10) is split into 5 blocks of size 2x2.
    Each block is multiplied by the 2x2 A matrix using Strassen's algorithm.
    
    A is 2x2: indices (i,j) where i in {0,1}, j in {0,1}
    B is 2x10: indices (j,k) where j in {0,1}, k in {0,...,9}
    C is 2x10: indices (i,k) where i in {0,1}, k in {0,...,9}
    
    The output factor W follows the trace convention W[k*n+i, r]
    where n=2, so W has shape (10*2, rank) = (20, rank).
    
    For block t (0 <= t < 5):
      B block columns: 2t, 2t+1
      C block columns: 2t, 2t+1
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # Strassen's 7 multiplications for 2x2 matrices:
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    
    for t in range(5):
        col_base = t * 7
        
        # V indices for block t:
        # B[j, 2t]   -> V index 10*j + 2t
        # B[j, 2t+1] -> V index 10*j + 2t + 1
        # So: e = B[0,2t] = V[2t], f = B[0,2t+1] = V[2t+1]
        #     g = B[1,2t] = V[10+2t], h = B[1,2t+1] = V[11+2t]
        
        # W indices for block t (trace convention W[k*n+i, r]):
        # C[i, 2t]   -> W index 2t*n + i = 4t + i
        # C[i, 2t+1] -> W index (2t+1)*n + i = 4t+2+i
        # So: C[0,2t] = W[4t], C[1,2t] = W[4t+1]
        #     C[0,2t+1] = W[4t+2], C[1,2t+1] = W[4t+3]
        
        # U indices: A[i,j] -> U index i*2+j
        # a = A[0,0] = U[0], b = A[0,1] = U[1]
        # c = A[1,0] = U[2], d = A[1,1] = U[3]
        
        # P1 = (a+d)(e+h)
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # P2 = (c+d)e
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # P3 = a(f-h)
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # P4 = d(g-e)
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # P5 = (a+b)h
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # P6 = (c-a)(e+f)
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # P7 = (b-d)(g+h)
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
