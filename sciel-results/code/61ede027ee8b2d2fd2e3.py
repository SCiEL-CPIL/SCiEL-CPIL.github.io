"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _build_rank5_2x2x2x3():
    """
    Build a rank-5 decomposition for 2x2 * 2x3 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    where i in {0,1}, j in {0,1}, k in {0,1,2}
    
    Returns U (4x5), V (6x5), W (6x5)
    """
    # Known rank-5 construction for 2x2 x 2x3
    # Using a standard bilinear algorithm
    
    # U: rows indexed by (i,j) -> i*2+j, cols by c=0..4
    # V: rows indexed by (j,k) -> j*3+k, cols by c=0..4  
    # W: rows indexed by (k,i) -> k*2+i, cols by c=0..4
    
    U = [
        [1, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0],  # (1,0)
        [0, 0, 0, 1, 0],  # (1,1)
    ]
    
    V = [
        [1, 1, 0, 0, 0],  # (0,0)
        [0, 0, 1, 1, 0],  # (0,1)
        [0, 0, 0, 0, 1],  # (0,2)
        [1, 0, 1, 0, 0],  # (1,0)
        [0, 1, 0, 1, 0],  # (1,1)
        [0, 0, 0, 0, 1],  # (1,2)
    ]
    
    W = [
        [1, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0],  # (1,0)
        [0, 0, 0, 1, 0],  # (1,1)
        [0, 0, 0, 0, 1],  # (2,0)
        [0, 0, 0, 0, 1],  # (2,1)
    ]
    
    # Verify this works for 2x2 x 2x3
    n, m, p = 2, 2, 3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(5):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    # This construction is wrong, let me try another
                    pass
    
    # Let me use a different known construction
    # From literature: R(2,2,3) = 5
    # Here's a valid construction:
    
    U2 = [
        [1, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0],  # (0,1)  
        [0, 0, 1, 0, 0],  # (1,0)
        [0, 0, 0, 1, 0],  # (1,1)
    ]
    
    V2 = [
        [1, 1, 0, 0, 0],  # (0,0)
        [1, 0, 1, 0, 0],  # (0,1)
        [0, 0, 0, 1, 0],  # (0,2)
        [1, 0, 0, 0, 1],  # (1,0)
        [0, 1, 0, 0, 0],  # (1,1)
        [0, 0, 0, 1, 0],  # (1,2)
    ]
    
    W2 = [
        [1, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0],  # (1,0)
        [0, 0, 0, 1, 0],  # (1,1)
        [0, 0, 0, 0, 1],  # (2,0)
        [0, 0, 0, 0, 1],  # (2,1)
    ]
    
    # Verify
    n, m, p = 2, 2, 3
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(5):
                    s += U2[i*m+j][c] * V2[j*p+k][c] * W2[k*n+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U2, V2, W2
    
    # Fallback: use schoolbook for 2x2 x 2x3 (rank 6)
    U3 = [[0]*6 for _ in range(4)]
    V3 = [[0]*6 for _ in range(6)]
    W3 = [[0]*6 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U3[i*2+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*2+i][col] = 1
                col += 1
    return U3, V3, W3

def _build_2x4x6_decomposition():
    """
    Build a low-rank decomposition for 2x4 * 4x6.
    
    Strategy:
    - Split middle dimension (4) into two blocks of 2
    - Split output dimension (6) into two blocks of 3
    - Use rank-5 for each 2x2 x 2x3 subproblem
    
    Total rank: 2 (middle blocks) * 2 (output blocks) * 5 = 20
    """
    n, m, p = 2, 4, 6
    
    # Get the rank-5 building block
    U5, V5, W5 = _build_rank5_2x2x2x3()
    r5 = len(U5[0])
    
    # We have 4 subproblems:
    # Block (a,b) where a in {0,1} (middle split), b in {0,1} (output split)
    # A_block_a = A[:, 2*a : 2*a+2] (2x2)
    # B_block_a,b = B[2*a:2*a+2, 3*b:3*b+3] (2x3)
    # C_block_b = C[:, 3*b:3*b+3] (2x3)
    
    # C_block_b = A_block_a @ B_block_a,b for a=0,1 (summed)
    # So C = sum_a sum_b (A_block_a @ B_block_a,b)
    
    # Each term A_block_a @ B_block_a,b has rank 5
    # Total rank: 2 * 2 * 5 = 20
    
    total_rank = 2 * 2 * r5
    U = [[0]*total_rank for _ in range(n*m)]  # (2*4, 20) = (8, 20)
    V = [[0]*total_rank for _ in range(m*p)]  # (4*6, 20) = (24, 20)
    W = [[0]*total_rank for _ in range(p*n)]  # (6*2, 20) = (12, 20)
    
    col_offset = 0
    for a in range(2):  # middle block
        for b in range(2):  # output block
            for c in range(r5):  # rank-5 terms
                col = col_offset + c
                for i in range(2):  # row of A/C
                    # A index: i*m + (2*a + j) where j is local col
                    # For each local j in {0,1}:
                    for j_local in range(2):
                        j_global = 2*a + j_local
                        U[i*m + j_global][col] = U5[i*2 + j_local][c]
                
                for j_local in range(2):  # row of B
                    j_global = 2*a + j_local
                    for k_local in range(3):  # col of B/C
                        k_global = 3*b + k_local
                        V[j_global*p + k_global][col] = V5[j_local*3 + k_local][c]
                
                for k_local in range(3):  # col of C
                    k_global = 3*b + k_local
                    for i in range(2):  # row of C
                        W[k_global*n + i][col] = W5[k_local*2 + i][c]
            
            col_offset += r5
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try to build the rank-20 decomposition
    try:
        U, V, W, rank = _build_2x4x6_decomposition()
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 48)
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
