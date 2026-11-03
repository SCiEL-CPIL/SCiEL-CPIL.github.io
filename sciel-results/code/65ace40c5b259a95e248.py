"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks.
Note: Rank 28 is believed to be optimal for 2x3x5 matrix multiplication.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 150.0 # 150s wall clock, use 145s to be safe

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy: Use Strassen's 7-term decomposition for two disjoint 2x2x2 blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    The remaining 14 terms are handled by schoolbook multiplication.
    Total rank: 7 + 7 + 14 = 28.
    """
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return None
        
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    # Helper to add a Strassen block
    # j_offset, k_offset
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1 = (a00+a11)*(b00+b11), contributes to c00(+1), c11(+1)
        # Indices:
        # U: i=0,j=j_off; i=1,j=j_off+1
        # V: j=j_off,k=k_off; j=j_off+1,k=k_off+1
        # W: k=k_off,i=0; k=k_off+1,i=1
        
        U[0*m + (j_off)][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + (k_off)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m2 = (a10+a11)*b00, contributes to c10(+1), c11(-1)
        # Note: Strassen standard:
        # m2 = (A[1][0] + A[1][1]) * B[0][0]
        # C[1][0] += m2
        # C[1][1] -= m2
        U[1*m + (j_off)][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[(j_off)*p + (k_off)][c] = 1
        W[(k_off)*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # m3 = a00*(b01-b11), contributes to c01(+1), c11(+1)
        # m3 = A[0][0] * (B[0][1] - B[1][1])
        # C[0][1] += m3
        # C[1][1] += m3
        U[0*m + (j_off)][c] = 1
        V[(j_off)*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m4 = a11*(b10-b00), contributes to c00(+1), c10(+1)
        # m4 = A[1][1] * (B[1][0] - B[0][0])
        # C[0][0] += m4
        # C[1][0] += m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off)][c] = 1; V[(j_off)*p + (k_off)][c] = -1
        W[(k_off)*n + 0][c] = 1; W[(k_off)*n + 1][c] = 1
        c += 1
        
        # m5 = (a00+a01)*b11, contributes to c00(-1), c01(+1)
        # m5 = (A[0][0] + A[0][1]) * B[1][1]
        # C[0][0] -= m5
        # C[0][1] += m5
        U[0*m + (j_off)][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        # m6 = (a10-a00)*(b00+b01), contributes to c11(+1)
        # m6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
        # C[1][1] += m6
        U[1*m + (j_off)][c] = 1; U[0*m + (j_off)][c] = -1
        V[(j_off)*p + (k_off)][c] = 1; V[(j_off)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m7 = (a01-a11)*(b10+b11), contributes to c00(+1)
        # m7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
        # C[0][0] += m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + (k_off)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off)*n + 0][c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Remaining 14 schoolbook terms
    # Total 30 terms.
    # Block 1 covers j in {0,1}, k in {0,1} (8 terms)
    # Block 2 covers j in {0,1}, k in {2,3} (8 terms)
    # Remaining:
    # j in {0,1}, k in {4} -> 2*2 = 4 terms
    # j in {2}, k in {0,1,2,3,4} -> 2*5 = 10 terms
    # Total 14.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip terms covered by Strassen blocks
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
