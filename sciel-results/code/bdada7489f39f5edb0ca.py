"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Reconstruct: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # U indices: i*m+j where i=0,1 and j=0,1
    # V indices: j*p+k where j=0,1 and k=0,1
    # W indices: k*n+i where k=0,1 and i=0,1
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1.0  # a11
    U[3][0] = 1.0  # a22
    V[0][0] = 1.0  # b11
    V[3][0] = 1.0  # b22
    W[0][0] = 1.0  # c11 (k=0,i=0)
    W[3][0] = 1.0  # c22 (k=1,i=1)
    
    # M2 = (a21+a22)b11
    U[2][1] = 1.0  # a21
    U[3][1] = 1.0  # a22
    V[0][1] = 1.0  # b11
    W[2][1] = 1.0  # c21 (k=0,i=1)
    
    # M3 = a11(b12-b22)
    U[0][2] = 1.0  # a11
    V[1][2] = 1.0  # b12
    V[3][2] = -1.0 # b22
    W[1][2] = 1.0  # c12 (k=1,i=0)
    
    # M4 = a22(b21-b11)
    U[3][3] = 1.0  # a22
    V[2][3] = 1.0  # b21
    V[0][3] = -1.0 # b11
    W[0][3] = 1.0  # c11 (k=0,i=0)
    W[2][3] = 1.0  # c21 (k=0,i=1)
    
    # M5 = (a11+a12)b22
    U[0][4] = 1.0  # a11
    U[1][4] = 1.0  # a12
    V[3][4] = 1.0  # b22
    W[0][4] = -1.0 # c11 (k=0,i=0)
    W[1][4] = 1.0  # c12 (k=1,i=0)
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1.0  # a21
    U[0][5] = -1.0 # a11
    V[0][5] = 1.0  # b11
    V[1][5] = 1.0  # b12
    W[3][5] = 1.0  # c22 (k=1,i=1)
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1.0  # a12
    U[3][6] = -1.0 # a22
    V[2][6] = 1.0  # b21
    V[3][6] = 1.0  # b22
    W[0][6] = 1.0  # c11 (k=0,i=0)
    W[3][6] = 1.0  # c22 (k=1,i=1)
    
    return U, V, W

def _combine_decompositions(decs):
    """Combine multiple decompositions by concatenation."""
    total_rank = sum(len(d[0][0]) for d in decs)
    n = m = p = 2
    U = [[0.0] * total_rank for _ in range(n*m)]
    V = [[0.0] * total_rank for _ in range(m*p)]
    W = [[0.0] * total_rank for _ in range(p*n)]
    
    offset = 0
    for U_d, V_d, W_d in decs:
        r = len(U_d[0])
        for i in range(n*m):
            for rr in range(r):
                U[i][offset+rr] += U_d[i][rr]
        for i in range(m*p):
            for rr in range(r):
                V[i][offset+rr] += V_d[i][rr]
        for i in range(p*n):
            for rr in range(r):
                W[i][offset+rr] += W_d[i][rr]
        offset += r
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using Strassen blocks."""
    n, m, p = 2, 4, 7
    
    # Strategy: Split middle dimension (4) into two halves of 2.
    # A = [A1 A2] (each 2x2), B = [B1; B2] (each 2x7)
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is M_{2,2,7}
    # For M_{2,2,7}: split 7 cols into 2+2+2+1
    #   Three M_{2,2,2} blocks (Strassen, rank 7 each)
    #   One M_{2,2,1} block (schoolbook, rank 2)
    # Total per M_{2,2,7}: 3*7 + 4 = 25
    # Grand total: 2 * 25 = 50
    
    U_strassen, V_strassen, W_strassen = _build_strassen_2x2x2()
    
    total_rank = 50
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # For each of the two halves (j_offset = 0, 2)
    for j_offset in [0, 2]:
        # For each column block in the 7 outputs: cols 0-1, 2-3, 4-5, 6
        # Blocks: (k_start, k_end) for k in 0..6
        col_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for k_start, k_end in col_blocks:
            k_width = k_end - k_start
            
            if k_width == 2:
                # Use Strassen for M_{2,2,2}
                r = 7
                for r_idx in range(r):
                    c = col_offset + r_idx
                    # U: index i*m + (j_offset + j), for i in 0,1 and j in 0,1
                    for i in range(2):
                        for j in range(2):
                            U[i * m + (j_offset + j)][c] = U_strassen[i * 2 + j][r_idx]
                    # V: index (j_offset + j)*p + (k_start + k), for j in 0,1 and k in 0,1
                    for j in range(2):
                        for k in range(2):
                            V[(j_offset + j) * p + (k_start + k)][c] = V_strassen[j * 2 + k][r_idx]
                    # W: index (k_start + k)*n + i, for k in 0,1 and i in 0,1
                    for k in range(2):
                        for i in range(2):
                            W[(k_start + k) * n + i][c] = W_strassen[k * 2 + i][r_idx]
                col_offset += r
            else:
                # k_width == 1: schoolbook for M_{2,2,1}, rank 2
                r = 2
                for r_idx in range(r):
                    c = col_offset + r_idx
                    # Schoolbook: C[i,k] = sum_j A[i,j]*B[j,k]
                    # For M_{2,2,1}: two terms
                    # Term 0: A[0,0]*B[0,0] -> C[0,0]
                    # Term 1: A[1,0]*B[0,0] -> C[1,0]
                    # Wait, M_{2,2,1} means 2x2 times 2x1 = 2x1
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                    # Schoolbook rank = 2*2*1 = 4? No...
                    # Actually for M_{n,m,p}, schoolbook rank = n*m*p
                    # M_{2,2,1}: rank = 2*2*1 = 4
                    # But we can do better: 
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                    # We need 4 products in schoolbook.
                    # But there's no known improvement for M_{2,2,1} below 4.
                    # Actually, let me reconsider. The tensor is:
                    # T[i,j,k] where i in {0,1}, j in {0,1}, k in {0}
                    # T[i,j,0] = 1 for all i,j
                    # This is just the outer product of (1,1,1,1) with itself... 
                    # No, it's a 2x2x1 tensor with all ones.
                    # Rank of all-ones 2x2x1 tensor: 
                    # T[i,j,0] = 1 = sum_r U[i,j,r]*V[j,0,r]*W[0,i,r]
                    # We can do this with rank 2:
                    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1; U[1,0]=0, V[1,0]=0, W[0,1]=0
                    # r=1: U[0,1]=0, V[0,1]=0, W[0,0]=0; U[1,1]=1, V[1,1]=1, W[0,1]=1
                    # Check: T[0,0,0] = 1*1*1 + 0*0*0 = 1 ✓
                    # T[0,1,0] = 0*0*1 + 0*0*0 = 0 ✗ (should be 1)
                    
                    # Let me think again. T[i,j,k] = 1 for all i,j,k.
                    # For M_{2,2,1}: i in {0,1}, j in {0,1}, k in {0}
                    # We need: 1 = sum_r U[i,j,r] * V[j,0,r] * W[0,i,r]
                    # for all i,j.
                    # This is equivalent to: the 2x2 matrix of all ones has rank 1,
                    # but we have the constraint that it factors as sum_r u_r * v_r^T * w_r^T
                    # where u_r is 2x2, v_r is 2x1, w_r is 1x2.
                    # Actually, let's just use schoolbook rank 4 for safety.
                    
                    pass
                
                # Use schoolbook for M_{2,2,1}: rank 4
                r = 4
                for i in range(2):
                    for j in range(2):
                        c = col_offset + i * 2 + j
                        U[i * m + (j_offset + j)][c] = 1.0
                        V[(j_offset + j) * p + k_start][c] = 1.0
                        W[k_start * n + i][c] = 1.0
                col_offset += r
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find a low-rank decomposition
    U, V, W = _build_2x4x7_decomposition()
    
    rank = len(U[0])
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
    
    # Convert to lists if needed
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
