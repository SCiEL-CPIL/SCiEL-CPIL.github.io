"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 3, 5
    rank = 41
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Part 1: 3x3x3 using rank-23 Strassen-like decomposition
    # Strassen 2x2x2 terms (8 terms) for the top-left 2x2 block
    # Plus additional terms for the remaining entries
    
    # Strassen terms for 2x2x2 (indices i,j in {0,1}, k in {0,1}):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B10)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # For 3x3x3, we extend this. Use a known rank-23 construction.
    # I'll use a direct construction with 23 terms.
    
    # Terms 0-7: Strassen for 2x2x2 block (i,j in {0,1}, k in {0,1})
    # Term 0: M1
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*5+0][0] = 1; V[1*5+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1; W[0*3+1][0] = -1; W[1*3+0][0] = -1
    
    # Term 1: M2
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*5+0][1] = 1
    W[1*3+0][1] = 1; W[1*3+1][1] = -1
    
    # Term 2: M3
    U[0*3+0][2] = 1
    V[0*5+1][2] = 1; V[1*5+1][2] = -1
    W[0*3+1][2] = 1; W[1*3+1][2] = 1
    
    # Term 3: M4
    U[1*3+1][3] = 1
    V[1*5+0][3] = 1; V[0*5+0][3] = -1
    W[0*3+0][3] = 1; W[1*3+1][3] = -1
    
    # Term 4: M5
    U[0*3+0][4] = 1; U[1*3+0][4] = 1
    V[1*5+1][4] = 1
    W[0*3+0][4] = -1; W[1*3+1][4] = 1
    
    # Term 5: M6
    U[1*3+1][5] = 1; U[0*3+0][5] = -1
    V[0*5+0][5] = 1; V[1*5+0][5] = 1
    W[1*3+1][5] = 1
    
    # Term 6: M7
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*5+0][6] = 1; V[1*5+1][6] = 1
    W[0*3+0][6] = -1; W[1*3+1][6] = 1
    
    # Term 7: M8 (additional for 2x2x2 completeness if needed)
    # Actually 7 terms is standard Strassen. Let me recount.
    # Strassen uses 7 multiplications for 2x2x2.
    # So terms 0-6 are the 7 Strassen terms.
    
    # Now I need terms for the 3rd row and column.
    # This is getting complex. Let me use a simpler verified approach.
    
    # Reset and use a cleaner construction
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Use schoolbook for 3x3x3 (27 terms) + schoolbook for 2 columns (18 terms) = 45
    # But let me try to reduce.
    
    # Actually, let me just use a verified rank-41:
    # 23 terms for 3x3x3 + 9 terms for col 3 + 9 terms for col 4 = 41
    
    # For the 3x3x3 part, use a known rank-23 construction.
    # I'll use a construction based on the following:
    # The 3x3x3 tensor can be written as 23 rank-1 tensors.
    
    # Let me use a simpler approach: just use schoolbook for everything but verify rank 41.
    # Actually, let me use 27+9+9=45 and accept that, or find a better construction.
    
    # Let me try: use rank 23 for 3x3x3 via a known explicit formula.
    # One known rank-23 construction for 3x3x3:
    
    # I'll use the following 23-term construction:
    # Based on the work of Bini et al.
    
    # For safety, let me just return a valid rank-41 using:
    # - 23 terms: a valid 3x3x3 decomposition
    # - 9 terms: schoolbook for column 3  
    # - 9 terms: schoolbook for column 4
    
    # Here's a valid rank-23 construction for 3x3x3:
    # Using the "3x3 Strassen" extension
    
    # Terms 0-6: 7 Strassen terms for 2x2x2 (i,j,k in {0,1})
    # Terms 7-22: 16 additional terms for the 3rd row/col
    
    # Let me implement this carefully.
    
    # Strassen 2x2x2 (7 terms):
    # t0: (a00+a11)(b00+b11) -> c00 += t0, c11 += t0
    # t1: (a10+a11)b00 -> c10 += t1, c11 -= t1
    # t2: a00(b01-b11) -> c01 += t2, c11 += t2
    # t3: a11(b10-b00) -> c00 += t3, c11 -= t3
    # t4: (a00+a10)b11 -> c00 -= t4, c11 += t4
    # t5: (a11-a00)(b00+b10) -> c11 += t5
    # t6: (a01-a11)(b10+b11) -> c00 -= t6, c11 += t6
    
    # Wait, let me use the standard Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a11-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1+m4-m5-m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m7
    
    # Term 0: m1
    U[0][0] = 1; U[4][0] = 1  # a00, a11 (indices: i*m+j, so a00=0, a11=4)
    V[0][0] = 1; V[6][0] = 1  # b00, b11 (j*p+k: b00=0, b11=6)
    W[0][0] = 1; W[7][0] = 1  # c00, c11 (k*n+i: c00=0, c11=7)
    W[1][0] = -1; W[6][0] = -1  # c01? No...
    # c00 is at index 0*3+0=0, c01 at 1*3+0=3, c10 at 0*3+1=1, c11 at 1*3+1=4
    # Wait, W is (p*n) = (3*3) = 9 for 3x3x3, but here p=5 so W is (5*3)=15
    # For the 3x3x3 part, we only use k=0,1,2 so W indices 0-8
    
    # Let me restart with clear indexing:
    # U: (n*m) = 9 rows, indexed by i*m+j where i in 0..2, j in 0..2
    # V: (m*p) = 15 rows, indexed by j*p+k where j in 0..2, k in 0..4
    # W: (p*n) = 15 rows, indexed by k*n+i where k in 0..4, i in 0..2
    
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((15, rank), dtype=int)
    W = np.zeros((15, rank), dtype=int)
    
    # 3x3x3 part (k=0,1,2): 23 terms
    # Using a known rank-23 construction
    
    # I'll use a simple but valid construction: 27 schoolbook terms for 3x3x3
    # This gives total 27+9+9 = 45, same as before.
    
    # Let me try a different approach: use rank 23 from literature.
    # For now, let me just implement a valid rank-41 by using:
    # - A verified 23-term 3x3x3 decomposition
    # - 9 schoolbook terms for column 3
    # - 9 schoolbook terms for column 4
    
    # Here's a rank-23 construction for 3x3x3 (from Bini et al. 1979):
    # I'll encode it explicitly.
    
    # For simplicity and correctness, let me just use the schoolbook (rank 45)
    # but I know rank 41 is achievable. Let me implement rank 41 properly.
    
    # Rank-41 construction:
    # Part A (terms 0-22): 23 terms for 3x3x3
    # Part B (terms 23-31): 9 terms for 3x3x1 (column 3)
    # Part C (terms 32-40): 9 terms for 3x3x1 (column 4)
    
    # Part B: schoolbook for column 3
    for i in range(3):
        for j in range(3):
            t = 23 + i*3 + j
            U[i*3+j][t] = 1
            V[j*5+3][t] = 1
            W[3*3+i][t] = 1  # k=3, so index 3*3+i = 9+i
    
    # Part C: schoolbook for column 4
    for i in range(3):
        for j in range(3):
            t = 32 + i*3 + j
            U[i*3+j][t] = 1
            V[j*5+4][t] = 1
            W[4*3+i][t] = 1  # k=4, so index 4*3+i = 12+i
    
    # Part A: 23 terms for 3x3x3
    # Use a known rank-23 construction
    # I'll use the following explicit construction:
    
    # Based on the standard 3x3x3 rank-23 decomposition:
    # Terms are defined by their (U, V, W) supports
    
    # Let me use a construction from the literature:
    # The 23 terms for 3x3x3 can be constructed as follows:
    
    # I'll use a verified construction. Here are the 23 terms:
    # Each term has U support (subset of {0..8}), V support (subset of {0..8} for k=0,1,2), W support (subset of {0..8} for k=0,1,2)
    
    # For k in {0,1,2}, V index is j*3+k (since p=3 for this part), W index is k*3+i
    
    # Actually, V is indexed by j*p+k with p=5, so for k=0,1,2: V[j*5+k]
    # W is indexed by k*n+i with n=3, so for k=0,1,2: W[k*3+i]
    
    # Let me define the 23 terms:
    
    # Term 0: a00*b00 -> c00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: a00*b10 -> c10  
    U[0][1] = 1; V[5][1] = 1; W[3][1] = 1
    # Term 2: a00*b20 -> c20
    U[0][2] = 1; V[10][2] = 1; W[6][2] = 1
    # Term 3: a01*b00 -> c00
    U[1][3] = 1; V[0][3] = 1; W[1][3] = 1
    # Term 4: a01*b10 -> c10
    U[1][4] = 1; V[5][4] = 1; W[4][4] = 1
    # Term 5: a01*b20 -> c20
    U[1][5] = 1; V[10][5] = 1; W[7][5] = 1
    # Term 6: a02*b00 -> c00
    U[2][6] = 1; V[0][6] = 1; W[2][6] = 1
    # Term 7: a02*b10 -> c10
    U[2][7] = 1; V[5][7] = 1; W[5][7] = 1
    # Term 8: a02*b20 -> c20
    U[2][8] = 1; V[10][8] = 1; W[8][8] = 1
    # Term 9: a10*b01 -> c01
    U[3][9] = 1; V[1][9] = 1; W[0][9] = 1
    # Term 10: a10*b11 -> c11
    U[3][10] = 1; V[6][10] = 1; W[3][10] = 1
    # Term 11: a10*b21 -> c21
    U[3][11] = 1; V[11][11] = 1; W[6][11] = 1
    # Term 12: a11*b01 -> c01
    U[4][12] = 1; V[1][12] = 1; W[1][12] = 1
    # Term 13: a11*b11 -> c11
    U[4][13] = 1; V[6][13] = 1; W[4][13] = 1
    # Term 14: a11*b21 -> c21
    U[4][14] = 1; V[11][14] = 1; W[7][14] = 1
    # Term 15: a12*b01 -> c01
    U[5][15] = 1; V[1][15] = 1; W[2][15] = 1
    # Term 16: a12*b11 -> c11
    U[5][16] = 1; V[6][16] = 1; W[5][16] = 1
    # Term 17: a12*b21 -> c21
    U[5][17] = 1; V[11][17] = 1; W[8][17] = 1
    # Term 18: a20*b02 -> c02
    U[6][18] = 1; V[2][18] = 1; W[0][18] = 1
    # Term 19: a20*b12 -> c12
    U[6][19] = 1; V[7][19] = 1; W[3][19] = 1
    # Term 20: a20*b22 -> c22
    U[6][20] = 1; V[12][20] = 1; W[6][20] = 1
    # Term 21: a21*b02 -> c02
    U[7][21] = 1; V[2][21] = 1; W[1][21] = 1
    # Term 22: a21*b12 -> c12
    U[7][22] = 1; V[7][22] = 1; W[4][22] = 1
    
    # Wait, that's only 23 terms but they're all schoolbook (rank 27 would be full schoolbook).
    # I'm missing 4 terms: a22*b02, a22*b12, a22*b22, and... let me count.
    # Full schoolbook for 3x3x3 is 27 terms. I have 23, so I'm missing 4.
    # This won't work - I need all 27 or use a proper rank-23 construction.
    
    # Let me just use full schoolbook: rank 45.
    # Or better, let me implement a proper rank-23 construction.
    
    # For now, return the schoolbook rank-45 as a valid fallback.
    U = np.zeros((9, 45), dtype=int)
    V = np.zeros((15, 45), dtype=int)
    W = np.zeros((15, 45), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U[i*3+j][col] = 1
                V[j*5+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
