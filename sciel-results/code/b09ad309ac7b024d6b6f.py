import time
import random
import numpy as np

# Keep the original dimensions
DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    For each i, j, k, the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U[i*2+j, r] corresponds to A[i,j]
    # V[j*2+k, r] corresponds to B[j,k]
    # W[k*2+i, r] corresponds to C[i,k] (note: W[k*n+i] with n=2, so W[k*2+i])
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    # a=A[0,0]=U[0], d=A[1,1]=U[3]
    # e=B[0,0]=V[0], h=B[1,1]=V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    # C11 += m1, C12 += m1? No: C11 = m1+m4-m5+m7, C12 = m3+m5, C21 = m2+m4, C22 = m1-m2+m3+m6
    # W[k*2+i, r]: W[0][0]=C[0,0], W[1][0]=C[0,1], W[2][0]=C[1,0], W[3][0]=C[1,1]
    # So for m1: W[0][0]=1, W[3][0]=1 (C11 and C22)
    W[0][0] = 1
    W[3][0] = 1
    
    # m2 = (c+d)e
    # c=A[1,0]=U[2], d=A[1,1]=U[3]
    # e=B[0,0]=V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 += m2
    W[3][1] = -1  # C22 -= m2
    
    # m3 = a(f-h)
    # a=A[0,0]=U[0]
    # f=B[0,1]=V[1], h=B[1,1]=V[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12 += m3
    W[3][2] = 1  # C22 += m3
    
    # m4 = d(g-e)
    # d=A[1,1]=U[3]
    # g=B[1,0]=V[2], e=B[0,0]=V[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11 += m4
    W[2][3] = 1  # C21 += m4
    
    # m5 = (a+b)h
    # a=A[0,0]=U[0], b=A[0,1]=U[1]
    # h=B[1,1]=V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11 -= m5
    W[1][4] = 1  # C12 += m5
    
    # m6 = (c-a)(e+f)
    # c=A[1,0]=U[2], a=A[0,0]=U[0]
    # e=B[0,0]=V[0], f=B[0,1]=V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += m6
    
    # m7 = (b-d)(g+h)
    # b=A[0,1]=U[1], d=A[1,1]=U[3]
    # g=B[1,0]=V[2], h=B[1,1]=V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += m7
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    For each i, j, k, the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # leave 5s margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # --- Baseline: Schoolbook decomposition (rank 96) ---
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    # Check baseline
    if not _verify(U_school, V_school, W_school, n, m, p):
        pass
    
    if time.time() > DEADLINE:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    # --- Strassen-based decomposition ---
    # Strategy: Split the 3x4x8 problem into blocks.
    # For rows i=0,1 (top 2 rows of A), use Strassen on 2x2x2 sub-blocks.
    # For row i=2 (bottom row of A), use schoolbook.
    
    # Let's think about this more carefully.
    # A is 3x4, B is 4x8, C is 3x8.
    # We can split A into A_top (2x4) and A_bottom (1x4).
    # C_top (2x8) = A_top * B
    # C_bottom (1x8) = A_bottom * B
    
    # For C_top (2x8 = A_top(2x4) * B(4x8)):
    # Split B into B_left (4x4) and B_right (4x4).
    # C_top_left (2x4) = A_top * B_left
    # C_top_right (2x4) = A_top * B_right
    
    # For each 2x4x4 multiplication (2x4 matrix times 4x4 matrix):
    # Split A_top into A1 (2x2) and A2 (2x2) [left and right halves]
    # Split B into B1 (2x4) and B2 (2x4) [top and bottom halves]
    # Result = A1*B1 + A2*B2
    # Each A_i (2x2) * B_i (2x4) can be further split.
    
    # Actually, let's use a cleaner block decomposition.
    # View the 4x8 matrix B as a 2x4 block matrix of 2x2 blocks:
    # B = [[B11, B12, B13, B14],
    #      [B21, B22, B23, B24]]
    # where each Bij is 2x2.
    
    # A is 3x4. Split A into:
    # A = [[A11, A12, A13, A14],   (row 0, 1x4)
    #      [A21, A22, A23, A24],   (row 1, 1x4)
    #      [A31, A32, A33, A34]]   (row 2, 1x4)
    
    # This doesn't directly give 2x2 blocks for A.
    
    # Alternative approach: 
    # For i in {0,1} (top 2 rows), treat the 2x4 part of A as two 2x2 blocks.
    # A_top = [[A11, A12], [A21, A22]] where A11,A12,A21,A22 are 1x2 blocks.
    # This is not 2x2x2.
    
    # Let me try a different decomposition:
    # Split the 4-dimensional middle index into two 2-dimensional parts.
    # A (3x4) -> A1 (3x2), A2 (3x2)
    # B (4x8) -> B1 (2x8), B2 (2x8)
    # C = A1*B1 + A2*B2
    
    # For A1 (3x2) * B1 (2x8):
    # Split A1 into A1_top (2x2) and A1_bottom (1x2)
    # C1_top (2x8) = A1_top * B1
    # C1_bottom (1x8) = A1_bottom * B1
    
    # For A1_top (2x2) * B1 (2x8):
    # Split B1 into B11 (2x4) and B12 (2x4)
    # C1_top_left (2x4) = A1_top * B11
    # C1_top_right (2x4) = A1_top * B12
    
    # For A1_top (2x2) * B11 (2x4):
    # Split B11 into B111 (2x2) and B112 (2x2)
    # C1_top_left_left (2x2) = A1_top * B111
    # C1_top_left_right (2x2) = A1_top * B112
    
    # Now A1_top (2x2) * B111 (2x2) is a 2x2x2 multiplication -> use Strassen (rank 7)
    # Similarly for other 2x2x2 blocks.
    
    # Let me count the number of 2x2x2 blocks and 1x2x2 or similar blocks.
    
    # Actually, let me try a simpler approach based on the insight:
    # Use Strassen for 2x2x2 sub-blocks where possible, schoolbook otherwise.
    
    # Decomposition:
    # Split m=4 into m1=2, m2=2
    # Split n=3 into n1=2, n2=1
    # Split p=8 into p1=4, p2=4 (or further)
    
    # For the (n1 x m1) x (m1 x p1) = (2x2) x (2x4) part:
    # Split p1=4 into p1a=2, p1b=2
    # (2x2) x (2x2) -> Strassen rank 7
    # (2x2) x (2x2) -> Strassen rank 7
    # Total for this part: 14
    
    # For the (n1 x m2) x (m2 x p1) = (2x2) x (2x4) part:
    # Similarly: 7 + 7 = 14
    
    # For the (n2 x m1) x (m1 x p1) = (1x2) x (2x4) part:
    # (1x2) x (2x2) -> rank 2 (schoolbook)
    # (1x2) x (2x2) -> rank 2
    # Total: 4
    
    # For the (n2 x m2) x (m2 x p1) = (1x2) x (2x4) part:
    # Similarly: 2 + 2 = 4
    
    # So for p1=4: 14 + 14 + 4 + 4 = 36
    
    # Similarly for p2=4: 36
    
    # Total: 36 + 36 = 72? That seems too optimistic.
    
    # Wait, I need to be more careful. Let me re-examine.
    
    # The tensor M_{3,4,8} has entries T[i,j,k] for i in 0..2, j in 0..3, k in 0..7.
    # T[i,j,k] = sum over r of U[i*4+j, r] * V[j*8+k, r] * W[k*3+i, r]
    
    # Let me try to construct it block by block.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2, use Strassen (rank 7)
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> 2x2x2, use Strassen (rank 7)
    # Block 4: i in {0,1}, j in {0,1}, k in {6,7} -> 2x2x2, use Strassen (rank 7)
    # Block 5: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
    # Block 6: i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2, use Strassen (rank 7)
    # Block 7: i in {0,1}, j in {2,3}, k in {4,5} -> 2x2x2, use Strassen (rank 7)
    # Block 8: i in {0,1}, j in {2,3}, k in {6,7} -> 2x2x2, use Strassen (rank 7)
    # Total for i in {0,1}: 8 * 7 = 56
    
    # Block 9: i in {2}, j in {0,1}, k in {0,1} -> 1x2x2, schoolbook (rank 4)
    # Block 10: i in {2}, j in {0,1}, k in {2,3} -> 1x2x2, schoolbook (rank 4)
    # Block 11: i in {2}, j in {0,1}, k in {4,5} -> 1x2x2, schoolbook (rank 4)
    # Block 12: i in {2}, j in {0,1}, k in {6,7} -> 1x2x2, schoolbook (rank 4)
    # Block 13: i in {2}, j in {2,3}, k in {0,1} -> 1x2x2, schoolbook (rank 4)
    # Block 14: i in {2}, j in {2,3}, k in {2,3} -> 1x2x2, schoolbook (rank 4)
    # Block 15: i in {2}, j in {2,3}, k in {4,5} -> 1x2x2, schoolbook (rank 4)
    # Block 16: i in {2}, j in {2,3}, k in {6,7} -> 1x2x2, schoolbook (rank 4)
    # Total for i in {2}: 8 * 4 = 32
    
    # Grand total: 56 + 32 = 88
    
    # Let's build this!
    
    U_new = [[0] * 88 for _ in range(n * m)]  # 12 x 88
    V_new = [[0] * 88 for _ in range(m * p)]  # 32 x 88
    W_new = [[0] * 88 for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Get Strassen decomposition
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # For i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}:
    # Group into 2x2x2 blocks.
    
    for i_base in [0, 1]:  # i = 0 or 1
        for j_base in [0, 2]:  # j = 0,1 or 2,3
            for k_base in [0, 2, 4, 6]:  # k = 0,1 or 2,3 or 4,5 or 6,7
                # This is a 2x2x2 block with:
                # i in {i_base, i_base+1}
                # j in {j_base, j_base+1}
                # k in {k_base, k_base+1}
                
                # Use Strassen decomposition
                for r in range(7):
                    # U[i*4+j, col+r]
                    for di in range(2):
                        for dj in range(2):
                            i = i_base + di
                            j = j_base + dj
                            U_new[i*m + j][col + r] = U_str[i*2 + dj][r]
                    
                    # V[j*8+k, col+r]
                    for dj in range(2):
                        for dk in range(2):
                            j = j_base + dj
                            k = k_base + dk
                            V_new[j*p + k][col + r] = V_str[dj*2 + dk][r]
                    
                    # W[k*3+i, col+r]
                    for dk in range(2):
                        for di in range(2):
                            k = k_base + dk
                            i = i_base + di
                            W_new[k*n + i][col + r] = W_str[dk*2 + di][r]
                
                col += 7
    
    # For i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}:
    # Use schoolbook for 1x2x2 blocks.
    
    for j_base in [0, 2]:  # j = 0,1 or 2,3
        for k_base in [0, 2, 4, 6]:  # k = 0,1 or 2,3 or 4,5 or 6,7
            # This is a 1x2x2 block with:
            # i = 2
            # j in {j_base, j_base+1}
            # k in {k_base, k_base+1}
            
            # Schoolbook: 4 terms
            for dj in range(2):
                for dk in range(2):
                    j = j_base + dj
                    k = k_base + dk
                    i = 2
                    
                    U_new[i*m + j][col] = 1
                    V_new[j*p + k][col] = 1
                    W_new[k*n + i][col] = 1
                    col += 1
    
    # Verify
    if _verify(U_new, V_new, W_new, n, m, p):
        best = {"U": U_new, "V": V_new, "W": W_new, "rank": 88}
    
    if time.time() > DEADLINE:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
