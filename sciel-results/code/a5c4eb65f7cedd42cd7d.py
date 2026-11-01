import numpy as np
from fractions import Fraction

def construct():
    # Decompose 2x3 x 3x3 multiplication using Strassen for the 2x2x2 core
    # plus explicit tail terms.
    # A is 2x3, B is 3x3. C = A*B is 2x3.
    # Let A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1.
    # Let B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We use Strassen for the 2x2x2 part involving A11, B11, C11.
    # But wait, C11 also has A12*B21.
    # Let's define the full tensor.
    # Indices: i in {0,1} (rows of A), j in {0,1,2} (cols of A = rows of B), k in {0,1,2} (cols of B).
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Strategy:
    # Split j into {0,1} and {2}.
    # Let A0 = A[:, 0:2] (2x2), A1 = A[:, 2] (2x1).
    # Let B0 = B[0:2, :] (2x3), B1 = B[2, :] (1x3).
    # C = A0 @ B0 + A1 @ B1 (where A1@B1 is outer product of 2x1 and 1x3 -> 2x3).
    
    # We need to compute M1 = A0 @ B0 (2x3 result) and M2 = A1 @ B1 (2x3 result).
    
    # For M1 = A0 @ B0:
    # A0 is 2x2. B0 is 2x3.
    # B0 = [B00, B01] where B00 is 2x2, B01 is 2x1.
    # M1 = A0 @ [B00, B01] = [A0 @ B00, A0 @ B01].
    # Let C1_left = A0 @ B00 (2x2).
    # Let C1_right = A0 @ B01 (2x1).
    
    # Use Strassen for C1_left (rank 7).
    # A0 = [a00 a01; a10 a11]
    # B00 = [b00 b01; b10 b11]
    # S1 = b11 - b01
    # S2 = a00 + a01
    # S3 = a10 + a11
    # S4 = b00 + b11
    # S5 = a01 - a11
    # S6 = b01 + b10
    # M1 = a11 * S1
    # M2 = a00 * b00
    # M3 = S5 * b11
    # M4 = a11 * b01
    # M5 = S2 * S4
    # M6 = S6 * S3
    # M7 = S5 * S1
    # C1_left = [M1+M4+M5+M7, M3+M5; M2+M6, M1+M2+M4]
    
    # Now C1_right = A0 @ B01.
    # B01 = [b02; b12] (2x1).
    # A0 @ B01 = [a00*b02 + a01*b12; a10*b02 + a11*b12].
    # This requires 4 multiplications: a00*b02, a01*b12, a10*b02, a11*b12.
    # Can we reuse Strassen terms?
    # Strassen uses a00, a01, a10, a11 and b00, b01, b10, b11.
    # C1_right uses a00, a01, a10, a11 and b02, b12.
    # The B indices are disjoint (b02, b12 vs b00..b11).
    # So we can't directly reuse the multiplications unless we modify the Strassen steps.
    # However, we can compute C1_right with 4 multiplications.
    
    # Now M2 = A1 @ B1.
    # A1 = [a02; a12] (2x1).
    # B1 = [b20, b21, b22] (1x3).
    # M2 = [a02*b20, a02*b21, a02*b22; a12*b20, a12*b21, a12*b22].
    # This requires 2*3 = 6 multiplications.
    
    # Total rank so far: 7 (Strassen) + 4 (C1_right) + 6 (M2) = 17.
    
    # Let's check if we can do better.
    # Is there a known lower bound?
    # Rank of 2x3x3 tensor.
    # The standard schoolbook is 2*3*3 = 18.
    # We found 17.
    
    # Let's verify the 17 construction carefully.
    # We need to output U, V, W matrices.
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    # Indices for U: i*m + j. i in 0..1, j in 0..2.
    # 0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2).
    # Let's denote A indices as A_ij.
    # A00 -> U[0], A01 -> U[1], A02 -> U[2], A10 -> U[3], A11 -> U[4], A12 -> U[5].
    
    # Indices for V: j*p + k. j in 0..2, k in 0..2.
    # 0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2), 6: (2,0), 7: (2,1), 8: (2,2).
    # B00 -> V[0], B01 -> V[1], B02 -> V[2], B10 -> V[3], B11 -> V[4], B12 -> V[5], B20 -> V[6], B21 -> V[7], B22 -> V[8].
    
    # Indices for W: k*n + i. k in 0..2, i in 0..1.
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1), 4: (2,0), 5: (2,1).
    # C00 -> W[0], C01 -> W[1], C10 -> W[2], C11 -> W[3], C20 -> W[4], C21 -> W[5].
    # Note: W index is k*n + i.
    # C[i,k] corresponds to W[k*n + i].
    # C[0,0] -> W[0*2+0] = W[0].
    # C[0,1] -> W[1*2+0] = W[2].
    # C[0,2] -> W[2*2+0] = W[4].
    # C[1,0] -> W[0*2+1] = W[1].
    # C[1,1] -> W[1*2+1] = W[3].
    # C[1,2] -> W[2*2+1] = W[5].
    
    # Let's build the 17 terms.
    # Terms 0-6: Strassen for C11 (which is C[0,0] and C[1,0] part? No, C1_left is C[0:2, 0:2]).
    # C1_left is the 2x2 block at top-left of C.
    # C[0,0] and C[0,1] come from A0 @ B0[:, 0:2].
    # C[1,0] and C[1,1] come from A0 @ B0[:, 0:2].
    
    # Let's re-evaluate the output indices.
    # C is 2x3.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # My previous split:
    # A0 = A[:, 0:2], A1 = A[:, 2].
    # B0 = B[0:2, :], B1 = B[2, :].
    # C = A0 @ B0 + A1 @ B1.
    # A0 @ B0:
    # (A0 @ B0)[0,0] = A0[0,0]B0[0,0] + A0[0,1]B0[1,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # (A0 @ B0)[0,1] = A0[0,0]B0[0,1] + A0[0,1]B0[1,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
    # (A0 @ B0)[0,2] = A0[0,0]B0[0,2] + A0[0,1]B0[1,2] = A[0,0]B[0,2] + A[0,1]B[1,2].
    # (A0 @ B0)[1,0] = A0[1,0]B0[0,0] + A0[1,1]B0[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0].
    # (A0 @ B0)[1,1] = A0[1,0]B0[0,1] + A0[1,1]B0[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1].
    # (A0 @ B0)[1,2] = A0[1,0]B0[0,2] + A0[1,1]B0[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2].
    
    # A1 @ B1:
    # (A1 @ B1)[0,0] = A1[0]B1[0] = A[0,2]B[2,0].
    # (A1 @ B1)[0,1] = A1[0]B1[1] = A[0,2]B[2,1].
    # (A1 @ B1)[0,2] = A1[0]B1[2] = A[0,2]B[2,2].
    # (A1 @ B1)[1,0] = A1[1]B1[0] = A[1,2]B[2,0].
    # (A1 @ B1)[1,1] = A1[1]B1[1] = A[1,2]B[2,1].
    # (A1 @ B1)[1,2] = A1[1]B1[2] = A[1,2]B[2,2].
    
    # So C[0,0] = (A0 @ B0)[0,0] + A[0,2]B[2,0].
    # C[0,1] = (A0 @ B0)[0,1] + A[0,2]B[2,1].
    # C[0,2] = (A0 @ B0)[0,2] + A[0,2]B[2,2].
    # C[1,0] = (A0 @ B0)[1,0] + A[1,2]B[2,0].
    # C[1,1] = (A0 @ B0)[1,1] + A[1,2]B[2,1].
    # C[1,2] = (A0 @ B0)[1,2] + A[1,2]B[2,2].
    
    # We need to compute A0 @ B0 efficiently.
    # A0 is 2x2, B0 is 2x3.
    # Let B0 = [B_left, B_right] where B_left is 2x2 (cols 0,1), B_right is 2x1 (col 2).
    # A0 @ B0 = [A0 @ B_left, A0 @ B_right].
    
    # Part 1: A0 @ B_left (2x2 matrix multiplication).
    # Use Strassen (rank 7).
    # Part 2: A0 @ B_right (2x1 matrix multiplication).
    # A0 @ B_right = [A0[0,0]*B_right[0,0] + A0[0,1]*B_right[1,0]; A0[1,0]*B_right[0,0] + A0[1,1]*B_right[1,0]].
    # This is 4 multiplications.
    
    # Part 3: A1 @ B1 (2x1 @ 1x3 -> 2x3).
    # 6 multiplications.
    
    # Total 7 + 4 + 6 = 17.
    
    # Let's construct the U, V, W matrices.
    rank = 17
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    
    # Strassen terms for A0 @ B_left
    # A0: U[0]=A00, U[1]=A01, U[3]=A10, U[4]=A11.
    # B_left: V[0]=B00, V[1]=B01, V[3]=B10, V[4]=B11.
    
    # M1 = A11 * (B11 - B01)
    U[4][0] = 1
    V[4][0] = 1
    V[1][0] = -1
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1] -> wait, C[1,1] is (A0 @ B_left)[1,1].
    # Strassen C11 (top-left of 2x2 result) = M1 + M4 + M5 + M7.
    # Strassen C12 (top-right) = M3 + M5.
    # Strassen C21 (bottom-left) = M2 + M6.
    # Strassen C22 (bottom-right) = M1 + M2 + M4.
    # Here result is A0 @ B_left.
    # (A0 @ B_left)[0,0] = M1 + M4 + M5 + M7.
    # (A0 @ B_left)[0,1] = M3 + M5.
    # (A0 @ B_left)[1,0] = M2 + M6.
    # (A0 @ B_left)[1,1] = M1 + M2 + M4.
    
    # Let's map Strassen M1..M7 to columns 0..6.
    
    # Col 0: M1 = A11 * (B11 - B01)
    U[4][0] = 1
    V[4][0] = 1
    V[1][0] = -1
    # Contributions to W:
    # (A0 @ B_left)[0,0] gets M1. W[0][0] += 1.
    # (A0 @ B_left)[1,1] gets M1. W[3][0] += 1.
    W[0][0] = 1
    W[3][0] = 1
    
    # Col 1: M2 = A00 * B00
    U[0][1] = 1
    V[0][1] = 1
    # (A0 @ B_left)[1,0] gets M2. W[2][1] += 1.
    # (A0 @ B_left)[1,1] gets M2. W[3][1] += 1.
    W[2][1] = 1
    W[3][1] = 1
    
    # Col 2: M3 = (A01 - A11) * B11
    U[1][2] = 1
    U[4][2] = -1
    V[4][2] = 1
    # (A0 @ B_left)[0,1] gets M3. W[2][2] += 1. (Wait, W index for C[0,1] is W[1*2+0]=W[2]).
    # Let's re-verify W indices.
    # C[i,k] -> W[k*n+i].
    # C[0,0] -> W[0].
    # C[0,1] -> W[2].
    # C[0,2] -> W[4].
    # C[1,0] -> W[1].
    # C[1,1] -> W[3].
    # C[1,2] -> W[5].
    
    # (A0 @ B_left)[0,1] is C[0,1] part. So W[2].
    W[2][2] = 1
    
    # Col 3: M4 = A11 * B01
    U[4][3] = 1
    V[1][3] = 1
    # (A0 @ B_left)[0,0] gets M4. W[0][3] += 1.
    # (A0 @ B_left)[1,1] gets M4. W[3][3] += 1.
    W[0][3] = 1
    W[3][3] = 1
    
    # Col 4: M5 = (A00 + A01) * (B00 + B11)
    U[0][4] = 1
    U[1][4] = 1
    V[0][4] = 1
    V[4][4] = 1
    # (A0 @ B_left)[0,0] gets M5. W[0][4] += 1.
    # (A0 @ B_left)[0,1] gets M5. W[2][4] += 1.
    W[0][4] = 1
    W[2][4] = 1
    
    # Col 5: M6 = (B01 + B10) * (A10 + A11)
    V[1][5] = 1
    V[3][5] = 1
    U[3][5] = 1
    U[4][5] = 1
    # (A0 @ B_left)[1,0] gets M6. W[1][5] += 1. (C[1,0] -> W[1]).
    W[1][5] = 1
    
    # Col 6: M7 = (A01 - A11) * (B11 - B01)
    U[1][6] = 1
    U[4][6] = -1
    V[4][6] = 1
    V[1][6] = -1
    # (A0 @ B_left)[0,0] gets M7. W[0][6] += 1.
    W[0][6] = 1
    
    # Now Part 2: A0 @ B_right (2x1).
    # B_right is B[:, 2]. V[2]=B02, V[5]=B12.
    # (A0 @ B_right)[0,0] = A00*B02 + A01*B12.
    # (A0 @ B_right)[1,0] = A10*B02 + A11*B12.
    # These contribute to C[0,2] and C[1,2].
    # C[0,2] -> W[4]. C[1,2] -> W[5].
    
    # Col 7: A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Col 8: A01 * B12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Col 9: A10 * B02
    U[3][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Col 10: A11 * B12
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    # Now Part 3: A1 @ B1.
    # A1 = A[:, 2]. U[2]=A02, U[5]=A12.
    # B1 = B[2, :]. V[6]=B20, V[7]=B21, V[8]=B22.
    # C[0,2] += A02 * B22.
    # C[1,2] += A12 * B22.
    # C[0,1] += A02 * B21.
    # C[1,1] += A12 * B21.
    # C[0,0] += A02 * B20.
    # C[1,0] += A12 * B20.
    
    # Col 11: A02 * B20
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1 # C[0,0]
    
    # Col 12: A02 * B21
    U[2][12] = 1
    V[7][12] = 1
    W[2][12] = 1 # C[0,1]
    
    # Col 13: A02 * B22
    U[2][13] = 1
    V[8][13] = 1
    W[4][13] = 1 # C[0,2]
    
    # Col 14: A12 * B20
    U[5][14] = 1
    V[6][14] = 1
    W[1][14] = 1 # C[1,0]
    
    # Col 15: A12 * B21
    U[5][15] = 1
    V[7][15] = 1
    W[3][15] = 1 # C[1,1]
    
    # Col 16: A12 * B22
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1 # C[1,2]
    
    return {"U": U, "V": V, "W": W}

DIMENSIONS = [2, 3, 3]

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify
    n, m, p = DIMENSIONS
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal delta[i,k] (since C[i,k] = sum_j A[i,j]B[j,k])
    # Actually, the tensor T_M[i,j,k] for matrix multiplication is 1 if i,j,k are valid indices.
    # The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] ? No.
    # The tensor is T[i,j,k].
    # U is indexed by (i,j). V by (j,k). W by (k,i).
    # So sum_c U[i,j,c] V[j,k,c] W[k,i,c] should be 1 for all i,j,k.
    
    tensor = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    expected = np.ones((n, m, p))
    if np.allclose(tensor, expected):
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
        print(tensor)
