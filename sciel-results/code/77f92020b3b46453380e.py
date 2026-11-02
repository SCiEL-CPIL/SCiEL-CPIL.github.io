"""Strassen-based rank-14 decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    # Rank-14 decomposition using Strassen's algorithm structure.
    # For multiplying A (2x2) by B (2x4), we can view B as two 2x2 blocks B1, B2.
    # C1 = A*B1, C2 = A*B2. Each 2x2*2x2 uses 7 Strassen multiplications.
    # But we can share some computations across the two block multiplications
    # because the left factor A is the same.
    #
    # Strassen for 2x2 * 2x2:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # For A (2x2) * B (2x4) where B = [B1 | B2] with B1, B2 being 2x2:
    # We compute C = A*B (2x4).
    #
    # Let's define the 14 bilinear forms. The key insight is that for the
    # second block B2, we can reuse some of the linear combinations of A.
    #
    # Actually, the standard result is that the tensor rank of [2,2,4] is 14.
    # This is achieved by a specific construction. Let me build it carefully.
    #
    # The matrix multiplication tensor for [n,m,p] has entries T[i,j,k] for
    # (i,j) in A, (j,k) in B, output at (i,k). The flattening for mode-1 (A-indices)
    # gives a matrix of size (n*m) x (m*p) with entries indexed by (i,j) vs (j',k)
    # where the tensor entry is 1 if j=j'.
    #
    # For the rank-14 decomposition, I'll use the known construction.
    # The 14 terms correspond to: 7 for the first 2x2 output column pair,
    # and 7 for the second, but with sharing of the A-side linear forms.
    #
    # Let me define it explicitly. A has entries A[0,0], A[0,1], A[1,0], A[1,1]
    # indexed as (i,j) with i=0,1 and j=0,1.
    # B has entries B[j,k] with j=0,1 and k=0,1,2,3.
    # C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    #
    # Strassen for first block (k=0,1):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M4 + M6 - M7  ... let me verify
    # Standard Strassen:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    #
    # For second block (k=2,3), we can reuse the same linear forms on A:
    # M8 = (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    # M9 = (A[1,0]+A[1,1]) * B[0,2]
    # M10 = A[0,0] * (B[1,2]-B[1,3])
    # M11 = A[1,1] * (B[0,3]-B[0,2])
    # M12 = (A[0,0]+A[0,1]) * B[1,3]
    # M13 = (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    # M14 = (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    #
    # C[0,2] = M8 - M11 + M13 - M14
    # C[0,3] = M10 + M12
    # C[1,2] = M9 + M11
    # C[1,3] = M8 - M10 + M12 + M14
    #
    # This gives rank 14. Let me verify the decomposition.

    n, m, p = 2, 2, 4
    rank = 14
    
    # U: shape (n*m, rank) = (4, 14), indexed by (i*m+j) = (i*2+j)
    # V: shape (m*p, rank) = (8, 14), indexed by (j*p+k) = (j*4+k)
    # W: shape (p*n, rank) = (8, 14), indexed by (k*n+i) = (k*2+i)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Helper to set column c with values for U, V, W
    def set_col(c, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            U[idx][c] = val
        for idx, val in v_vals.items():
            V[idx][c] = val
        for idx, val in w_vals.items():
            W[idx][c] = val
    
    # A indices: (i,j) -> i*2+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (j,k) -> j*4+k: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    # C indices: (k,i) -> k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5, (3,0)=6, (3,1)=7
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    set_col(0, {0: 1, 3: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    # C[0,0] += M1, C[1,1] += M1 (wait, need to check signs)
    
    # Let me redo this more carefully.
    # C[0,0] = M1 - M4 + M6 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 + M7
    
    # M1: U has A[0,0]+A[1,1], V has B[0,0]+B[1,1]
    # W contributions: C[0,0]+=1, C[1,1]+=1 (from M1 in both C[0,0] and C[1,1])
    # Wait, C[0,0] = M1 - M4 + M6 - M7, C[1,1] = M1 - M3 + M5 + M7
    # So M1 appears in C[0,0] with +1 and C[1,1] with +1.
    
    # Let me restart with a cleaner approach.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Term 1: (A00+A11) * (B00+B11)
    # A00=idx0, A11=idx3; B00=idx0, B11=idx5
    # Contributes to: C00(+1), C11(+1) -> C indices: C[0,0]=0, C[1,1]=3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: (A10+A11) * B00
    # A10=idx2, A11=idx3; B00=idx0
    # Contributes to: C10(+1) -> C[1,0]=2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: A00 * (B10-B11)
    # A00=idx0; B10=idx4, B11=idx5
    # Contributes to: C01(+1), C11(-1) -> C[0,1]=1, C[1,1]=3
    U[0][2] = 1
    V[4][2] = 1; V[5][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # Term 4: A11 * (B01-B00)
    # A11=idx3; B01=idx1, B00=idx0
    # Contributes to: C00(-1), C10(+1) -> C[0,0]=0, C[1,0]=2
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # Term 5: (A00+A01) * B11
    # A00=idx0, A01=idx1; B11=idx5
    # Contributes to: C01(+1), C11(+1) -> C[0,1]=1, C[1,1]=3
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # Term 6: (A10-A00) * (B00+B01)
    # A10=idx2, A00=idx0; B00=idx0, B01=idx1
    # Contributes to: C00(+1), C10(+1) -> C[0,0]=0, C[1,0]=2
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    # Term 7: (A01-A11) * (B10+B11)
    # A01=idx1, A11=idx3; B10=idx4, B11=idx5
    # Contributes to: C00(-1), C11(+1) -> C[0,0]=0, C[1,1]=3
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    # Now for the second block (k=2,3):
    # C[0,2] = M8 - M11 + M13 - M14
    # C[0,3] = M10 + M12
    # C[1,2] = M9 + M11
    # C[1,3] = M8 - M10 + M12 + M14
    
    # Term 8: (A00+A11) * (B02+B13)
    # A00=idx0, A11=idx3; B02=idx2, B13=idx7
    # Contributes to: C02(+1), C13(+1) -> C[0,2]=4, C[1,3]=7
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # Term 9: (A10+A11) * B02
    # A10=idx2, A11=idx3; B02=idx2
    # Contributes to: C12(+1) -> C[1,2]=6
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[6][8] = 1
    
    # Term 10: A00 * (B12-B13)
    # A00=idx0; B12=idx6, B13=idx7
    # Contributes to: C03(+1), C13(-1) -> C[0,3]=5, C[1,3]=7
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[5][9] = 1; W[7][9] = -1
    
    # Term 11: A11 * (B03-B02)
    # A11=idx3; B03=idx3, B02=idx2
    # Contributes to: C02(-1), C12(+1) -> C[0,2]=4, C[1,2]=6
    U[3][10] = 1
    V[3][10] = 1; V[2][10] = -1
    W[4][10] = -1; W[6][10] = 1
    
    # Term 12: (A00+A01) * B13
    # A00=idx0, A01=idx1; B13=idx7
    # Contributes to: C03(+1), C13(+1) -> C[0,3]=5, C[1,3]=7
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[5][11] = 1; W[7][11] = 1
    
    # Term 13: (A10-A00) * (B02+B03)
    # A10=idx2, A00=idx0; B02=idx2, B03=idx3
    # Contributes to: C02(+1), C12(+1) -> C[0,2]=4, C[1,2]=6
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[4][12] = 1; W[6][12] = 1
    
    # Term 14: (A01-A11) * (B12+B13)
    # A01=idx1, A11=idx3; B12=idx6, B13=idx7
    # Contributes to: C02(-1), C13(+1) -> C[0,2]=4, C[1,3]=7
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = -1; W[7][13] = 1
    
    return {"U": U, "V": V, "W": W}
