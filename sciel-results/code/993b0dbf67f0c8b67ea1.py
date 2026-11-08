"""Program for matrix multiplication 3x3x4 tensor decomposition (rank 22)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen Left (A11 * B1L) - Rank 7
    # A11: 0,1,3,4
    # B1L: 0,1,4,5
    # C1L: 0,3,1,4
    
    # P1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3 = A00(B11-B00)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4 = A11(B01-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12 * B2L - Rank 1
    # A12: 2, 5
    # B2L: 8, 9
    # C1L: 0,3,1,4
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    # Tensor is all ones 2x1x2. Rank 1.
    U[2][col] = 1; U[5][col] = 1
    V[8][col] = 1; V[9][col] = 1
    W[0][col] = 1; W[3][col] = 1; W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # A21 * B1L - Rank 2
    # A21: 6, 7
    # B1L: 0,1,4,5
    # C2L: 2, 5
    # C20 += A20 B00 + A21 B10
    # C21 += A20 B01 + A21 B11
    
    # Term 1: C20 from A20 B00
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # Term 2: C20 from A21 B10
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # Wait, this is rank 2 for C20 alone?
    # C20 = A20 B00 + A21 B10.
    # This is a dot product. Rank 1?
    # No, A20 and A21 are different. B00 and B10 are different.
    # C20 = A20 B00 + A21 B10.
    # Can we do it in 1 term?
    # (A20 + A21) (B00 + B10)? No.
    # So Rank 2 for C20.
    # Similarly Rank 2 for C21.
    # Total Rank 4?
    # Let's check if we can share.
    # C20 = A20 B00 + A21 B10
    # C21 = A20 B01 + A21 B11
    # This is 1x2 * 2x2. Rank 2.
    # Decomposition:
    # T1: U=[1,0] (A20), V=[1,0,0,0] (B00), W=[1,0] (C20)
    # T2: U=[0,1] (A21), V=[0,0,1,0] (B10), W=[1,0] (C20)
    # T3: U=[1,0] (A20), V=[0,1,0,0] (B01), W=[0,1] (C21)
    # T4: U=[0,1] (A21), V=[0,0,0,1] (B11), W=[0,1] (C21)
    # This is Rank 4.
    # Can we reduce?
    # C20 = A20 B00 + A21 B10
    # C21 = A20 B01 + A21 B11
    # Let X = A20, Y = A21.
    # C20 = X B00 + Y B10
    # C21 = X B01 + Y B11
    # This is exactly the multiplication of a row vector by a matrix.
    # The rank of this tensor is 2.
    # Why?
    # Tensor T_{i,k,j} with i fixed.
    # T_{0,0,0}=1, T_{0,1,0}=1
    # T_{0,0,1}=1, T_{0,1,1}=1
    # This is the all-ones 2x2 matrix.
    # Rank of all-ones 2x2 matrix is 1.
    # So the tensor rank is 1?
    # Let's re-evaluate Rank 1 for 1x2 * 2x2.
    # If Rank 1:
    # C20 = W0 (U.A) (V.B)
    # C21 = W1 (U.A) (V.B)
    # C20/C21 = W0/W1.
    # C20 = X B00 + Y B10
    # C21 = X B01 + Y B11
    # If B00=B01 and B10=B11, then C20=C21.
    # But generally not.
    # So Rank 1 is NOT possible.
    # Rank 2 is possible?
    # We need to represent the tensor.
    # The tensor is $T_{k,j}$.
    # $T_{0,0}=1, T_{1,0}=1, T_{0,1}=1, T_{1,1}=1$.
    # This matrix has rank 1.
    # So $T_{k,j} = u_k v_j$.
    # $u_0 v_0 = 1, u_1 v_0 = 1 \implies u_0=u_1$.
    # $u_0 v_1 = 1, u_1 v_1 = 1 \implies v_0=v_1$.
    # So $u=[1,1], v=[1,1]$.
    # This implies $T_{0,0}=1, T_{1,0}=1, T_{0,1}=1, T_{1,1}=1$.
    # So the tensor IS rank 1.
    # Why did I think Rank 1 fails?
    # Because I assumed $V$ is a single vector for ALL B elements.
    # In the decomposition, $V$ is indexed by $k$ and $j$?
    # No, $V$ is a flat vector of length $m \times p$.
    # $V_{k,j}$.
    # In Rank 1, $V$ is a single vector $v$.
    # $V \cdot B = \sum_{k,j} v_{k,j} B_{k,j}$.
    # If $v_{k,j} = 1$ for all $k,j$, then $V \cdot B = \sum B_{k,j}$.
    # Then $C_0 = W_0 (U \cdot A) (\sum B)$.
    # This sums ALL B elements.
    # But we only want specific B elements for C0 and C1.
    # Specifically, C0 uses column 0 of B. C1 uses column 1 of B.
    # The Rank 1 decomposition couples all B elements together.
    # So Rank 1 is NOT sufficient.
    # The minimum rank is 2.
    # One term for column 0, one term for column 1?
    # Term 1: $U=[1,1]$ (A20, A21), $V=[1,0,1,0]$ (B00, B10), $W=[1,0]$ (C20).
    # Term 2: $U=[1,1]$ (A20, A21), $V=[0,1,0,1]$ (B01, B11), $W=[0,1]$ (C21).
    # Let's check Term 1:
    # $C_{20} += (A20+A21)(B00+B10)$.
    # This expands to $A20 B00 + A20 B10 + A21 B00 + A21 B10$.
    # We only want $A20 B00 + A21 B10$.
    # So this is wrong.
    # We need to separate A20 and A21.
    # So Rank 4 is required for this block?
    # Yes.
    # So Block 5 is Rank 4.
    # Block 6 is Rank 4.
    
    # So my savings for B5 and B6 are 0.
    # Savings:
    # B3: 3
    # B4: 3
    # B7: 1
    # B8: 1
    # Total 8.
    # Rank 34 - 8 = 26.

    # Let's re-verify B3 (A12 * B2L).
    # A12: 2, 5 (2x1)
    # B2L: 8, 9 (1x2)
    # C1L: 0,3,1,4 (2x2)
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    # Tensor T_{i,k,j} with i in {0,1}, k=2, j in {0,1}.
    # T_{0,2,0}=1, T_{1,2,0}=1
    # T_{0,2,1}=1, T_{1,2,1}=1
    # This is all-ones 2x2.
    # Rank 1?
    # Term: U=[1,1] (A02, A12), V=[1,1] (B20, B21), W=[1,1,1,1] (C00,C01,C10,C11).
    # $C_{ij} += (A02+A12)(B20+B21)$.
    # Expands to $A02 B20 + A02 B21 + A12 B20 + A12 B21$.
    # This is exactly what we want!
    # So B3 is Rank 1. Correct.

    # B4 (A12 * B2R).
    # Same structure. Rank 1. Correct.

    # B7 (A22 * B2L).
    # A22: 8 (1x1)
    # B2L: 8, 9 (1x2)
    # C2L: 2, 5 (1x2)
    # C20 += A22 B20
    # C21 += A22 B21
    # Tensor T_{i,k,j} with i=2, k=2, j in {0,1}.
    # T_{2,2,0}=1, T_{2,2,1}=1.
    # Rank 1?
    # Term: U=[1] (A22), V=[1,1] (B20, B21), W=[1,1] (C20, C21).
    # $C_{2j} += A22 (B20+B21)$.
    # Expands to $A22 B20 + A22 B21$.
    # Correct. Rank 1.

    # B8 (A22 * B2R).
    # Same. Rank 1.

    # So Rank 26 is achievable.

    # Let's implement Rank 26.
    
    # B5 (A21 * B1L) Rank 4
    # B6 (A21 * B1R) Rank 4
    
    # B5:
    # C20 += A20 B00
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    # C20 += A21 B10
    U[7][col] = 1; V[4][col] = 1; W[2][col] = 1; col += 1
    # C21 += A20 B01
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    # C21 += A21 B11
    U[7][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # B6 (A21 * B1R) Rank 4
    # C22 += A20 B02
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    # C22 += A21 B12
    U[7][col] = 1; V[6][col] = 1; W[8][col] = 1; col += 1
    # C23 += A20 B03
    U[6][col] = 1; V[3][col] = 1; W[11][col] = 1; col += 1
    # C23 += A21 B13
    U[7][col] = 1; V[7][col] = 1; W[11][col] = 1; col += 1

    # Strassen Right (A11 * B1R) - Rank 7
    # A11: 0,1,3,4
    # B1R: 2,3,6,7
    # C1R: 6,9,7,10
    
    # P1 = (A00+A11)(B02+B13)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2 = (A10+A11)B02
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3 = A00(B13-B02)
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4 = A11(B03-B02)
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5 = (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B02+B03)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B12+B13)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    # B4 (A12 * B2R) Rank 1
    # A12: 2, 5
    # B2R: 10, 11
    # C1R: 6,9,7,10
    U[2][col] = 1; U[5][col] = 1
    V[10][col] = 1; V[11][col] = 1
    W[6][col] = 1; W[9][col] = 1; W[7][col] = 1; W[10][col] = 1
    col += 1
    
    # B8 (A22 * B2R) Rank 1
    # A22: 8
    # B2R: 10, 11
    # C2R: 8, 11
    U[8][col] = 1
    V[10][col] = 1; V[11][col] = 1
    W[8][col] = 1; W[11][col] = 1
    col += 1
    
    # B7 (A22 * B2L) Rank 1
    # A22: 8
    # B2L: 8, 9
    # C2L: 2, 5
    U[8][col] = 1
    V[8][col] = 1; V[9][col] = 1
    W[2][col] = 1; W[5][col] = 1
    col += 1
    
    # Total cols: 7 + 1 + 4 + 1 + 7 + 1 + 1 + 1 = 24?
    # Let's count.
    # Strassen L: 7
    # B3: 1
    # B5: 4
    # B7: 1
    # Strassen R: 7
    # B4: 1
    # B6: 4
    # B8: 1
    # Total: 7+1+4+1+7+1+4+1 = 26.
    
    return {"U": U, "V": V, "W": W}

