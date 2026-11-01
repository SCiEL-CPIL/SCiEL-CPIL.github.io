"""
Rank-26 exact tensor decomposition for 3x3 matrix multiplication
using a block-partitioned approach: Strassen on the top-left 2x2 block
plus schoolbook for the remaining 1x2, 2x1, and 1x1 blocks.
"""

def construct():
    n, m, p = 3, 3, 3

    # We build rank-26 decomposition:
    # 7 terms for Strassen 2x2 (top-left block)
    # 4 terms for A_1 * b'  (2x2 * 2x1)
    # 4 terms for b^T * B_1 (1x2 * 2x2)
    # 2 terms for c * d^T  (scalar * 1x2)
    # 4 terms for e * A_2  (2x1 * 1x2)... wait let me recount
    #
    # Let's use the standard partition:
    # A = [A11 A12 A13]    B = [B11 B12 B13]    C = [C11 C12 C13]
    #     [A21 A22 A23]        [B21 B22 B23]        [C21 C22 C23]
    #     [A31 A32 A33]        [B31 B32 B33]        [C31 C32 C33]
    #
    # Partition A into:
    #   A1 = [[A11, A12],    a = [A13, A23]^T   b = [A31, A32]
    #         [A21, A22]]
    #   c = A33
    #
    # Similarly for B:
    #   B1 = [[B11, B12],    d = [B13, B23]^T   e = [B31, B32]
    #         [B21, B22]]
    #   f = B33
    #
    # C = A*B:
    #   C1 = A1*B1 + a*e      (top-left 2x2)
    #   C2 = A1*d + a*f        (top-right 2x1)
    #   C3 = b*B1 + c*e        (bottom-left 1x2)
    #   C4 = b*d + c*f         (bottom-right 1x1)
    #
    # Wait, that's not quite right. Let me re-derive:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # For i in {0,1}, k in {0,1}:
    #   C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k] + A[i,2]*B[2,k]
    #
    # For i in {0,1}, k=2:
    #   C[i,2] = A1[i,0]*B[0,2] + A1[i,1]*B[1,2] + A[i,2]*B[2,2]
    #
    # For i=2, k in {0,1}:
    #   C[2,k] = A[2,0]*B1[0,k] + A[2,1]*B1[1,k] + A[2,2]*B[2,k]
    #
    # For i=2, k=2:
    #   C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    #
    # So with:
    #   A1 = [[A00,A01],[A10,A11]], a=[A02,A12], b=[A20,A21], c=A22
    #   B1 = [[B00,B01],[B10,B11]], d=[B02,B12], e=[B20,B21], f=B22
    #
    # C11 (2x2) = A1*B1 + a*e^T    -> Strassen(7) + a*e^T(4) = 11? No, a*e^T is outer product
    # Actually a is 2x1, e is 1x2, so a*e is 2x2 with 4 mults
    # But wait: a*e^T where a=[A02,A12]^T, e=[B20,B21]
    # (a*e)[i,k] = A[i,2]*B[2,k] for i,k in {0,1}. That's 4 terms.
    #
    # C12 (2x1) = A1*d + a*f       -> 4 + 2 = 6 terms
    # C21 (1x2) = b*B1 + c*e       -> 4 + 2 = 6 terms
    # C22 (1x1) = b*d + c*f        -> 2 + 1 = 3 terms
    #
    # Total: 7 + 4 + 6 + 6 + 3 = 26. 

    # Strassen 2x2 terms for A1*B1:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let's define the 26 terms as (U_factor, V_factor, W_factor) triples
    # where the tensor is sum_t U[:,t] outer V[:,t] outer W[:,t]
    # and C[i,k] = sum_j A[i,j] B[j,k] = sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t]
    #
    # U has shape (9, 26), V has shape (9, 26), W has shape (9, 26)
    # Indexing: U[row, col] where row = i*m+j (i is row of A, j is col of A)
    # V[row, col] where row = j*p+k (j is row of B, k is col of B)
    # W[row, col] where row = k*n+i (k is col of B, i is row of A) -- trace convention W[k*n+i, r]

    rank = 26
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]

    # Helper: A[i,j] index in U is i*3+j
    # Helper: B[j,k] index in V is j*3+k
    # Helper: W index for C[i,k] is k*3+i

    def set_term(col, U_pairs, V_pairs, W_pairs):
        for (idx, val) in U_pairs:
            U[idx][col] += val
        for (idx, val) in V_pairs:
            V[idx][col] += val
        for (idx, val) in W_pairs:
            W[idx][col] += val

    t = 0

    # Strassen terms (7 terms)
    # M1 = (A00+A11)(B00+B11)
    # U: A00 at (0,0)=0, A11 at (1,1)=4
    # V: B00 at (0,0)=0, B11 at (1,1)=4
    # W: C00 at (0,0)=0
    set_term(t, [(0,1),(4,1)], [(0,1),(4,1)], [(0,1)]); t += 1

    # M2 = (A10+A11)(B00)
    # U: A10 at (1,0)=3, A11 at (1,1)=4
    # V: B00 at (0,0)=0
    # W: C10 at (0,1)=1
    set_term(t, [(3,1),(4,1)], [(0,1)], [(1,1)]); t += 1

    # M3 = A00(B01-B11)
    # U: A00 at (0,0)=0
    # V: B01 at (0,1)=1, B11 at (1,1)=4
    # W: C01 at (1,0)=1
    set_term(t, [(0,1)], [(1,1),(4,-1)], [(1,1)]); t += 1

    # M4 = A11(B10-B00)
    # U: A11 at (1,1)=4
    # V: B10 at (1,0)=3, B00 at (0,0)=0
    # W: C00 at (0,0)=0 and C10 at (0,1)=1
    set_term(t, [(4,1)], [(3,1),(0,-1)], [(0,1),(1,1)]); t += 1

    # M5 = (A00+A01)(B11)
    # U: A00 at (0,0)=0, A01 at (0,1)=1
    # V: B11 at (1,1)=4
    # W: C00 at (0,0)=0 with coeff -1
    set_term(t, [(0,1),(1,1)], [(4,1)], [(0,-1)]); t += 1

    # M6 = (A10-A00)(B00+B01)
    # U: A10 at (1,0)=3, A00 at (0,0)=0
    # V: B00 at (0,0)=0, B01 at (0,1)=1
    # W: C11 at (1,1)=4
    set_term(t, [(3,1),(0,-1)], [(0,1),(1,1)], [(4,1)]); t += 1

    # M7 = (A01-A11)(B10+B11)
    # U: A01 at (0,1)=1, A11 at (1,1)=4
    # V: B10 at (1,0)=3, B11 at (1,1)=4
    # W: C00 at (0,0)=0
    set_term(t, [(1,1),(4,-1)], [(3,1),(4,1)], [(0,1)]); t += 1

    # a*e^T terms (4 terms): A[i,2]*B[2,k] for i,k in {0,1}
    # C00 += A02*B20
    set_term(t, [(2,1)], [(6,1)], [(0,1)]); t += 1
    # C01 += A02*B21
    set_term(t, [(2,1)], [(7,1)], [(1,1)]); t += 1
    # C10 += A12*B20
    set_term(t, [(5,1)], [(6,1)], [(1,1)]); t += 1
    # C11 += A12*B21
    set_term(t, [(5,1)], [(7,1)], [(4,1)]); t += 1

    # A1*d terms (4 terms): A1[i,0]*B[0,2] + A1[i,1]*B[1,2]
    # C02 = A00*B02 + A01*B12
    set_term(t, [(0,1)], [(2,1)], [(2,1)]); t += 1  # C[0,2] index = 2*3+0=6
    set_term(t, [(1,1)], [(5,1)], [(2,1)]); t += 1  # C[0,2] index = 6
    # C12 = A10*B02 + A11*B12
    set_term(t, [(3,1)], [(2,1)], [(5,1)]); t += 1  # C[1,2] index = 2*3+1=7
    set_term(t, [(4,1)], [(5,1)], [(5,1)]); t += 1  # C[1,2] index = 7

    # a*f terms (2 terms): A[i,2]*B[2,2]
    # C02 += A02*B22
    set_term(t, [(2,1)], [(8,1)], [(2,1)]); t += 1  # C[0,2]=6
    # C12 += A12*B22
    set_term(t, [(5,1)], [(8,1)], [(5,1)]); t += 1  # C[1,2]=7

    # b*B1 terms (4 terms): A[2,j]*B1[j,k]
    # C20 = A20*B00 + A21*B10
    set_term(t, [(6,1)], [(0,1)], [(0,2)]); t += 1  # C[2,0] index = 0*3+2=2
    set_term(t, [(7,1)], [(3,1)], [(0,2)]); t += 1  # C[2,0] index = 2
    # C21 = A20*B01 + A21*B11
    set_term(t, [(6,1)], [(1,1)], [(1,2)]); t += 1  # C[2,1] index = 1*3+2=5
    set_term(t, [(7,1)], [(4,1)], [(1,2)]); t += 1  # C[2,1] index = 5

    # c*e terms (2 terms): A[2,2]*B[2,k]
    # C20 += A22*B20
    set_term(t, [(8,1)], [(6,1)], [(0,2)]); t += 1  # C[2,0]=2
    # C21 += A22*B21
    set_term(t, [(8,1)], [(7,1)], [(1,2)]); t += 1  # C[2,1]=5

    # b*d terms (2 terms): A[2,j]*B[j,2]
    # C22 = A20*B02 + A21*B12
    set_term(t, [(6,1)], [(2,1)], [(8,2)]); t += 1  # C[2,2] index = 2*3+2=8
    set_term(t, [(7,1)], [(5,1)], [(8,2)]); t += 1  # C[2,2] index = 8

    # c*f term (1 term): A22*B22
    # C22 += A22*B22
    set_term(t, [(8,1)], [(8,1)], [(8,2)]); t += 1  # C[2,2]=8

    # Total: 7+4+4+2+4+2+2+1 = 26
    assert t == 26, f"Expected 26 terms, got {t}"

    return {"U": U, "V": V, "W": W}
