"""Rewritten program for 2x3x10 matrix multiplication tensor rank minimization using Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Strategy: Split middle dimension m=3 into m1=2 and m2=1
    # For j in {0,1}: 2x2x10 sub-tensor -> group columns into 5 pairs of 2, use Strassen (rank 7) per 2x2x2
    # For j=2: 2x1x10 sub-tensor -> schoolbook rank 20
    # Total rank: 5*7 + 20 = 55

    m1 = 2  # first part of middle dimension
    m2 = 1  # second part of middle dimension
    p_groups = p // 2  # 5 groups of 2 columns each

    # Strassen coefficients for 2x2x2: 7 rank-1 terms
    # C11 = (A11+A22)(B11+B22) - E5 - E6
    # C12 = A11(B21+B22) - E4 - E6
    # C21 = (A11+A22)B11 - E1 - E3
    # C22 = (A21+A22)(B11+B22) - E2 - E5
    # where E1=(A21-A11)B11, E2=A22(B21-B11), E3=A11(B21-B11), E4=(A11+A22)B22,
    #       E5=A22(B11+B22), E6=(A11+A22)(B11+B22)
    # Actually, let me use the standard Strassen formulation:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7

    # For the 2x2x2 tensor, the 7 rank-1 components have these (U,V,W) values:
    # Each component c has U[i], V[j], W[k] for i,j,k in {0,1}
    # M1: U=[1,1], V=[1,1], W=[1,1] -> but this is for the product, need to map to tensor
    # 
    # Actually, let me think about this more carefully.
    # The tensor T[i,j,k] for 2x2x2 matrix mult has T[i,j,k]=1 for all i,j,k.
    # Wait no, that's not right either.
    #
    # For matrix multiplication A(2x2) * B(2x2) = C(2x2):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k in {0,1}
    # 
    # Strassen decomposition: C[i,k] = sum_c U_c[i] V_c[j] W_c[k]
    # where the 7 terms are:
    # Term 1: U=[1,1], V=[1,1], W=[1,1]  (M1)
    # Term 2: U=[0,1], V=[1,0], W=[1,0]  (M2)
    # Term 3: U=[1,0], V=[0,1], W=[1,0]  (M3) -- wait, let me be more careful
    #
    # Let me use the explicit Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # Now, for the tensor decomposition T[i,j,k] where C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]:
    # T[i,j,k] should satisfy: sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
    #
    # For Strassen, each M_c contributes: M_c = (linear combo of A entries) * (linear combo of B entries)
    # So for each c, U_c[i], V_c[j], W_c[k] encode the linear combos.
    #
    # M1 = (A00+A11)(B00+B11):
    #   U_1[0] = 1 (coefficient of A00), U_1[1] = 1 (coefficient of A11)
    #   V_1[0] = 1 (coefficient of B00), V_1[1] = 1 (coefficient of B11)
    #   W_1[0] = 1 (contributes to C00), W_1[1] = 1 (contributes to C11)
    #   So U_1=[1,1], V_1=[1,1], W_1=[1,1]
    #
    # M2 = (A10+A11)B00:
    #   U_2[0] = 0, U_2[1] = 1 (A10+A11 means coeff of A00 is 0, coeff of A11 is 1... wait)
    #   A10 is A[1,0], A11 is A[1,1]. So for row i=0: no contribution, for row i=1: A[1,0]+A[1,1]
    #   U_2[0] = 0, U_2[1] = 1? No, U_2[i] is the coefficient for A[i,j].
    #   M2 = (A[1,0] + A[1,1]) * B[0,0]
    #   So for each j: if j=0, coefficient from A is 1 (for i=1); if j=1, coefficient from A is 1 (for i=1)
    #   U_2[0] = 0 (no A[0,j] contribution), U_2[1] = 1 (A[1,j] contribution)
    #   V_2[0] = 1 (B[0,k] contribution), V_2[1] = 0
    #   W_2[0] = 1 (contributes to C[1,0]), W_2[1] = 0
    #   So U_2=[0,1], V_2=[1,0], W_2=[1,0]
    #
    # M3 = A00(B01-B11):
    #   U_3[0] = 1, U_3[1] = 0
    #   V_3[0] = 0, V_3[1] = 1... wait, B01 is B[0,1], B11 is B[1,1]
    #   M3 = A[0,0] * (B[0,1] - B[1,1])
    #   U_3[0] = 1, U_3[1] = 0
    #   V_3[0] = 1, V_3[1] = -1
    #   W_3[0] = 0, W_3[1] = 1 (contributes to C[0,1])
    #   So U_3=[1,0], V_3=[1,-1], W_3=[0,1]
    #
    # M4 = A11(B10-B00):
    #   M4 = A[1,1] * (B[1,0] - B[0,0])
    #   U_4[0] = 0, U_4[1] = 1
    #   V_4[0] = -1, V_4[1] = 1
    #   W_4[0] = 1, W_4[1] = 0 (contributes to C[1,0] and C[0,0])
    #   Wait, C00 = M1+M4-M5-M7, C10 = M2+M4. So M4 contributes to C00 and C10.
    #   W_4[0] = 1, W_4[1] = 0? But C10 is C[1,0], so W_4 should have W_4[0]=1 for C[*,0]
    #   Actually W_c[k] is the coefficient for C[i,k]. So:
    #   M4 contributes to C[0,0] with coeff +1 and C[1,0] with coeff +1
    #   So W_4[0] = 1, W_4[1] = 0
    #   U_4=[0,1], V_4=[-1,1], W_4=[1,0]
    #
    # M5 = (A00+A01)B11:
    #   M5 = (A[0,0]+A[0,1]) * B[1,1]
    #   U_5[0] = 1, U_5[1] = 0
    #   V_5[0] = 0, V_5[1] = 1
    #   W_5[0] = -1 (C00 = ...-M5...), W_5[1] = 1 (C01 = M3+M5)
    #   So U_5=[1,0], V_5=[0,1], W_5=[-1,1]
    #
    # M6 = (A10-A00)(B00+B01):
    #   M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    #   U_6[0] = -1, U_6[1] = 1
    #   V_6[0] = 1, V_6[1] = 0
    #   W_6: M6 doesn't appear in any C formula! Let me recheck.
    #   Standard Strassen: C00=M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M7
    #   M6 is not used? That can't be right. Let me use a different formulation.
    
    # Let me use a cleaner Strassen formulation:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00-B10)
    # M3 = (A00-A10)(B10+B11)
    # M4 = A00(B01-B11)
    # M5 = A11(B10-B00)
    # M6 = (A00+A01)(B11-B01)
    # M7 = (A10+A11)(B00-B10) -- this is same as M2, so let me use different
    #
    # I'm getting confused. Let me just use the well-known 7-term Strassen:
    # p1 = (A11+A22)(B11+B22)
    # p2 = (A21+A22)B11
    # p3 = A11(B12-B22)
    # p4 = A22(B21-B11)
    # p5 = (A11+A12)B22
    # p6 = (A21-A11)(B11+B12)
    # p7 = (A12-A22)(B21+B22)
    #
    # C11 = p1 + p4 - p5 - p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p7
    #
    # Mapping to 0-indexed (i,j,k in {0,1}):
    # p1: U=[1,1], V=[1,1], W=[1,1]
    # p2: U=[0,1], V=[1,0], W=[0,1]  (C21 = p2+p4, so W[1]=1 for C[1,0])
    #     Wait, C21 means C[1,0] in 0-indexed. So p2 contributes to C[1,0].
    #     p2 = (A21+A22)B11 = (A[1,0]+A[1,1])*B[0,0]
    #     U2[0]=0, U2[1]=1; V2[0]=1, V2[1]=0; W2[0]=1, W2[1]=0 (C[1,0])
    #     But C21 is C[2,1] in 1-indexed = C[1,0] in 0-indexed. Yes.
    #     So W2=[1,0]
    # p3: U=[1,0], V=[1,-1], W=[0,1]  (C12=p3+p5, C[0,1])
    #     p3 = A11(B12-B22) = A[0,0]*(B[0,1]-B[1,1])
    #     U3[0]=1, U3[1]=0; V3[0]=1, V3[1]=-1; W3[0]=0, W3[1]=1
    # p4: U=[0,1], V=[-1,1], W=[1,0]  (C11=p1+p4-..., C21=p2+p4)
    #     p4 = A22(B21-B11) = A[1,1]*(B[1,0]-B[0,0])
    #     U4[0]=0, U4[1]=1; V4[0]=-1, V4[1]=1; W4[0]=1, W4[1]=0
    #     But p4 contributes to C11 (C[0,0]) and C21 (C[1,0]), so W4=[1,0]
    # p5: U=[1,0], V=[0,1], W=[-1,1]  (C11=...-p5, C12=p3+p5)
    #     p5 = (A11+A12)B22 = (A[0,0]+A[0,1])*B[1,1]
    #     U5[0]=1, U5[1]=0; V5[0]=0, V5[1]=1; W5[0]=-1, W5[1]=1
    # p6: U=[-1,1], V=[1,0], W=[0,0]  -- p6 doesn't appear in C formulas!
    #     Let me recheck. In this formulation, is p6 used?
    #     C11=p1+p4-p5-p7, C12=p3+p5, C21=p2+p4, C22=p1-p2+p3+p7
    #     p6 is not used. That's a problem.
    
    # I think I have the wrong Strassen formulas. Let me use the canonical ones:
    # C = A*B where A,B,C are 2x2
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)(b11-b21)
    # m3 = (a11-a21)(b21+b22)
    # m4 = a11(b12-b22)
    # m5 = a22(b21-b11)
    # m6 = (a11+a12)(b22-b12)
    # m7 = (a12+a22)(b11-b21)
    #
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    #
    # Now in 0-indexed (i,j,k in {0,1}):
    # m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    #   U=[1,1], V=[1,1], W=[1,1]
    # m2 = (A[1,0]+A[1,1])(B[0,0]-B[1,0])
    #   U=[0,1], V=[1,-1], W=[1,0]  (c21=m2+m4, so W[0]=1)
    # m3 = (A[0,0]-A[1,0])(B[1,0]+B[1,1])
    #   U=[1,-1], V=[0,1], W=[0,1]  (c12=m3+m5, c22=...+m3, so W[1]=1)
    #   Wait, c12 is C[0,1], c22 is C[1,1]. So m3 contributes to C[0,1] and C[1,1].
    #   W=[0,1]
    # m4 = A[0,0](B[0,1]-B[1,1])
    #   U=[1,0], V=[1,-1], W=[1,0]  (c11=m1+m4-..., c21=m2+m4)
    #   m4 contributes to C[0,0] and C[1,0]. W=[1,0]
    # m5 = A[1,1](B[1,0]-B[0,0])
    #   U=[0,1], V=[-1,1], W=[0,1]  (c11=...-m5, c12=m3+m5)
    #   m5 contributes to C[0,0] (with -1) and C[0,1] (with +1). W=[-1,1]
    # m6 = (A[0,0]+A[0,1])(B[1,1]-B[0,1])
    #   U=[1,0], V=[-1,1], W=[0,1]  (c22=...+m6, so W[1]=1)
    #   m6 contributes to C[1,1]. W=[0,1]
    # m7 = (A[0,1]+A[1,1])(B[0,0]-B[1,0])
    #   U=[0,1], V=[1,-1], W=[0,1]  (c11=...-m7, c22=...+m7)
    #   m7 contributes to C[0,0] (with -1) and C[1,1] (with +1). W=[-1,1]

    # Let me verify: c11 = m1+m4-m5-m7
    # m1: W[0]=1, m4: W[0]=1, m5: W[0]=-1, m7: W[0]=-1
    # Sum of W[0] contributions: 1+1-(-1)-(-1) = 1+1+1+1 = 4? That's not right.
    # I think the issue is that W_c[k] is the coefficient, and the sum is:
    # sum_c U_c[i] V_c[j] W_c[k]
    # For C[0,0]: sum_c U_c[0] V_c[j] W_c[0] summed over j
    # This is getting complex. Let me just verify numerically.

    # Let me define the Strassen components and verify
    def strassen_components():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        # 7 components, each with U[2], V[2], W[2]
        comps = [
            # m1
            ([1, 1], [1, 1], [1, 1]),
            # m2
            ([0, 1], [1, -1], [1, 0]),
            # m3
            ([1, -1], [0, 1], [0, 1]),
            # m4
            ([1, 0], [1, -1], [1, 0]),
            # m5
            ([0, 1], [-1, 1], [0, 1]),
            # m6
            ([1, 0], [-1, 1], [0, 1]),
            # m7
            ([0, 1], [1, -1], [0, 1]),
        ]
        return comps

    def verify_strassen():
        """Verify Strassen decomposition for 2x2x2."""
        comps = strassen_components()
        # Check that sum_c U_c[i] V_c[j] W_c[k] = 1 for all i,j,k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for U_c, V_c, W_c in comps:
                        s += U_c[i] * V_c[j] * W_c[k]
                    if s != 1:
                        print(f"Strassen verification failed at ({i},{j},{k}): {s}")
                        return False
        return True

    # Verify Strassen
    if not verify_strassen():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Build the 55-rank decomposition
    # For j in {0,1} (m1=2): 5 groups of 2 columns, each using Strassen (7 terms)
    # For j=2 (m2=1): schoolbook, 2*10=20 terms
    total_rank = 5 * 7 + 20  # 35 + 20 = 55

    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55

    col = 0
    strassen_comps = strassen_components()

    # Part 1: j in {0,1}, grouped into 5 pairs of k values
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        for c_idx in range(7):
            U_c, V_c, W_c = strassen_comps[c_idx]
            # For this group, the local k indices are 0,1 mapping to global k0, k1
            # U: rows are i*m+j for i in {0,1}, j in {0,1}
            #   U[i, j] = U_c[i] for the Strassen component
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_c[i]
            # V: rows are j*p+k for j in {0,1}, k in {0,1} (local)
            #   V[j, k_local] = V_c[j] * (1 if k_local maps to k0 or k1)
            #   Actually, V_c[j] is the coefficient for B[j, k_local]
            #   For k_local=0 -> global k=k0, for k_local=1 -> global k=k1
            for j in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j * p + k_global][col] = V_c[j]
            # W: rows are k*n+i for k in {0,1} (local), i in {0,1}
            #   W[k_local, i] = W_c[k_local] * ... wait
            #   Actually W_c[k] is the coefficient for C[i, k_local]
            #   For k_local=0 -> global k=k0, for k_local=1 -> global k=k1
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(2):
                    W[k_global * n + i][col] = W_c[k_local]
            col += 1

    # Part 2: j=2, schoolbook for 2x1x10
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == total_rank, f"Column count {col} != total_rank {total_rank}"

    # Verify the full decomposition
    def verify_full(U_mat, V_mat, W_mat, rnk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rnk):
                        s += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not verify_full(U, V, W, total_rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
