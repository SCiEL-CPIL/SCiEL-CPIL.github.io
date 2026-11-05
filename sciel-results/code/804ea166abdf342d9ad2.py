"""Matrix multiplication 4x5x5 - algebraic elimination via structured ansatz.

Strategy: Instead of recursive Strassen, use a direct algebraic elimination
approach. We model the tensor decomposition as a system of bilinear equations
and use a structured ansatz to find rank-minimal solutions.

Key insight: The 4x5x5 multiplication tensor has 100 entries (4*5*5). The
schoolbook decomposition uses rank 100. We can potentially reduce this by
finding linear dependencies.

New approach: Use a "split" decomposition where we partition the problem
into overlapping sub-problems and use Strassen on each, then carefully
combine to avoid double-counting.

Specifically:
- Partition A into two 4x3 and 4x2 parts (columns 0-2 and 3-4)
- Partition B into corresponding 3x5 and 2x5 parts
- C = A_left * B_left + A_right * B_right
- For the 4x3x5 part: use Strassen-like on 4x3x3 (rank ~32) + 4x3x2 (rank 24)
- For the 4x2x5 part: schoolbook rank 40
- Total: ~96? Not better.

Let me try a different structural approach: use the fact that we can
embed 4x5x5 into a 5x5x5 problem and use the known rank-19 decomposition,
then restrict.

Actually, the cleanest structural change: use a DIRECT linear algebra
approach. Set up the tensor as a 3-tensor and use rank minimization via
a structured ansatz with rational coefficients.

The most promising structural change: Use the "border rank" insight.
For 4x5x5, we can potentially achieve rank lower than the schoolbook 100
by using the structure of the multiplication tensor.

Let me try: decompose as C = A*B where we write:
  A = [A_1 | A_2] where A_1 is 4x4, A_2 is 4x1
  B = [B_1; B_2] where B_1 is 4x5, B_2 is 1x5
  
  C = A_1 * B_1 + A_2 * B_2
  
  A_1*B_1 is 4x4x5 (rank 80 schoolbook, can use Strassen on 4x4x4 part)
  A_2*B_2 is 4x1x5 (rank 20, outer product)
  
  For A_1*B_1: use Strassen on the 4x4x4 part (rank 49) + schoolbook for 4x4x1 (rank 16)
  Total: 49 + 16 + 20 = 85

This is the same as before. Let me try a fundamentally different approach.

NEW STRUCTURAL APPROACH: Use a "tensor train" or "MPS" style decomposition
where we factor the tensor through intermediate indices, potentially
achieving lower rank by exploiting the rectangular structure.

Actually, let me try the most direct approach: use a randomized algebraic
method to find a lower-rank decomposition. Set up the equations and use
a structured search over rational coefficients.

For the 4x5x5 case, the tensor T[i,j,k] = 1 for all i in 0..3, j in 0..4, k in 0..4.
We want T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r where:
  T[i,j,k] = sum_r u_r[i*5+j] * v_r[j*5+k] * w_r[k*4+i]

This is a system of 100 equations in 20R + 25R + 20R = 65R unknowns (with R rank).
For R < 100, we have more equations than unknowns, so we need the right structure.

Let me try a direct construction using the "lattice" method from algebraic complexity.
"""
import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # Strategy: Use a structured algebraic approach.
    # 
    # The key structural insight: For 4x5x5, we can use a "hybrid" approach
    # where we apply Strassen to a 4x4x4 sub-block and handle the remaining
    # parts optimally.
    #
    # But let's try something different: use a DIRECT rank minimization
    # by exploiting the rectangular structure.
    #
    # Partition: C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # 
    # Write j = j1 + 4*j2 where j1 in {0,1,2,3}, j2 in {0,1}
    # C[i,k] = sum_{j1=0}^{3} A[i,j1] * B[j1,k] + sum_{j1=0}^{3} A[i,4+j1] * B[4+j1,k]
    #         = C_left[i,k] + C_right[i,k]
    #
    # C_left: 4x4x5 multiplication (A[:,0:4] * B[0:4,:])
    # C_right: 4x4x5 multiplication (A[:,4:8] * B[4:8,:]) - but A only has 5 cols!
    #
    # Wait, A is 4x5, so A[i,j] for j in 0..4.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # Split: j in {0,1,2,3} and j=4
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    #
    # Part 1: 4x4x5 (rank 80 schoolbook, or 49+16=65 with Strassen)
    # Part 2: 4x1x5 (rank 20, outer product)
    # Total: 65 + 20 = 85
    #
    # Can we do better on the 4x4x5 part?
    # 4x4x5: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # This is like multiplying a 4x4 by a 4x5.
    # We can write B = [B_4x4 | B_4x1]
    # C = A*B_4x4 + A*B_4x1
    # A*B_4x4: 4x4x4 (Strassen rank 49)
    # A*B_4x1: 4x4x1 (rank 16, mat-vec)
    # Total: 49 + 16 = 65
    #
    # So total 85. Same as before.
    #
    # Let me try a completely different structural approach:
    # Use the "Kronecker" structure. 4 = 2*2, 5 = 2+3.
    # 
    # Write A as a 2x2 block matrix of 2x(2+3) blocks? This gets messy.
    #
    # Alternative: Use a "winograd" style approach.
    # For the 4x4x4 part, instead of Strassen (rank 49), can we do better?
    # The known lower bound for 4x4x4 is 49 (Strassen is optimal for 4x4x4? No,
    # the lower bound is 49 but Strassen achieves 49, so it's optimal).
    #
    # Actually, for 4x4x4, Strassen gives rank 49, and this is known to be
    # the minimum (lower bound is 49). So 49 is optimal for the 4x4x4 part.
    #
    # The 4x4x1 part has rank 16 (schoolbook). Can we reduce this?
    # 4x4x1: C[i,0] = sum_j A[i,j]*B[j,0]. This is a mat-vec product.
    # The rank of the 4x4x1 mat-vec tensor is 16 (it's a "slice" of the
    # multiplication tensor). No reduction possible.
    #
    # The 4x1x5 part has rank 20. This is an outer product, rank = 4*5 = 20.
    # No reduction possible.
    #
    # So 85 seems to be the natural limit with this partitioning.
    #
    # Let me try a DIFFERENT partitioning:
    # C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k]
    #
    # What if we group differently?
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k] + sum_{j=3}^{4} A[i,j]*B[j,k]
    # Part 1: 4x3x5 (rank 60 schoolbook)
    # Part 2: 4x2x5 (rank 40 schoolbook)
    # Total: 100. Worse.
    #
    # What about:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    # Part 1: 4x2x5 (rank 40)
    # Part 2: 4x3x5 (rank 60)
    # Total: 100. Same.
    #
    # The 4x4x5 + 4x1x5 split is best: 65 + 20 = 85.
    #
    # Now, can we reduce the 4x4x5 part further?
    # 4x4x5: We used Strassen on 4x4x4 (49) + schoolbook on 4x4x1 (16) = 65.
    # What if we use a different decomposition?
    #
    # 4x4x5 can be seen as: multiply 4x4 by 4x5.
    # Write the 4x5 as [4x4 | 4x1].
    # C = A*[B4 | B1] = A*B4 + A*B1
    # A*B4: 4x4x4, rank 49 (Strassen, optimal)
    # A*B1: 4x4x1, rank 16
    # Total: 65.
    #
    # Alternative: Write the 4x4 A as [2x4; 2x4] and use Strassen on the 2x2
    # block structure?
    # A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 2x4
    # B = [B_left | B_right] where B_left is 4x2, B_right is 4x3
    # This doesn't directly help.
    #
    # Let me try yet another approach: use a "tensor product" decomposition.
    # 4 = 2*2, so we can write the 4x4x4 as a tensor product of two 2x2x2's.
    # But that gives rank 7*7 = 49, same as Strassen.
    #
    # I think 85 is the best we can achieve with this type of decomposition.
    # Let me implement it cleanly and verify.

    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0

    def add_rank1(u_entries, v_entries, w_entries, coeff=1):
        nonlocal rank
        for ui, uv in u_entries.items():
            if uv != 0:
                U_dict[(ui, rank)] = U_dict.get((ui, rank), 0) + uv * coeff
        for vi, vv in v_entries.items():
            if vv != 0:
                V_dict[(vi, rank)] = V_dict.get((vi, rank), 0) + vv * coeff
        for wi, wv in w_entries.items():
            if wv != 0:
                W_dict[(wi, rank)] = W_dict.get((wi, rank), 0) + wv * coeff
        rank += 1

    def idx_A(i, j):
        return i * 5 + j

    def idx_B(j, k):
        return j * 5 + k

    def idx_C(i, k):
        return k * 4 + i

    # Strassen 2x2x2: 7 rank-1 terms
    def strassen_2x2(X_u, Y_v, Z_w, coeff=1):
        # M1: (X00+X11)*(Y00+Y11) -> Z00+Z11
        add_rank1({X_u[0][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)
        # M2: (X10+X11)*Y00 -> Z10
        add_rank1({X_u[1][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1},
                  {Z_w[1][0]: 1}, coeff)
        # M3: X00*(Y01-Y11) -> Z01
        add_rank1({X_u[0][0]: 1},
                  {Y_v[0][1]: 1, Y_v[1][1]: -1},
                  {Z_w[0][1]: 1}, coeff)
        # M4: X11*(Y10-Y00) -> Z10
        add_rank1({X_u[1][1]: 1},
                  {Y_v[1][0]: 1, Y_v[0][0]: -1},
                  {Z_w[1][0]: 1}, coeff)
        # M5: (X00+X01)*Y11 -> Z01
        add_rank1({X_u[0][0]: 1, X_u[0][1]: 1},
                  {Y_v[1][1]: 1},
                  {Z_w[0][1]: 1}, coeff)
        # M6: (X10-X00)*(Y00+Y01) -> Z11
        add_rank1({X_u[1][0]: 1, X_u[0][0]: -1},
                  {Y_v[0][0]: 1, Y_v[0][1]: 1},
                  {Z_w[1][1]: 1}, coeff)
        # M7: (X01-X11)*(Y10+Y11) -> Z00+Z11
        add_rank1({X_u[0][1]: 1, X_u[1][1]: -1},
                  {Y_v[1][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)

    # 4x4x4 Strassen: 7 * 7 = 49 terms
    # Using the "combined" approach for the 7 outer products

    def strassen_2x2_combined(X_u, Y_v, Z_w, coeff=1):
        # X_u, Y_v, Z_w are 2x2 lists of lists of (index, coeff)
        def merge(*lists):
            d = {}
            for lst in lists:
                for idx, c in lst:
                    d[idx] = d.get(idx, 0) + c
            return d

        # M1
        u1 = merge(X_u[0][0], X_u[1][1])
        v1 = merge(Y_v[0][0], Y_v[1][1])
        w1 = merge(Z_w[0][0], Z_w[1][1])
        add_rank1(u1, v1, w1, coeff)
        # M2
        u2 = merge(X_u[1][0], X_u[1][1])
        v2 = merge(Y_v[0][0])
        w2 = merge(Z_w[1][0])
        add_rank1(u2, v2, w2, coeff)
        # M3
        u3 = merge(X_u[0][0])
        v3 = {}
        for idx, c in Y_v[0][1]:
            v3[idx] = v3.get(idx, 0) + c
        for idx, c in Y_v[1][1]:
            v3[idx] = v3.get(idx, 0) - c
        w3 = merge(Z_w[0][1])
        add_rank1(u3, v3, w3, coeff)
        # M4
        u4 = merge(X_u[1][1])
        v4 = {}
        for idx, c in Y_v[1][0]:
            v4[idx] = v4.get(idx, 0) + c
        for idx, c in Y_v[0][0]:
            v4[idx] = v4.get(idx, 0) - c
        w4 = merge(Z_w[1][0])
        add_rank1(u4, v4, w4, coeff)
        # M5
        u5 = merge(X_u[0][0], X_u[0][1])
        v5 = merge(Y_v[1][1])
        w5 = merge(Z_w[0][1])
        add_rank1(u5, v5, w5, coeff)
        # M6
        u6 = {}
        for idx, c in X_u[1][0]:
            u6[idx] = u6.get(idx, 0) + c
        for idx, c in X_u[0][0]:
            u6[idx] = u6.get(idx, 0) - c
        v6 = merge(Y_v[0][0], Y_v[0][1])
        w6 = merge(Z_w[1][1])
        add_rank1(u6, v6, w6, coeff)
        # M7
        u7 = {}
        for idx, c in X_u[0][1]:
            u7[idx] = u7.get(idx, 0) + c
        for idx, c in X_u[1][1]:
            u7[idx] = u7.get(idx, 0) - c
        v7 = merge(Y_v[1][0], Y_v[1][1])
        w7 = merge(Z_w[0][0], Z_w[1][1])
        add_rank1(u7, v7, w7, coeff)

    def simple_block(idx_func, r1, r2, c1, c2):
        return [[[(idx_func(r1+i, c1+j), 1)] for j in range(2)] for i in range(2)]

    def combined_block(b1, b2, sign2=1):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                entries = list(b1[i][j])
                for idx, c in b2[i][j]:
                    entries.append((idx, sign2 * c))
                row.append(entries)
            result.append(row)
        return result

    # 4x4x4 Strassen
    A00 = simple_block(idx_A, 0, 1, 0, 1)
    A01 = simple_block(idx_A, 0, 1, 2, 3)
    A10 = simple_block(idx_A, 2, 3, 0, 1)
    A11 = simple_block(idx_A, 2, 3, 2, 3)

    B00 = simple_block(idx_B, 0, 1, 0, 1)
    B01 = simple_block(idx_B, 0, 1, 2, 3)
    B10 = simple_block(idx_B, 2, 3, 0, 1)
    B11 = simple_block(idx_B, 2, 3, 2, 3)

    C00 = simple_block(idx_C, 0, 1, 0, 1)
    C01 = simple_block(idx_C, 0, 1, 2, 3)
    C10 = simple_block(idx_C, 2, 3, 0, 1)
    C11 = simple_block(idx_C, 2, 3, 2, 3)

    strassen_2x2_combined(combined_block(A00, A11), combined_block(B00, B11), combined_block(C00, C11), 1)
    strassen_2x2_combined(combined_block(A10, A11), B00, C10, 1)
    strassen_2x2_combined(A00, combined_block(B01, B11, -1), C01, 1)
    strassen_2x2_combined(A11, combined_block(B10, B00, -1), C11, 1)
    strassen_2x2_combined(combined_block(A00, A01), B11, C01, 1)
    strassen_2x2_combined(combined_block(A10, A00, -1), combined_block(B00, B01), C10, 1)
    strassen_2x2_combined(combined_block(A01, A11, -1), combined_block(B10, B11), C11, 1)

    # 4x4x1 schoolbook: A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    for i in range(4):
        for j in range(4):
            add_rank1({idx_A(i,j): 1}, {idx_B(j,4): 1}, {idx_C(i,4): 1}, 1)

    # 4x1x5 schoolbook: A[0:4,4] * B[4,0:5] -> C[0:4,0:5]
    for i in range(4):
        for k in range(5):
            add_rank1({idx_A(i,4): 1}, {idx_B(4,k): 1}, {idx_C(i,k): 1}, 1)

    # Total rank: 49 + 16 + 20 = 85

    # Convert to arrays
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for (row, col), val in U_dict.items():
        U[row, col] = val
    for (row, col), val in V_dict.items():
        V[row, col] = val
    for (row, col), val in W_dict.items():
        W[row, col] = val

    # Verification
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_recon[i, k] += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]

        if not np.allclose(C_recon, C_expected, atol=1e-10):
            # Fallback to schoolbook
            U_dict = {}
            V_dict = {}
            W_dict = {}
            rank = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_dict[(i*m+j, rank)] = 1
                        V_dict[(j*p+k, rank)] = 1
                        W_dict[(k*n+i, rank)] = 1
                        rank += 1

            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for (row, col), val in U_dict.items():
                U[row, col] = val
            for (row, col), val in V_dict.items():
                V[row, col] = val
            for (row, col), val in W_dict.items():
                W[row, col] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
