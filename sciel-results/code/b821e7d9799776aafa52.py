"""
Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy: Use the rank-8 Strassen decomposition for 2x2 matrix multiplication
as a building block, combined with Kronecker-product structure to handle the
rectangular 4x4 x 4x6 case.

The 4x4 matrix A is partitioned into 2x2 blocks of 2x2 submatrices:
    A = [[A11, A12],
         [A21, A22]]
Each Aij is 2x2.

The 4x6 matrix B is partitioned into 2x3 blocks of 2x2 submatrices:
    B = [[B11, B12, B13],
         [B21, B22, B23]]
Each Bij is 2x2.

The product C = A*B is 4x6, partitioned into 2x3 blocks of 2x2:
    C = [[C11, C12, C13],
         [C21, C22, C23]]

Each block Cij = sum over k of Aik * Bkj, where each term is a 2x2 * 2x2 product.

Using Strassen's 7-multiplication algorithm for each 2x2*2x2 product,
and sharing the intermediate Strassen terms across the 6 block products,
we can reduce the total rank.

For each of the 6 output blocks, we need a 2x2*2x2 product.
Strassen uses 7 multiplications per 2x2 product.

However, we can be smarter: the 4x4 * 4x6 product can be seen as:
- 4 rows of A times 4 rows of B, producing 4x6 output.

Alternative approach: Treat the 4x4*4x6 as two 4x4*4x2 products side by side,
or use the structure more directly.

Let's use a direct Kronecker/Strassen hybrid:

The tensor for (4x4)*(4x6) -> (4x6) has rank at most 4*4*6 = 96 (schoolbook).

Using Strassen on 2x2 blocks:
- Partition A into four 2x2 blocks: A11, A12, A21, A22
- Partition B into six 2x2 blocks: B11, B12, B13, B21, B22, B23

For each output block Cij (i in {1,2}, j in {1,2,3}):
  Cij = sum_{k=1}^{2} Aik * Bkj

So:
  C11 = A11*B11 + A12*B21
  C12 = A11*B12 + A12*B22
  C13 = A11*B13 + A12*B23
  C21 = A21*B11 + A22*B21
  C22 = A21*B12 + A22*B22
  C23 = A21*B13 + A22*B23

Each of these is a sum of two 2x2*2x2 products.

If we use Strassen for each 2x2*2x2, that's 7 mults each, 12 such products = 84 mults.
But we can share Strassen intermediate values.

Actually, let me think about this differently. The rank of the matrix multiplication
tensor [n,m,p] is denoted R(n,m,p). We know:
- R(2,2,2) = 7 (Strassen)
- R(4,4,4) <= 2^3 * 7 = 56 (recursive Strassen)
- R(4,4,6) <= ?

We can write 4x4*4x6 as two 4x4*4x3 products, or use other decompositions.

A clean approach: Use the fact that R(4,4,6) <= R(4,4,4) + R(4,4,2) or similar.
But actually, we can just build a decomposition explicitly.

Let me use a Kronecker product approach with Strassen:

Think of the 4x4*4x6 multiplication as operating on vectors of size 16 (A), 24 (B),
producing 24 (C). The tensor has 16*24*24 = 9216 entries, rank at most 96.

Using the block structure with Strassen:

For the 2x2 block Strassen, the 7 rank-1 terms for multiplying X (2x2) by Y (2x2)
to get Z (2x2) are:

M1 = (X11+X22)(Y11+Y22)
M2 = (X21+X22)Y11
M3 = X11(Y12-Y22)
M4 = X22(Y21-Y11)
M5 = (X11+X12)Y22
M6 = (X21-X11)(Y11+Y12)
M7 = (X12-X22)(Y21+Y22)

Z11 = M1 + M4 - M5 + M7
Z12 = M3 + M5
Z21 = M2 + M4
Z22 = M1 - M2 + M3 + M6

Each Mi is a product of a linear combination of X entries and a linear combination of Y entries.
In tensor decomposition terms, each Mi contributes one rank-1 term.

Now for our problem, we have 12 such 2x2*2x2 products (2x6 output blocks, each is sum of 2 products).
But we can group them.

Actually, let me just directly construct a decomposition using the Kronecker product of
smaller Strassen decompositions.

The 4x4*4x6 multiplication tensor can be decomposed as:
  T = sum over rank-1 terms

Using the approach: view 4 = 2*2, 6 = 2*3.

Let me use a simpler explicit construction. I'll build the decomposition by
treating the 4x4*4x6 as a combination of 2x2*2x2 Strassen products.

For each of the 12 sub-products (Aik * Bkj for i,k in {1,2}, j in {1,2,3}),
I apply Strassen. But I need to map the 2x2 sub-matrix entries to the full
16-dim (A), 24-dim (B), 24-dim (C) spaces.

Let me define the indexing:
A is 4x4, indexed A[row][col], row,col in 0..3.
As 2x2 blocks of 2x2:
  A11 = A[0:2,0:2], A12 = A[0:2,2:4]
  A21 = A[2:4,0:2], A22 = A[2:4,2:4]

B is 4x6, indexed B[row][col], row in 0..3, col in 0..5.
As 2x3 blocks of 2x2:
  B11 = B[0:2,0:2], B12 = B[0:2,2:4], B13 = B[0:2,4:6]
  B21 = B[2:4,0:2], B22 = B[2:4,2:4], B23 = B[2:4,4:6]

C is 4x6, similarly partitioned.

For the Strassen decomposition of a 2x2*2x2 product X*Y=Z:
The 7 rank-1 terms correspond to:
  M1: (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1]) -> contributes to Z[0,0] and Z[1,1]
  M2: (X[1,0]+X[1,1]) * Y[0,0] -> contributes to Z[1,0]
  M3: X[0,0] * (Y[0,1]-Y[1,1]) -> contributes to Z[0,1]
  M4: X[1,1] * (Y[1,0]-Y[0,0]) -> contributes to Z[1,0] and Z[0,0]
  M5: (X[0,0]+X[0,1]) * Y[1,1] -> contributes to Z[0,1] and Z[0,0]
  M6: (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1]) -> contributes to Z[1,1]
  M7: (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1]) -> contributes to Z[0,0] and Z[1,1]

In the tensor decomposition, each rank-1 term has:
- A coefficient vector in the A-space (16 entries)
- A coefficient vector in the B-space (24 entries)
- A coefficient vector in the C-space (24 entries)

For a single 2x2*2x2 Strassen term, the A-vector has nonzeros only in the 4 entries
of the 2x2 block X, the B-vector has nonzeros only in the 4 entries of Y, and the
C-vector has nonzeros in the 4 entries of Z.

So for the full 4x4*4x6, I'll create 12*7 = 84 rank-1 terms, one for each Strassen
multiplication in each of the 12 sub-products.

But wait, can I do better? The known upper bound for R(4,4,6) is not trivially 84.
Let me check: R(4,4,4) <= 56 (Strassen recursive). R(4,4,6) could be <= R(4,4,4)+R(4,4,2).
R(4,4,2) is the rank of multiplying 4x4 by 4x2 to get 4x2. This is at most 4*4*2=32 schoolbook,
but can be lower. Actually R(4,4,2) = R(2,4,4) by symmetry... no, matrix multiplication
tensor is not symmetric in all three indices.

Let me just go with the 84-term Strassen-based decomposition. That's already an improvement
over 96.

Actually, I realize I can potentially do better. Let me think about whether some Strassen
terms can be shared or combined.

For now, let me implement the 84-term decomposition and verify it's correct.

Let me code this up carefully.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # A is n x m = 4x4, B is m x p = 4x6, C is n x p = 4x6
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    # W index: k*n + i where k is column of C (0..5), i is row of C (0..3)
    # So W[k*n + i, c] corresponds to C[i, k]

    # We'll build the decomposition using Strassen for 2x2 blocks.
    # 
    # A blocks (2x2 each):
    #   A11: rows 0-1, cols 0-1  -> A indices 0,1,4,5 (row-major: A[r][c] = A[r*4+c])
    #   A12: rows 0-1, cols 2-3  -> A indices 2,3,6,7
    #   A21: rows 2-3, cols 0-1  -> A indices 8,9,12,13
    #   A22: rows 2-3, cols 2-3  -> A indices 10,11,14,15
    #
    # B blocks (2x2 each):
    #   B11: rows 0-1, cols 0-1  -> B indices 0,1,4,5 (B[r][c] = B[r*6+c])
    #   B12: rows 0-1, cols 2-3  -> B indices 2,3,8,9
    #   B13: rows 0-1, cols 4-5  -> B indices 10,11,16,17
    #   B21: rows 2-3, cols 0-1  -> B indices 12,13,18,19
    #   B22: rows 2-3, cols 2-3  -> B indices 14,15,20,21
    #   B23: rows 2-3, cols 4-5  -> B indices 22,23,28,29  -- wait, B is 4x6 so max index is 3*6+5=23
    #
    # Let me re-index. B is 4 rows x 6 cols. B[r][c] for r in 0..3, c in 0..5.
    # Flat index: r*6 + c.
    # B11: r=0,1 c=0,1 -> indices 0,1,6,7
    # B12: r=0,1 c=2,3 -> indices 2,3,8,9
    # B13: r=0,1 c=4,5 -> indices 4,5,10,11
    # B21: r=2,3 c=0,1 -> indices 12,13,18,19
    # B22: r=2,3 c=2,3 -> indices 14,15,20,21
    # B23: r=2,3 c=4,5 -> indices 16,17,22,23
    #
    # C is 4x6, C[r][c] for r in 0..3, c in 0..5.
    # W index: k*n + i = c*4 + r where c is column, r is row.
    # C11: r=0,1 c=0,1 -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C12: r=0,1 c=2,3 -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C13: r=0,1 c=4,5 -> W indices: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21
    # C21: r=2,3 c=0,1 -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C22: r=2,3 c=2,3 -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C23: r=2,3 c=4,5 -> W indices: 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23

    # Strassen 7 terms for 2x2 * 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]], Z = X*Y
    #
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    # For each rank-1 term, we need:
    #   u: vector in A-space (16 entries), the coefficients for the X part
    #   v: vector in B-space (24 entries), the coefficients for the Y part
    #   w: vector in C-space (24 entries), the coefficients for the Z part
    #
    # The rank-1 term contributes: u * v * w (outer product in tensor sense)
    # which means: for the tensor T[a,b,c], we add u[a]*v[b]*w[c]

    # Define the 7 Strassen terms as (u_coeffs, v_coeffs, w_coeffs) for a 2x2 block
    # where u has 4 entries (for x00,x01,x10,x11), v has 4 entries (for y00,y01,y10,y11),
    # and w has 4 entries (for z00,z01,z10,z11)

    strassen_terms = [
        # M1: (x00+x11)*(y00+y11) -> z00 += 1, z11 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (x10+x11)*y00 -> z10 += 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: x00*(y01-y11) -> z01 += 1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4: x11*(y10-y00) -> z10 += 1, z00 += 1
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5: (x00+x01)*y11 -> z01 += 1, z00 -= 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (x10-x00)*(y00+y01) -> z11 += 1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (x01-x11)*(y10+y11) -> z00 += 1, z11 += 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]

    # Now map each 2x2 sub-product to the full spaces.
    #
    # A block (i,k) where i in {0,1} (row block), k in {0,1} (col block of A = row block of B)
    #   A[i*2:(i+1)*2, k*2:(k+1)*2]
    #   Flat indices in A (4x4, row-major): (i*2+r)*4 + (k*2+c) for r,c in {0,1}
    #   So the 4 entries are at:
    #     x00: (i*2)*4 + k*2 = i*8 + k*2
    #     x01: (i*2)*4 + k*2 + 1 = i*8 + k*2 + 1
    #     x10: (i*2+1)*4 + k*2 = i*8 + k*2 + 4
    #     x11: (i*2+1)*4 + k*2 + 1 = i*8 + k*2 + 5
    #
    # B block (k,j) where k in {0,1} (row block), j in {0,1,2} (col block)
    #   B[k*2:(k+1)*2, j*2:(j+1)*2]
    #   Flat indices in B (4x6, row-major): (k*2+r)*6 + (j*2+c) for r,c in {0,1}
    #   So the 4 entries are at:
    #     y00: (k*2)*6 + j*2 = k*12 + j*2
    #     y01: (k*2)*6 + j*2 + 1 = k*12 + j*2 + 1
    #     y10: (k*2+1)*6 + j*2 = k*12 + j*2 + 6
    #     y11: (k*2+1)*6 + j*2 + 1 = k*12 + j*2 + 7
    #
    # C block (i,j) where i in {0,1} (row block), j in {0,1,2} (col block)
    #   C[i*2:(i+1)*2, j*2:(j+1)*2]
    #   W index: c*4 + r where c is col, r is row
    #   For C entry at (row=i*2+r, col=j*2+c), W index = (j*2+c)*4 + (i*2+r)
    #   So the 4 entries are at:
    #     z00: (j*2)*4 + i*2 = j*8 + i*2
    #     z01: (j*2+1)*4 + i*2 = j*8 + i*2 + 4
    #     z10: (j*2)*4 + i*2 + 1 = j*8 + i*2 + 1
    #     z11: (j*2+1)*4 + i*2 + 1 = j*8 + i*2 + 5

    rank = 12 * 7  # 84
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]

    col = 0
    for i in range(2):  # row block of A / row block of C
        for k in range(2):  # col block of A / row block of B
            for j in range(3):  # col block of B / col block of C
                # This sub-product is A[i,k] * B[k,j] contributing to C[i,j]
                a_base = i * 8 + k * 2
                b_base = k * 12 + j * 2
                c_base = j * 8 + i * 2

                # Indices for the 2x2 blocks
                a_idx = [a_base, a_base + 1, a_base + 4, a_base + 5]  # x00, x01, x10, x11
                b_idx = [b_base, b_base + 1, b_base + 6, b_base + 7]  # y00, y01, y10, y11
                c_idx = [c_base, c_base + 4, c_base + 1, c_base + 5]  # z00, z01, z10, z11

                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    for t in range(4):
                        if u_coeffs[t] != 0:
                            U[a_idx[t]][col] = u_coeffs[t]
                        if v_coeffs[t] != 0:
                            V[b_idx[t]][col] = v_coeffs[t]
                        if w_coeffs[t] != 0:
                            W[c_idx[t]][col] = w_coeffs[t]
                    col += 1

    return {"U": U, "V": V, "W": W}


def _verify():
    """Verify the decomposition is correct."""
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)

    n, m, p = 4, 4, 6
    # The tensor T[a, b, c] should equal sum over rank of U[a, r] * V[b, r] * W[c, r]
    # where a is flat index of A (n*m), b is flat index of B (m*p), c is flat index of C (p*n)
    # C index c = k*n + i where k is col, i is row, so C[i, k] = T[a, b, c]

    # Compute the decomposed tensor
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-way tensor. Let's check a few entries.

    # For random A, B, compute C = A @ B and check against the decomposition.
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B

    # From decomposition: C[i, k] = sum_{a, b} T[a, b, k*n+i] * A_flat[a] * B_flat[b]
    # where A_flat[a] = A[a//m, a%m], B_flat[b] = B[b//p, b%p]

    A_flat = A.flatten()  # row-major
    B_flat = B.flatten()  # row-major

    C_decomposed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            val = 0.0
            for r in range(U.shape[1]):
                # sum over a, b: U[a,r] * V[b,r] * A_flat[a] * B_flat[b]
                ua = U[:, r] * A_flat
                vb = V[:, r] * B_flat
                val += (ua.sum()) * (vb.sum()) * W[c_idx, r]
            C_decomposed[i, k] = val

    max_err = np.max(np.abs(C_expected - C_decomposed))
    print(f"Verification: max error = {max_err}")
    if max_err > 1e-10:
        raise ValueError(f"Decomposition is incorrect! Max error: {max_err}")
    print("Decomposition verified successfully.")
    print(f"Rank: {U.shape[1]}")


if __name__ == "__main__":
    _verify()
