"""Matrix multiplication 6x6x6 via 2x2 block decomposition of 3x3 blocks
with schoolbook rank-9 decomposition per block product.
Total rank: 4 * 9 = 36 (block-level) * ... wait, let me re-read.

Actually: 6x6 = 2x2 grid of 3x3 blocks.
C_{ij} = sum_k A_{ik} B_{kj} where each is a 3x3 matrix product.
So C has 4 blocks (top-left, top-right, bottom-left, bottom-right),
and for each output block position, we need one 3x3 matrix multiplication.

Each 3x3 multiplication has rank 9 (schoolbook).
So total rank = 4 * 9 = 36.

Wait, that's much better than 216! Let me think about this more carefully.

For 6x6 matrix multiplication:
C = A * B where A, B are 6x6.

Partition into 2x2 blocks of size 3x3:
A = [[A00, A01], [A10, A11]]
B = [[B00, B01], [B10, B11]]
C = [[C00, C01], [C10, C11]]

Where:
C00 = A00*B00 + A01*B10
C01 = A00*B01 + A01*B11
C10 = A10*B00 + A11*B10
C11 = A10*B01 + A11*B11

Each C block is a 3x3 matrix, and each A_ij, B_ij is a 3x3 matrix.
The product A00*B00 is a 3x3 matrix multiplication, which has a rank-9 tensor decomposition (schoolbook).

So we need:
- 8 individual 3x3 matrix multiplications: A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11
- Each has rank 9
- Total rank = 8 * 9 = 72

This is better than 216! Let me implement this.

The U matrix has shape (n*m, rank) = (36, rank)
The V matrix has shape (m*p, rank) = (36, rank)
The W matrix has shape (p*n, rank) = (36, rank)

For the schoolbook 3x3 decomposition:
For each pair of 3x3 matrices (X, Y) producing Z:
Z[i][j] = sum_k X[i][k] * Y[k][j]

The rank-9 decomposition has terms indexed by (i, k, j) where i, j, k are in 0..2.

For a single 3x3 multiplication, the rank-9 terms are:
For each (i, k, j):
  U part: X[i][k] -> index i*3+k in the 9x9 input
  V part: Y[k][j] -> index k*3+j in the 9x9 input  
  W part: Z[i][j] -> index j*3+i in the 9x9 output (following W[k*n+i,r] convention)

Wait, the task says: "The output factor follows the official trace convention W[k*n+i,r]."
So W[r][c] = sum over k of V[k*3 + something] * U[something + r]... 

Let me be more careful. In the standard formulation:
For matrix multiplication of (n x m) * (m x p) -> (n x p):
T[i,j,k] = A[i,k] * B[k,j] where i in 0..n-1, k in 0..m-1, j in 0..p-1.

The tensor decomposition gives:
C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] ... no wait.

Actually in the standard notation for matrix multiplication tensor:
We have input A of size n x m, input B of size m x p, output C of size n x p.
The multiplication tensor has trilinear form: sum_{i,k,j} A[i,k] * B[k,j] * C[i,j].

The rank-r decomposition means:
C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
... no, that's not right either.

Let me use the standard formulation. For matrix multiplication (n,m,p):
- U has shape (n*m, rank): U[a, r] where a = i*m + k
- V has shape (m*p, rank): V[b, r] where b = k*p + j  
- W has shape (p*n, rank): W[c, r] where c = j*n + i

And the condition is: for all i, k, j:
sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
For all other (i,k,j) combinations that don't correspond to valid matrix mult, it should be 0.

Actually, the standard check is:
C[i,j] = sum_{k} A[i,k] * B[k,j]

In the tensor form:
sum_{i,k,j} A[i,k] B[k,j] C^*[i,j] where C^* is the dual.

The rank-r decomposition: T = sum_{r=0}^{rank-1} u_r ⊗ v_r ⊗ w_r

Where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.

And: T[i,k;j,l] = sum_r u_r[i,k] * v_r[j,l] * w_r[?]... 

I'm getting confused with the index conventions. Let me just look at the current program:

Current schoolbook for (n,m,p) = (6,6,6), rank = 216:
For each (i, j, k):
  U[i*m + j][column] = 1  => U[i*6+j, col] = 1
  V[j*p + k][column] = 1  => V[j*6+k, col] = 1
  W[k*n + i][column] = 1  => W[k*6+i, col] = 1

So the convention is:
- U index: i*m + j (row i, col j of A)
- V index: j*p + k (row j, col k of B)
- W index: k*n + i (row k, col i of... C transposed? or C indexed as C[k][i]?)

Wait, the task says "W[k*n+i,r]" which matches. So W's first index is k*n+i, meaning it's indexed by (k, i) where k is the column of B / row of C's dual and i is the row of A / column of C's dual.

Actually, I think the convention is:
- A is n x m: entries A[i,j] for i in [0,n), j in [0,m)
- B is m x p: entries B[j,k] for j in [0,m), k in [0,p)
- C is n x p: entries C[i,k] for i in [0,n), k in [0,p)
- C[i,k] = sum_j A[i,j] * B[j,k]

Tensor: sum_{i,j,k} A[i,j] * B[j,k] * C^*[i,k]

Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
where u_r has indices (i,j), v_r has indices (j,k), w_r has indices (i,k)

But the program says:
- U has shape (n*m, rank): indexed by i*m+j
- V has shape (m*p, rank): indexed by j*p+k
- W has shape (p*n, rank): indexed by k*n+i

So w_r has indices (k,i) not (i,k). That means W is indexed as W[k*n+i] which is (k, i) where k is the column index of C and i is the row index of C, but reversed in the flattening.

So C^*[i,k] corresponds to W[k*n+i]. This is just a reindexing.

OK so for my 6x6 = 2x2 blocks of 3x3:

Block A00 is rows 0-2, cols 0-2 of A
Block A01 is rows 0-2, cols 3-5 of A
Block A10 is rows 3-5, cols 0-2 of A
Block A11 is rows 3-5, cols 3-5 of A

Similarly for B and C.

C00 = A00 * B00 + A01 * B10 (all 3x3 matrix products)
C01 = A00 * B01 + A01 * B11
C10 = A10 * B00 + A11 * B10
C11 = A10 * B01 + A11 * B11

8 products total, each 3x3 with rank 9, so total rank 72.

For each 3x3 product, say X * Y = Z where X is rows x0-x2, cols y0-y2 of the full 6x6:

Schoolbook rank-9 for 3x3:
For each (i_local, k_local, j_local) where each is in 0..2:
  The local row i_local corresponds to global row x0 + i_local
  The local col of X, k_local corresponds to global col y0 + k_local
  The local row of Y, k_local corresponds to global row y1 + k_local (where y1 is start row of Y block)
  The local col of Y, j_local corresponds to global col z0 + j_local
  The local row of Z, i_local corresponds to global row x0 + i_local
  The local col of Z, j_local corresponds to global col z0 + j_local

U index: global_row_of_A * 6 + global_col_of_A
V index: global_row_of_B * 6 + global_col_of_B
W index: global_col_of_C * 6 + global_row_of_C

Let me define the 8 products:
1. A00 * B00 -> C00
2. A01 * B10 -> C00
3. A00 * B01 -> C01
4. A01 * B11 -> C01
5. A10 * B00 -> C10
6. A11 * B10 -> C10
7. A10 * B01 -> C11
8. A11 * B11 -> C11

For product (Ablock, Bblock, Cblock) where:
  Ablock starts at row ar, col ac
  Bblock starts at row br, col bc
  Cblock starts at row cr, col cc

For each (i, k, j) in range(3)^3:
  global A index: (ar+i) * 6 + (ac+k)
  global B index: (br+k) * 6 + (bc+j)
  global C index: (cc+j) * 6 + (cr+i)  [since W[k*n+i] means col*k + row... wait]

W index is k*n + i where k is the column of C and i is the row of C.
So W index = global_col_C * 6 + global_row_C = (cc+j) * 6 + (cr+i)

Let me code this up.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen decomposition at the 2x2 block level (each entry is a 3x3 matrix)
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    # Strassen 7 virtual products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    # Each term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # output_contrib[term_idx] = list of (output_block_idx, coeff)
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    # Total rank: 7 Strassen terms * 27 schoolbook terms per 3x3 product = 189
    rank = 7 * 27

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):   # local row in 3x3 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a

                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b

                    # W contributions: C entries (W index = global_col * n + global_row)
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
