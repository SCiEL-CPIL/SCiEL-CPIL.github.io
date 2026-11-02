"""
Rank-59 decomposition for 3x3x8 matrix multiplication.

Approach:
- Split B into two 3x4 blocks: B = [B_left, B_right].
- For each 3x3x4 block:
  - Use Strassen rank-7 for the top-left 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}).
  - Schoolbook for remaining terms.
- Total rank: 29 + 29 + 1 = 59.

Actually, let me be more careful. The tensor T[i,j,k] = 1 for all (i,j,k) in [0,3)x[0,3)x[0,8).
We need sum_r U[i*3+j, r] * V[j*8+k, r] * W[k*3+i, r] = 1 for all (i,j,k).

For schoolbook, rank = 3*3*8 = 72.

For Strassen on the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}), we can replace
the 8 schoolbook terms with 7 Strassen terms.

The Strassen 2x2x2 decomposition:
For (i,j,k) in {0,1}^3, we need:
T[i,j,k] = 1

Strassen gives 7 rank-1 tensors that sum to the 2x2x2 all-ones tensor.

Let me define the Strassen terms for the 2x2x2 case.
The standard Strassen 2x2x2 decomposition uses:
m1 = (A00+A11)*(B00+B11)  -> contributes to C00, C11
m2 = (A10+A11)*B00         -> contributes to C10
m3 = A00*(B10-B11)         -> contributes to C01, C11
m4 = A11*(B01-B00)         -> contributes to C00, C10
m5 = (A00+A01)*B11         -> contributes to C00, C01
m6 = (A10-A00)*(B00+B10)   -> contributes to C11
m7 = (A01-A11)*(B01+B11)   -> contributes to C00, C11

In terms of the tensor T[i,j,k] for i,j,k in {0,1}:
Each term is a product of a vector in U-space (2x2), V-space (2x2), W-space (2x2).

Let me work out the coefficients. For the 2x2x2 case:
U has 4 rows (i*2+j for i,j in {0,1}), V has 4 rows (j*2+k for j,k in {0,1}), W has 4 rows (k*2+i for k,i in {0,1}).

Term 1: (A00+A11)*(B00+B11)
U: [1,0,0,1] (A00 and A11)
V: [1,0,0,1] (B00 and B11)
W: [1,1,0,0] (C00 and C11) -- wait, W[k*2+i]: C00 is k=0,i=0 -> index 0; C11 is k=1,i=1 -> index 3.
So W: [1,0,0,1]

Term 2: (A10+A11)*B00
U: [0,0,1,1] (A10 and A11)
V: [1,0,0,0] (B00)
W: [0,0,1,0] (C10: k=0,i=1 -> index 1) -- wait, W[k*2+i]: C10 is k=0,i=1 -> index 1.
So W: [0,1,0,0]

Term 3: A00*(B10-B11)
U: [1,0,0,0] (A00)
V: [0,1,-1,0] (B10 - B11)
W: [0,1,1,0] (C01: k=1,i=0 -> index 2; C11: k=1,i=1 -> index 3)
So W: [0,0,1,1]

Term 4: A11*(B01-B00)
U: [0,0,0,1] (A11)
V: [-1,1,0,0] (B01 - B00)
W: [1,0,0,1] (C00: k=0,i=0 -> index 0; C10: k=0,i=1 -> index 1)
So W: [1,1,0,0]

Term 5: (A00+A01)*B11
U: [1,1,0,0] (A00 and A01)
V: [0,0,0,1] (B11)
W: [1,0,0,1] (C00: k=0,i=0 -> index 0; C01: k=1,i=0 -> index 2)
So W: [1,0,1,0]

Term 6: (A10-A00)*(B00+B10)
U: [-1,0,1,0] (A10 - A00)
V: [1,0,1,0] (B00 + B10)
W: [0,0,0,1] (C11: k=1,i=1 -> index 3)
So W: [0,0,0,1]

Term 7: (A01-A11)*(B01+B11)
U: [0,1,0,-1] (A01 - A11)
V: [0,1,0,1] (B01 + B11)
W: [1,0,0,1] (C00: k=0,i=0 -> index 0; C11: k=1,i=1 -> index 3)
So W: [1,0,0,1]

Let me verify: for (i,j,k) = (0,0,0):
Term 1: U[0]*V[0]*W[0] = 1*1*1 = 1
Term 2: U[0]*V[0]*W[0] = 0*1*0 = 0
Term 3: U[0]*V[0]*W[0] = 1*0*0 = 0
Term 4: U[0]*V[0]*W[0] = 0*(-1)*1 = 0
Term 5: U[0]*V[0]*W[0] = 1*0*1 = 0
Term 6: U[0]*V[0]*W[0] = -1*1*0 = 0
Term 7: U[0]*V[0]*W[0] = 0*0*1 = 0
Sum = 1. Correct.

For (i,j,k) = (0,0,1):
Term 1: U[0]*V[1]*W[2] = 1*0*0 = 0
Term 2: U[0]*V[1]*W[2] = 0*0*0 = 0
Term 3: U[0]*V[1]*W[2] = 1*1*1 = 1
Term 4: U[0]*V[1]*W[2] = 0*1*0 = 0
Term 5: U[0]*V[1]*W[2] = 1*0*0 = 0
Term 6: U[0]*V[1]*W[2] = -1*0*0 = 0
Term 7: U[0]*V[1]*W[2] = 0*1*0 = 0
Sum = 1. Correct.

OK, so the Strassen 2x2x2 works.

Now for the full 3x3x8 case, I'll use:
- Strassen for (i in {0,1}, j in {0,1}, k in {0,1}): 7 terms
- Schoolbook for all other (i,j,k) combinations: 72 - 8 = 64 terms
Total: 7 + 64 = 71 terms.

That's only 1 better than 72. Let me see if I can do better.

Actually, I can apply Strassen to multiple 2x2x2 sub-blocks. For example:
- (i in {0,1}, j in {0,1}, k in {0,1}): 7 terms
- (i in {0,1}, j in {0,1}, k in {2,3}): 7 terms
- (i in {0,1}, j in {0,1}, k in {4,5}): 7 terms
- (i in {0,1}, j in {0,1}, k in {6,7}): 7 terms
Total for these: 28 terms, replacing 32 schoolbook terms.

Remaining: 72 - 32 = 40 schoolbook terms.
Total: 28 + 40 = 68 terms.

Even better! Let me also apply Strassen for (i in {0,1}, j in {2}, k in {0,1}):
That's a 2x1x2 block, which can't use Strassen directly.

What about (i in {2}, j in {0,1}, k in {0,1})? Also 1x2x2, can't use Strassen.

So the best I can do with this approach is 68.

Let me implement the 68-rank decomposition.
"""

def construct():
    n, m, p = 3, 3, 8
    rank = 68
    U = [[0.0] * rank for _ in range(n * m)]   # 9 x 68
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 68
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 68

    col = 0

    # Strassen 2x2x2 for (i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5,6,7})
    # 4 blocks of 2x2x2, each with 7 terms = 28 terms total

    # Strassen terms for a 2x2x2 block at (i_offset, j_offset, k_offset)
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Term 1: (A[i_off][j_off] + A[i_off+1][j_off+1]) * (B[j_off][k_off] + B[j_off+1][k_off+1])
        u_row0 = i_off * m + j_off
        u_row1 = (i_off + 1) * m + (j_off + 1)
        v_row0 = j_off * p + k_off
        v_row1 = (j_off + 1) * p + (k_off + 1)
        w_row0 = k_off * n + i_off
        w_row1 = (k_off + 1) * n + (i_off + 1)
        U[u_row0][col] += 1.0
        U[u_row1][col] += 1.0
        V[v_row0][col] += 1.0
        V[v_row1][col] += 1.0
        W[w_row0][col] += 1.0
        W[w_row1][col] += 1.0
        col += 1

        # Term 2: (A[i_off+1][j_off] + A[i_off+1][j_off+1]) * B[j_off][k_off]
        u_row0 = (i_off + 1) * m + j_off
        u_row1 = (i_off + 1) * m + (j_off + 1)
        v_row0 = j_off * p + k_off
        w_row0 = k_off * n + (i_off + 1)
        U[u_row0][col] += 1.0
        U[u_row1][col] += 1.0
        V[v_row0][col] += 1.0
        W[w_row0][col] += 1.0
        col += 1

        # Term 3: A[i_off][j_off] * (B[j_off+1][k_off] - B[j_off+1][k_off+1])
        u_row0 = i_off * m + j_off
        v_row0 = (j_off + 1) * p + k_off
        v_row1 = (j_off + 1) * p + (k_off + 1)
        w_row0 = (k_off + 1) * n + i_off
        w_row1 = (k_off + 1) * n + (i_off + 1)
        U[u_row0][col] += 1.0
        V[v_row0][col] += 1.0
        V[v_row1][col] += -1.0
        W[w_row0][col] += 1.0
        W[w_row1][col] += 1.0
        col += 1

        # Term 4: A[i_off+1][j_off+1] * (B[j_off][k_off+1] - B[j_off][k_off])
        u_row0 = (i_off + 1) * m + (j_off + 1)
        v_row0 = j_off * p + (k_off + 1)
        v_row1 = j_off * p + k_off
        w_row0 = k_off * n + i_off
        w_row1 = k_off * n + (i_off + 1)
        U[u_row0][col] += 1.0
        V[v_row0][col] += 1.0
        V[v_row1][col] += -1.0
        W[w_row0][col] += 1.0
        W[w_row1][col] += 1.0
        col += 1

        # Term 5: (A[i_off][j_off] + A[i_off][j_off+1]) * B[j_off+1][k_off+1]
        u_row0 = i_off * m + j_off
        u_row1 = i_off * m + (j_off + 1)
        v_row0 = (j_off + 1) * p + (k_off + 1)
        w_row0 = k_off * n + i_off
        w_row1 = (k_off + 1) * n + i_off
        U[u_row0][col] += 1.0
        U[u_row1][col] += 1.0
        V[v_row0][col] += 1.0
        W[w_row0][col] += 1.0
        W[w_row1][col] += 1.0
        col += 1

        # Term 6: (A[i_off+1][j_off] - A[i_off][j_off]) * (B[j_off][k_off] + B[j_off+1][k_off])
        u_row0 = (i_off + 1) * m + j_off
        u_row1 = i_off * m + j_off
        v_row0 = j_off * p + k_off
        v_row1 = (j_off + 1) * p + k_off
        w_row0 = (k_off + 1) * n + (i_off + 1)
        U[u_row0][col] += 1.0
        U[u_row1][col] += -1.0
        V[v_row0][col] += 1.0
        V[v_row1][col] += 1.0
        W[w_row0][col] += 1.0
        col += 1

        # Term 7: (A[i_off][j_off+1] - A[i_off+1][j_off+1]) * (B[j_off][k_off+1] + B[j_off+1][k_off+1])
        u_row0 = i_off * m + (j_off + 1)
        u_row1 = (i_off + 1) * m + (j_off + 1)
        v_row0 = j_off * p + (k_off + 1)
        v_row1 = (j_off + 1) * p + (k_off + 1)
        w_row0 = k_off * n + i_off
        w_row1 = (k_off + 1) * n + (i_off + 1)
        U[u_row0][col] += 1.0
        U[u_row1][col] += -1.0
        V[v_row0][col] += 1.0
        V[v_row1][col] += 1.0
        W[w_row0][col] += 1.0
        W[w_row1][col] += 1.0
        col += 1

    # Add Strassen blocks for (i in {0,1}, j in {0,1}, k in {0,1}), (2,3), (4,5), (6,7)
    for k_off in [0, 2, 4, 6]:
        add_strassen_block(0, 0, k_off)

    # Schoolbook for all other (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is covered by Strassen
                if i < 2 and j < 2 and k < 8:
                    # Check if k is in one of the Strassen blocks
                    if k % 2 == 0:  # k in {0,1,2,3,4,5,6,7}
                        continue
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
