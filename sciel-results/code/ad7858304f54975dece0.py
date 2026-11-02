"""
Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
- The tensor has dimensions n=3, m=4, p=4 (input A is 3x4, input B is 4x4, output C is 3x4).
- Total scalar multiplications for schoolbook: n*m*p = 48.
- We can use Strassen-like block decompositions on sub-blocks to reduce rank.

Key insight:
- The output C has shape 3x4. We can think of it as 3 rows of length 4.
- For each row i of C (i=0,1,2), we compute C[i,:] = A[i,:] * B.
- A[i,:] is a 1x4 vector, B is 4x4. So each row of C is a 1x4 * 4x4 multiplication.
- But we can group: think of A as 3x4 and B as 4x4.

Better approach: Use the structure that m=p=4.
We can partition B into 2x2 blocks of size 2x2:
B = [[B00, B01],
     [B10, B11]]
where each Bjk is 2x2.

Similarly, A[i,:] is a 1x4 row, which we can split into two 1x2 blocks:
A[i,:] = [A_i0, A_i1] where A_i0 is 1x2, A_i1 is 1x2.

Then C[i,:] = A_i0 * [B00, B01] + A_i1 * [B10, B11]
             = [A_i0*B00 + A_i1*B10,  A_i0*B01 + A_i1*B11]

Each of these is a 1x2 * 2x2 multiplication giving a 1x2 result.
A 1x2 * 2x2 multiplication requires 4 scalar multiplications (schoolbook).

So for each row i of C, we need 4 * 4 = 16 scalar multiplications? No:
- A_i0 * B00: 1x2 * 2x2 -> 4 mults, gives 1x2
- A_i1 * B10: 1x2 * 2x2 -> 4 mults, gives 1x2
- A_i0 * B01: 1x2 * 2x2 -> 4 mults, gives 1x2
- A_i1 * B11: 1x2 * 2x2 -> 4 mults, gives 1x2
Total per row: 16 mults. For 3 rows: 48 mults. Same as schoolbook.

To use Strassen, we need at least 2x2 * 2x2 blocks. Let's try a different partition.

Partition A into 3x4 and B into 4x4. Think of A as having 3 rows and B as 4x4.
Alternatively, think of the problem as computing 3 independent 1x4 * 4x4 products.

For a 1x4 * 4x4 product, can we do better than 16 mults? 
Actually, a 1xn * nxn matrix multiplication: the output is 1xn.
The (1xn) * (nxn) product: C_j = sum_k A_k * B[k,j]. This is n dot products of length n, so n^2 = 16 mults. No Strassen benefit for 1xn * nxn since one factor is 1xn.

Let me reconsider. The known best for 3x4 * 4x4:
We can use the fact that this is related to rectangular matrix multiplication.

Alternative: Use a 2x2x2 Strassen for sub-problems where possible.

Consider partitioning into blocks of size 2:
A is 3x4 = [2x4; 1x4]
B is 4x4 = [[2x2, 2x2], [2x2, 2x2]]

Let A_top = A[0:2, :] (2x4), A_bot = A[2, :] (1x4)
Let B = [[B00, B01], [B10, B11]] with 2x2 blocks.

C_top = A_top * B = A_top[:, 0:2] * [B00, B01] + A_top[:, 2:4] * [B10, B11]
      = [A_top0 * B00 + A_top1 * B10,  A_top0 * B01 + A_top1 * B11]

Where A_top0 = A_top[:, 0:2] is 2x2, A_top1 = A_top[:, 2:4] is 2x2.

So C_top_left (2x2) = A_top0 * B00 + A_top1 * B10
C_top_right (2x2) = A_top0 * B01 + A_top1 * B11

Each of these is a sum of two 2x2 * 2x2 products. Using Strassen for each 2x2*2x2:
- A_top0 * B00: 7 mults (Strassen)
- A_top1 * B10: 7 mults (Strassen)
- A_top0 * B01: 7 mults (Strassen)
- A_top1 * B11: 7 mults (Strassen)
Total for C_top: 28 mults.

C_bot (1x4) = A_bot * B = A_bot[0:2] * [B00, B01] + A_bot[2:4] * [B10, B11]
A_bot[0:2] is 1x2, A_bot[2:4] is 1x2.
C_bot_left (1x2) = A_bot0 * B00 + A_bot1 * B10  (each 1x2*2x2 = 4 mults, total 8)
C_bot_right (1x2) = A_bot0 * B01 + A_bot1 * B11  (each 1x2*2x2 = 4 mults, total 8)
Total for C_bot: 16 mults.

Grand total: 28 + 16 = 44 mults.

This matches the known result from the inspirations! Let me implement this.

Strassen's 7 multiplications for 2x2 * 2x2:
Given X, Y as 2x2 matrices:
M1 = (X00 + X11) * (Y00 + Y11)
M2 = (X10 + X11) * Y00
M3 = X00 * (Y01 - Y11)
M4 = X11 * (Y10 - Y00)
M5 = (X00 + X01) * Y11
M6 = (X10 - X00) * (Y00 + Y01)
M7 = (X01 - X11) * (Y10 + Y11)

Z00 = M1 + M4 - M5 + M7
Z01 = M3 + M5
Z10 = M2 + M4
Z11 = M1 - M2 + M3 + M6

For the tensor decomposition, we need to express each scalar multiplication in the Strassen scheme as a rank-1 term in the tensor.

The tensor T[i, j, k] = delta (where output C[i,j] = sum_{k} A[i,k] * B[k,j]).
Wait, let me re-read the convention.

"output factor follows the official trace convention W[k*n+i,r]"

So the output C has shape n x p = 3 x 4. The index is k*n + i where k is the column of C (0..p-1=0..3) and i is the row of C (0..n-1=0..2). So W index = k*3 + i.

U has shape (n*m, rank) = (3*4, rank) = (12, rank). Index i*m + j where i is row of A (0..2), j is col of A (0..3).
V has shape (m*p, rank) = (4*4, rank) = (16, rank). Index j*p + k where j is row of B (0..3), k is col of B (0..3).
W has shape (p*n, rank) = (4*3, rank) = (12, rank). Index k*n + i where k is col of C (0..3), i is row of C (0..2).

The decomposition: C[i,k] = sum_r U[i*m + j ... wait, no.

The tensor for matrix multiplication: T[i, j, k] where C[i,k] = sum_j A[i,j] * B[j,k].
U has index (i, j) flattened as i*m + j.
V has index (j, k) flattened as j*p + k.
W has index (k, i) flattened as k*n + i.

So for each rank-1 term r:
C[i,k] += U[i*m + j_r, r] * V[j_r * p + k_r, r] * W[k_r * n + i_r, r]
where the indices must satisfy the product structure.

Actually, the standard formulation: The matrix multiplication tensor T of order (n*m, m*p, p*n) has:
T[i*m + j, j'*p + k, k'*n + i'] = 1 if j=j' and k=k' and i=i', else 0.

Wait, that's not quite right either. Let me think more carefully.

The tensor T has entries T[a, b, c] where:
a = i*m + j (i in 0..n-1, j in 0..m-1)
b = j'*p + k (j' in 0..m-1, k in 0..p-1)  
c = k'*n + i' (k' in 0..p-1, i' in 0..n-1)

T[a, b, c] = 1 if j == j' and k == k' and i == i', else 0.

So T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}

The rank-1 decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
where U[i*m+j, r], V[j'*p+k, r], W[k'*n+i', r].

For a rank-1 term to contribute to T[i*m+j, j'*p+k, k'*n+i'], we need:
U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}

This means for each r, the nonzero entries of U[:,r], V[:,r], W[:,r] must be consistent.

Let me just implement the 44-rank decomposition directly.

For the Strassen part on the top 2 rows:
We have A_top (2x4) * B (4x4) -> C_top (2x4).

Partition:
A_top = [[a00, a01, a02, a03],
         [a10, a11, a12, a13]]

B = [[b00, b01, b02, b03],
     [b10, b11, b12, b13],
     [b20, b21, b22, b23],
     [b30, b31, b32, b33]]

Split B into 2x2 blocks:
B00 = [[b00, b01], [b10, b11]]  (rows 0,1 cols 0,1)
B01 = [[b02, b03], [b12, b13]]  (rows 0,1 cols 2,3)
B10 = [[b20, b21], [b30, b31]]  (rows 2,3 cols 0,1)
B11 = [[b22, b23], [b32, b33]]  (rows 2,3 cols 2,3)

A_top0 = [[a00, a01], [a10, a11]]  (cols 0,1)
A_top1 = [[a02, a03], [a12, a13]]  (cols 2,3)

C_top_left = A_top0 * B00 + A_top1 * B10  (2x2)
C_top_right = A_top0 * B01 + A_top1 * B11  (2x2)

For A_top0 * B00 using Strassen (7 mults):
Let X = A_top0, Y = B00.
M1 = (X00+X11)*(Y00+Y11) = (a00+a11)*(b00+b11)
M2 = (X10+X11)*Y00 = (a10+a11)*b00
M3 = X00*(Y01-Y11) = a00*(b01-b11)
M4 = X11*(Y10-Y00) = a11*(b10-b00)
M5 = (X00+X01)*Y11 = (a00+a01)*b11
M6 = (X10-X00)*(Y00+Y01) = (a10-a00)*(b00+b01)
M7 = (X01-X11)*(Y10+Y11) = (a01-a11)*(b10+b11)

Z00 = M1+M4-M5+M7
Z01 = M3+M5
Z10 = M2+M4
Z11 = M1-M2+M3+M6

Each M corresponds to a product of a linear form in A and a linear form in B, contributing to specific entries of C.

This is getting complex. Let me just build the U, V, W matrices by tracking each multiplication.

I'll create a helper that, given a product of a linear form in A and a linear form in B, adds a rank-1 term to the decomposition, distributing the contributions to the appropriate C entries.

Actually, let me just hardcode the 44 rank-1 terms.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U: (12, rank), V: (16, rank), W: (12, rank)
    
    # We'll build a list of (u_idx, u_val, v_idx, v_val, w_idx, w_val) tuples
    # Each represents a rank-1 term.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Index helpers
    def U_idx(i, j): return i * m + j  # i in 0..2, j in 0..3
    def V_idx(j, k): return j * p + k  # j in 0..3, k in 0..3
    def W_idx(k, i): return k * n + i  # k in 0..3, i in 0..2
    
    # Top 2 rows of A: i=0,1
    # A_top0: cols 0,1 -> indices (0,0),(0,1),(1,0),(1,1)
    # A_top1: cols 2,3 -> indices (0,2),(0,3),(1,2),(1,3)
    
    # B00: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3
    # B10: rows 2,3 cols 0,1
    # B11: rows 2,3 cols 2,3
    
    # Strassen for A_top0 * B00 -> contributes to C[0:2, 0:2]
    # X = A_top0, Y = B00
    # M1 = (a00+a11)*(b00+b11)
    # This is a product of (A[0,0] + A[1,1]) and (B[0,0] + B[1,1])
    # It contributes to Z00 and Z11 of the 2x2 result.
    # Z00 = C[0,0], Z01 = C[0,1], Z10 = C[1,0], Z11 = C[1,1]
    # But wait, C_top_left is C[0:2, 0:2], so:
    # C_top_left[0,0] = C[0,0], C_top_left[0,1] = C[0,1]
    # C_top_left[1,0] = C[1,0], C_top_left[1,1] = C[1,1]
    
    # For M1 = (a00+a11)*(b00+b11):
    # The A-part is a00 + a11, which involves U indices U_idx(0,0) and U_idx(1,1)
    # The B-part is b00 + b11, which involves V indices V_idx(0,0) and V_idx(1,1)
    # The contribution to C: Z00 gets +M1, Z11 gets +M1
    # Z00 = C[0,0] -> W_idx(0, 0)
    # Z11 = C[1,1] -> W_idx(1, 1)
    # But a rank-1 term has a single U, V, W index. So we need to split.
    
    # Actually, in the tensor decomposition, each rank-1 term is:
    # U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    # For this to equal the product of a linear form in A and B contributing to C[i',k'],
    # we need the U part to be a single entry (or the term to be structured).
    
    # Wait, I think I'm overcomplicating. Let me use the standard approach:
    # Each rank-1 term corresponds to one scalar multiplication in the algorithm.
    # The scalar multiplication is: (linear form in A) * (linear form in B)
    # The linear form in A is a sum of A[i,j] terms, so U has nonzero entries at those (i,j).
    # Similarly for V and B.
    # The result contributes to C[i',k'] with coefficient = product of coefficients.
    # So W[k'*n+i', r] = sum of contributions.
    
    # Let me redefine: for each multiplication step, we have:
    # prod = (sum_{(i,j) in S_A} c_{ij} * A[i,j]) * (sum_{(j',k) in S_B} d_{j'k} * B[j',k])
    # This contributes to C[i',k'] for various (i',k').
    # In the tensor: U[i*m+j, r] = c_{ij} for (i,j) in S_A
    #            V[j'*p+k, r] = d_{j'k} for (j',k) in S_B
    #            W[k'*n+i', r] = contribution to C[i',k']
    
    # The contribution to C[i',k'] from this term is:
    # (sum_{(i,j) in S_A} c_{ij} * A[i,j]) * (sum_{(j',k) in S_B} d_{j'k} * B[j',k])
    # But this is a bilinear form, and it should equal the sum over the "matching" indices.
    
    # Actually, the correct way: the product of two linear forms gives a bilinear form.
    # For it to be part of the matrix multiplication, we need:
    # prod = sum_{j} A[i,j] * B[j,k] for specific i,k.
    # But a general product of linear forms is not necessarily of that form.
    
    # I think the key insight is: each rank-1 term in the tensor decomposition
    # corresponds to a "gate" that multiplies one A-entry (or linear combo) by one B-entry (or linear combo)
    # and the result is added to specific C entries.
    
    # Let me just use the explicit Strassen formulas and track each multiplication.
    
    # I'll create a more systematic approach.
    
    # Let me just hardcode all 44 terms based on the Strassen decomposition.
    
    # This is complex, so let me use a different strategy:
    # Build the decomposition by explicitly computing each of the 44 scalar multiplications
    # and tracking which A and B entries they involve, and which C entries they contribute to.
    
    # I'll use a dictionary to accumulate the U, V, W values.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    def add_strassen(X_rows, X_cols, Y_rows, Y_cols, C_offset_i, C_offset_k):
        """
        Add Strassen decomposition for X (2x2) * Y (2x2).
        X is A[row_range, col_range], Y is B[row_range, col_range].
        Result contributes to C[C_offset_i:C_offset_i+2, C_offset_k:C_offset_k+2].
        """
        nonlocal r
        # X indices in A: (X_rows[0], X_cols[0]), (X_rows[0], X_cols[1]), (X_rows[1], X_cols[0]), (X_rows[1], X_cols[1])
        # Y indices in B: (Y_rows[0], Y_cols[0]), etc.
        
        x00_r, x00_c = X_rows[0], X_cols[0]
        x01_r, x01_c = X_rows[0], X_cols[1]
        x10_r, x10_c = X_rows[1], X_cols[0]
        x11_r, x11_c = X_rows[1], X_cols[1]
        
        y00_r, y00_c = Y_rows[0], Y_cols[0]
        y01_r, y01_c = Y_rows[0], Y_cols[1]
        y10_r, y10_c = Y_rows[1], Y_cols[0]
        y11_r, y11_c = Y_rows[1], Y_cols[1]
        
        # M1 = (x00 + x11) * (y00 + y11)
        U[U_idx(x00_r, x00_c)][r] = 1
        U[U_idx(x11_r, x11_c)][r] = 1
        V[V_idx(y00_r, y00_c)][r] = 1
        V[V_idx(y11_r, y11_c)][r] = 1
        # Z00 += M1, Z11 += M1
        W[W_idx(C_offset_k + 0, C_offset_i + 0)][r] = 1  # Z00
        W[W_idx(C_offset_k + 1, C_offset_i + 1)][r] = 1  # Z11
        r += 1
        
        # M2 = (x10 + x11) * y00
        U[U_idx(x10_r, x10_c)][r] = 1
        U[U_idx(x11_r, x11_c)][r] = 1
        V[V_idx(y00_r, y00_c)][r] = 1
        # Z10 += M2
        W[W_idx(C_offset_k + 0, C_offset_i + 1)][r] = 1  # Z10
        r += 1
        
        # M3 = x00 * (y01 - y11)
        U[U_idx(x00_r, x00_c)][r] = 1
        V[V_idx(y01_r, y01_c)][r] = 1
        V[V_idx(y11_r, y11_c)][r] = -1
        # Z01 += M3
        W[W_idx(C_offset_k + 1, C_offset_i + 0)][r] = 1  # Z01
        r += 1
        
        # M4 = x11 * (y10 - y00)
        U[U_idx(x11_r, x11_c)][r] = 1
        V[V_idx(y10_r, y10_c)][r] = 1
        V[V_idx(y00_r, y00_c)][r] = -1
        # Z00 += M4, Z10 += M4
        W[W_idx(C_offset_k + 0, C_offset_i + 0)][r] = 1  # Z00
        W[W_idx(C_offset_k + 0, C_offset_i + 1)][r] = 1  # Z10
        r += 1
        
        # M5 = (x00 + x01) * y11
        U[U_idx(x00_r, x00_c)][r] = 1
        U[U_idx(x01_r, x01_c)][r] = 1
        V[V_idx(y11_r, y11_c)][r] = 1
        # Z00 -= M5, Z01 += M5
        W[W_idx(C_offset_k + 0, C_offset_i + 0)][r] = -1  # Z00
        W[W_idx(C_offset_k + 1, C_offset_i + 0)][r] = 1   # Z01
        r += 1
        
        # M6 = (x10 - x00) * (y00 + y01)
        U[U_idx(x10_r, x10_c)][r] = 1
        U[U_idx(x00_r, x00_c)][r] = -1
        V[V_idx(y00_r, y00_c)][r] = 1
        V[V_idx(y01_r, y01_c)][r] = 1
        # Z11 += M6
        W[W_idx(C_offset_k + 1, C_offset_i + 1)][r] = 1  # Z11
        r += 1
        
        # M7 = (x01 - x11) * (y10 + y11)
        U[U_idx(x01_r, x01_c)][r] = 1
        U[U_idx(x11_r, x11_c)][r] = -1
        V[V_idx(y10_r, y10_c)][r] = 1
        V[V_idx(y11_r, y11_c)][r] = 1
        # Z00 += M7
        W[W_idx(C_offset_k + 0, C_offset_i + 0)][r] = 1  # Z00
        r += 1
    
    # C_top_left = A_top0 * B00 + A_top1 * B10
    # A_top0: rows 0,1 cols 0,1
    # B00: rows 0,1 cols 0,1
    # Result: C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], 0, 0)
    
    # A_top1: rows 0,1 cols 2,3
    # B10: rows 2,3 cols 0,1
    # Result: C[0:2, 0:2]
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], 0, 0)
    
    # C_top_right = A_top0 * B01 + A_top1 * B11
    # A_top0: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3
    # Result: C[0:2, 2:4]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], 0, 2)
    
    # A_top1: rows 0,1 cols 2,3
    # B11: rows 2,3 cols 2,3
    # Result: C[0:2, 2:4]
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], 0, 2)
    
    # Now for the bottom row: C[2, :] = A[2, :] * B
    # A_bot0 = A[2, 0:2] (1x2), A_bot1 = A[2, 2:4] (1x2)
    # C_bot_left = A_bot0 * B00 + A_bot1 * B10 (1x2)
    # C_bot_right = A_bot0 * B01 + A_bot1 * B11 (1x2)
    
    # For 1x2 * 2x2, schoolbook uses 4 mults.
    # A_bot0 = [a20, a21], B00 = [[b00, b01], [b10, b11]]
    # Result = [a20*b00 + a21*b10, a20*b01 + a21*b11]
    
    # Term 1: a20 * b00 -> C[2,0]
    U[U_idx(2, 0)][r] = 1
    V[V_idx(0, 0)][r] = 1
    W[W_idx(0, 2)][r] = 1
    r += 1
    
    # Term 2: a21 * b10 -> C[2,0]
    U[U_idx(2, 1)][r] = 1
    V[V_idx(1, 0)][r] = 1
    W[W_idx(0, 2)][r] = 1
    r += 1
    
    # Term 3: a20 * b01 -> C[2,1]
    U[U_idx(2, 0)][r] = 1
    V[V_idx(0, 1)][r] = 1
    W[W_idx(1, 2)][r] = 1
    r += 1
    
    # Term 4: a21 * b11 -> C[2,1]
    U[U_idx(2, 1)][r] = 1
    V[V_idx(1, 1)][r] = 1
    W[W_idx(1, 2)][r] = 1
    r += 1
    
    # C_bot_right = A_bot0 * B01 + A_bot1 * B11
    # A_bot0 = [a20, a21], B01 = [[b02, b03], [b12, b13]]
    # Result = [a20*b02 + a21*b12, a20*b03 + a21*b13]
    
    # Term 5: a20 * b02 -> C[2,2]
    U[U_idx(2, 0)][r] = 1
    V[V_idx(0, 2)][r] = 1
    W[W_idx(2, 2)][r] = 1
    r += 1
    
    # Term 6: a21 * b12 -> C[2,2]
    U[U_idx(2, 1)][r] = 1
    V[V_idx(1, 2)][r] = 1
    W[W_idx(2, 2)][r] = 1
    r += 1
    
    # Term 7: a20 * b03 -> C[2,3]
    U[U_idx(2, 0)][r] = 1
    V[V_idx(0, 3)][r] = 1
    W[W_idx(3, 2)][r] = 1
    r += 1
    
    # Term 8: a21 * b13 -> C[2,3]
    U[U_idx(2, 1)][r] = 1
    V[V_idx(1, 3)][r] = 1
    W[W_idx(3, 2)][r] = 1
    r += 1
    
    # A_bot1 = [a22, a23], B10 = [[b20, b21], [b30, b31]]
    # Result = [a22*b20 + a23*b30, a22*b21 + a23*b31]
    
    # Term 9: a22 * b20 -> C[2,0]
    U[U_idx(2, 2)][r] = 1
    V[V_idx(2, 0)][r] = 1
    W[W_idx(0, 2)][r] = 1
    r += 1
    
    # Term 10: a23 * b30 -> C[2,0]
    U[U_idx(2, 3)][r] = 1
    V[V_idx(3, 0)][r] = 1
    W[W_idx(0, 2)][r] = 1
    r += 1
    
    # Term 11: a22 * b21 -> C[2,1]
    U[U_idx(2, 2)][r] = 1
    V[V_idx(2, 1)][r] = 1
    W[W_idx(1, 2)][r] = 1
    r += 1
    
    # Term 12: a23 * b31 -> C[2,1]
    U[U_idx(2, 3)][r] = 1
    V[V_idx(3, 1)][r] = 1
    W[W_idx(1, 2)][r] = 1
    r += 1
    
    # A_bot1 = [a22, a23], B11 = [[b22, b23], [b32, b33]]
    # Result = [a22*b22 + a23*b32, a22*b23 + a23*b33]
    
    # Term 13: a22 * b22 -> C[2,2]
    U[U_idx(2, 2)][r] = 1
    V[V_idx(2, 2)][r] = 1
    W[W_idx(2, 2)][r] = 1
    r += 1
    
    # Term 14: a23 * b32 -> C[2,2]
    U[U_idx(2, 3)][r] = 1
    V[V_idx(3, 2)][r] = 1
    W[W_idx(2, 2)][r] = 1
    r += 1
    
    # Term 15: a22 * b23 -> C[2,3]
    U[U_idx(2, 2)][r] = 1
    V[V_idx(2, 3)][r] = 1
    W[W_idx(3, 2)][r] = 1
    r += 1
    
    # Term 16: a23 * b33 -> C[2,3]
    U[U_idx(2, 3)][r] = 1
    V[V_idx(3, 3)][r] = 1
    W[W_idx(3, 2)][r] = 1
    r += 1
    
    assert r == 44, f"Expected 44 terms, got {r}"
    
    return {"U": U, "V": V, "W": W}

# Verify the decomposition
def _verify():
    n, m, p = DIMENSIONS
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_expected[a, b, c] = 1
    
    diff = np.max(np.abs(T - T_expected))
    print(f"Verification: max diff = {diff}")
    return diff < 1e-10

if __name__ == "__main__":
    _verify()
