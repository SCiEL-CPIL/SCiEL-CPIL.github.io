"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition.

Strategy: Treat the 2x2 by 2x13 multiplication as matrix multiplication over
polynomials in one variable.  Each column of B is a linear combination of two
"basis" products, and each output column is a linear combination of the same
seven Strassen-style intermediate products shifted in the polynomial ring.

In practice we use the following construction:

- For 13 output columns, we can pair them into 6 pairs + 1 leftover.
- For each pair (k, k+1) we use 7 Strassen terms = 42 terms.
- For the leftover column (k=12) we use 4 schoolbook terms = 4 terms.
- Total rank = 46.

However, we can do better by using a polynomial-based approach where we
represent the 2x13 matrix as a 2x2 matrix over Z[x] with x being a formal
variable, but this is complex.

A simpler improvement: use the fact that for 2x2 matrix multiplication,
the rank is 7 (Strassen). For 2x2 by 2x13, we can view this as multiplying
a 2x2 matrix by 13 different 2x1 matrices. Each 2x2 by 2x1 multiplication
has rank 4 (schoolbook). But we can share structure.

Actually, let me think about this differently. The tensor for 2x2 by 2x13
multiplication has entries T[i,j,k,l] where:
- i in {0,1} (row of A)
- j in {0,1} (col of A, row of B)
- k in {0,1} (col of B, row of C)
- l in {0,...,12} (col of B, col of C)

T[i,j,k,l] = 1 if the product A[i,j]*B[k,l] contributes to C[i,l]... wait.

Let me re-read the convention. W[k*n+i, r]. So W has p*n = 13*2 = 26 rows.
W index = k*n + i where k is output column (0..12), i is output row (0..1).

U has n*m = 2*2 = 4 rows: U[i*m+j, r] where i is A row, j is A col.
V has m*p = 2*13 = 26 rows: V[j*p+k, r] where j is B row, k is B col.

The tensor is: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta...

For matrix multiplication C = A * B:
C[i,k] = sum_j A[i,j] * B[j,k]

So T[i,j,j',k] = 1 if j == j' (A col matches B row), 0 otherwise.

The decomposition requires:
sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[i,j,j',k] * delta[k,k']

Wait, the output is C[i',k'] = sum_{j,j'} A[i,j] B[j,k] * T[i,j,j',k] * delta...

Actually the standard tensor for matmul:
T_{i,j,k} = 1 if the product of A[i,j] and B[j,k] contributes to C[i,k].

For 2x2 by 2x13:
A is 2x2, B is 2x13, C is 2x13.
C[i,k] = sum_j A[i,j] B[j,k]

Tensor entries: T[i, j, j', k] = 1 if j == j', else 0. (i is row of A/C, j is col of A, j' is row of B, k is col of B/C)

The decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k]) * W[k*n+i, r]

This must equal sum_j A[i,j] B[j,k] for all i,k.

So we need: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'} for all i,k,j,j'.

This is a constraint that must hold for all i,k,j,j'. For a fixed (j,j'), the sum over r must give delta_{j,j'} regardless of i and k. This means the factorization must work uniformly.

Let me just implement the known rank-46 construction carefully and verify it.

Rank 46: 6 Strassen blocks (7 each) + 1 schoolbook block (4) = 42 + 4 = 46.

For each block b (b=0..5), columns k=2b and k=2b+1:
A is 2x2, B is 2x2 (columns 2b, 2b+1).

Strassen for 2x2 matmul:
M1 = (A00+A11)(B00+B11)
M2 = (A10+A11)B00
M3 = A00(B10-B11)
M4 = A11(B01-B00)
M5 = (A00+A01)B11
M6 = (A10-A00)(B00+B10)
M7 = (A01-A11)(B01+B11)

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

Let me verify:
C00 = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B01+B11)
= A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
= A00B00 + A01B01. Correct!

C01 = A00(B10-B11) + (A00+A01)B11 = A00B10 - A00B11 + A00B11 + A01B11 = A00B10 + A01B11. Correct!

C10 = (A10+A11)B00 + A11(B01-B00) = A10B00 + A11B00 + A11B01 - A11B00 = A10B00 + A11B01. Correct!

C11 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + (A10-A00)(B00+B10)
= A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A10B00 + A10B10 - A00B00 - A00B10
= A10B10 + A11B11. Correct!

Good. Now implement this.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (26, rank)  
    # W: (p*n, rank) = (26, rank)
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*m + j where i=0,1 (row), j=0,1 (col)
    # A00 = U[0], A01 = U[1], A10 = U[2], A11 = U[3]
    
    # B indices: j*p + k where j=0,1 (row), k=0..12 (col)
    # B0k = V[k], B1k = V[13+k]
    
    # W indices: k*n + i where k=0..12 (col), i=0,1 (row)
    # C0k = W[k], C1k = W[k+1]
    
    # 6 Strassen blocks for k=0..11
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices for this block
        b00 = k0           # B[0, k0]
        b10 = p + k0       # B[1, k0]
        b01 = k1           # B[0, k1]
        b11 = p + k1       # B[1, k1]
        
        # W indices
        c00 = k0 * n + 0   # C[0, k0]
        c01 = k1 * n + 0   # C[0, k1]
        c10 = k0 * n + 1   # C[1, k0]
        c11 = k1 * n + 1   # C[1, k1]
        
        base = 7 * b
        
        # M1 = (A00+A11)(B00+B11)
        col = base
        U[0][col] += 1; U[3][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        
        # M2 = (A10+A11)B00
        col = base + 1
        U[2][col] += 1; U[3][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        
        # M3 = A00(B10-B11)
        col = base + 2
        U[0][col] += 1
        V[b10][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        
        # M4 = A11(B01-B00)
        col = base + 3
        U[3][col] += 1
        V[b01][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        
        # M5 = (A00+A01)B11
        col = base + 4
        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        
        # M6 = (A10-A00)(B00+B10)
        col = base + 5
        U[2][col] += 1; U[0][col] -= 1
        V[b00][col] += 1; V[b10][col] += 1
        W[c11][col] += 1
        
        # M7 = (A01-A11)(B01+B11)
        col = base + 6
        U[1][col] += 1; U[3][col] -= 1
        V[b01][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
    
    # Schoolbook block for k=12
    k = 12
    b0 = k          # B[0, 12]
    b1 = p + k      # B[1, 12]
    c0 = k * n + 0  # C[0, 12]
    c1 = k * n + 1  # C[1, 12]
    
    # A00 * B0 -> C0
    col = 42
    U[0][col] += 1
    V[b0][col] += 1
    W[c0][col] += 1
    
    # A01 * B1 -> C0
    col = 43
    U[1][col] += 1
    V[b1][col] += 1
    W[c0][col] += 1
    
    # A10 * B0 -> C1
    col = 44
    U[2][col] += 1
    V[b0][col] += 1
    W[c1][col] += 1
    
    # A11 * B1 -> C1
    col = 45
    U[3][col] += 1
    V[b1][col] += 1
    W[c1][col] += 1
    
    # Verify the decomposition
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each (i,j,j',k), sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k*n+i,r] should be delta_{j,j'}
        # Actually, the decomposition is:
        # C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_j' V[j'*p+k,r] B[j',k]) * W[k*n+i,r]
        # This equals sum_{j,j'} A[i,j] B[j',k] * W[k*n+i,r] * sum_r U[i*m+j,r] V[j'*p+k,r]
        # Wait, that's not right. Let me think again.
        
        # The tensor decomposition means:
        # T[i,j,j',k] = sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k*n+i,r]
        # where T[i,j,j',k] = 1 if j==j' else 0 (for matmul)
        
        # Let's verify a few entries
        max_err = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        expected = 1 if j == jp else 0
                        actual = sum(U[i*m+j][r] * V[jp*p+k][r] * W[k*n+i][r] for r in range(rank))
                        err = abs(expected - actual)
                        if err > max_err:
                            max_err = err
                            if max_err > 1e-10:
                                print(f"Error at i={i}, j={j}, j'={jp}, k={k}: expected={expected}, actual={actual}")
        
        if max_err > 1e-10:
            print(f"Max error: {max_err}")
    
    return {"U": U, "V": V, "W": W}
