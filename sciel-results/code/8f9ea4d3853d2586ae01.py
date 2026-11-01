"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x8 matrix.

Strategy: Use Strassen's algorithm on 2x2 blocks of the output.
Group the 8 output columns into 4 pairs. For each pair (k, k+1), we compute
C[:, k:k+2] = A * B[:, k:k+2] using Strassen's 7-multiplication algorithm
for 2x2 matrix multiplication, where B_block is 2x2.

This shares linear combinations of A across the two columns in each pair,
reducing the rank from 32 (schoolbook) to 4 * 7 = 28.

Indexing convention (from task contract):
- U shape: (n*m, rank) = (4, rank), indexed by [i*m + j, r]
- V shape: (m*p, rank) = (16, rank), indexed by [j*p + k, r]  
- W shape: (p*n, rank) = (16, rank), indexed by [k*n + i, r]

The bilinear form: C_flat[k*n + i] = sum_r U[i*m+j, r] * A_flat[i*m+j] * V[j*p+k, r] * B_flat[j*p+k] * W[k*n+i, r]
More precisely: for each rank term r, we compute M_r = (U_r^T A_flat) * (V_r^T B_flat),
and C_flat = sum_r M_r * W_r (outer product sense).

Actually the standard tensor decomposition:
T = sum_r u_r ⊗ v_r ⊗ w_r
where T[i,j,k] = 1 for all valid i,j,k (the multiplication tensor).
And the reconstruction is: C[k,i] = sum_{j,r} A[i,j] * B[j,k] * ... 

Let me be precise. The matrix multiplication tensor T_{n,m,p} has:
T[i, j, k] = 1  (for all i in 0..n-1, j in 0..m-1, k in 0..p-1)

A rank-r decomposition writes:
T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]

With flattened indices:
U_flat[i*m + j, r], V_flat[j*p + k, r], W_flat[k*n + i, r]

And the reconstruction:
C_flat[k*n + i] = sum_{j=0}^{m-1} sum_{r=0}^{r-1} U[i*m+j, r] * A[i*m+j] * V[j*p+k, r] * B[j*p+k] * W[k*n+i, r]

Wait, that's not quite right either. The standard formulation is:

For each rank term r:
  M_r = (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_{j,k} V[j,k,r] * B[j,k])
  Then C[i,k] += M_r * W[k,i,r]

So: C_flat[k*n+i] = sum_r W[k*n+i, r] * M_r
where M_r = (U_r · A_flat) * (V_r · B_flat)

This means:
U_flat[i*m+j, r] gives the coefficient of A[i,j] in the first factor of M_r
V_flat[j*p+k, r] gives the coefficient of B[j,k] in the second factor of M_r
W_flat[k*n+i, r] gives the coefficient of M_r in C[i,k]

For Strassen on a 2x2 block:
A = [[a, b], [c, d]], B_block = [[e, f], [g, h]], C_block = [[i, j], [k, l]]

M1 = (a+d)(e+h)
M2 = (c+d)e
M3 = a(f-h)
M4 = d(g-e)
M5 = (a+b)h
M6 = (c-a)(e+f)
M7 = (b-d)(g+h)

C:
i = M1 - M5 + M7
j = M3 + M5
k = M2 + M4
l = M1 - M3 + M6

For the full 2x2 * 2x8 problem, we group columns into 4 pairs:
Block 0: columns k=0,1 -> B_block = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
Block 1: columns k=2,3 -> B_block = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]]
Block 2: columns k=4,5 -> B_block = [[B[0,4], B[0,5]], [B[1,4], B[1,5]]]
Block 3: columns k=6,7 -> B_block = [[B[0,6], B[0,7]], [B[1,6], B[1,7]]]

For each block b (0..3) and each Strassen term s (0..6):
  Column index in rank: col = b*7 + s

  U entries (coefficients for A = [[a,b],[c,d]], flat indices [0,1,2,3]):
  V entries (coefficients for B_block, flat indices for this block's 4 B entries)
  W entries (coefficients for C_block, flat indices for this block's 4 C entries)

Let's map the indices carefully:

A flat index: i*m + j where i is row, j is col of A (2x2)
  A[0,0]=a -> index 0
  A[0,1]=b -> index 1
  A[1,0]=c -> index 2
  A[1,1]=d -> index 3

B flat index: j*p + k where j is row, k is col of B (2x8)
  For block b (columns 2b, 2b+1):
  B[0,2b] -> index 0*8 + 2b = 2b
  B[0,2b+1] -> index 0*8 + 2b+1 = 2b+1
  B[1,2b] -> index 1*8 + 2b = 8+2b
  B[1,2b+1] -> index 1*8 + 2b+1 = 8+2b+1

C flat index: k*n + i where k is col, i is row of C (2x8, stored as W[k*n+i])
  For block b (columns 2b, 2b+1):
  C[0,2b] -> index 2b*2 + 0 = 4b
  C[1,2b] -> index 2b*2 + 1 = 4b+1
  C[0,2b+1] -> index (2b+1)*2 + 0 = 4b+2
  C[1,2b+1] -> index (2b+1)*2 + 1 = 4b+3

Strassen U vectors (for A = [a,b,c,d]):
  M1: a+d      -> [1, 0, 0, 1]
  M2: c+d      -> [0, 0, 1, 1]
  M3: a        -> [1, 0, 0, 0]
  M4: d        -> [0, 0, 0, 1]
  M5: a+b      -> [1, 1, 0, 0]
  M6: c-a      -> [1, 0, -1, 0]  -- wait, c-a means A[1,0] - A[0,0] = c - a
      So [−1, 0, 1, 0]
  M7: b-d      -> [0, 1, 0, −1]

Let me re-verify M6 and M7:
  M6 = (c - a)(e + f): c = A[1,0], a = A[0,0], so coefficient of a is -1, c is +1
    -> [-1, 0, 1, 0]
  M7 = (b - d)(g + h): b = A[0,1], d = A[1,1], so coefficient of b is +1, d is -1
    -> [0, 1, 0, -1]

Strassen V vectors (for B_block = [e, f, g, h] where e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]):
  M1: e+h      -> [1, 0, 0, 1]
  M2: e        -> [1, 0, 0, 0]
  M3: f-h      -> [0, 1, 0, -1]
  M4: g-e      -> [0, 0, 1, -1]
  M5: h        -> [0, 0, 0, 1]
  M6: e+f      -> [1, 1, 0, 0]
  M7: g+h      -> [0, 0, 1, 1]

Strassen W vectors (for C_block = [i, j, k, l] where i=C[0,2b], j=C[0,2b+1], k=C[1,2b], l=C[1,2b+1]):
  Wait, let me re-check the C indexing.
  
  C is 2x8. C[row, col]. 
  W index is k*n + i where k is the column index of C and i is the row index.
  So W[col*2 + row, r].
  
  For block b (columns 2b, 2b+1):
  C[0, 2b]   -> W index: 2b*2 + 0 = 4b
  C[0, 2b+1] -> W index: (2b+1)*2 + 0 = 4b+2
  C[1, 2b]   -> W index: 2b*2 + 1 = 4b+1
  C[1, 2b+1] -> W index: (2b+1)*2 + 1 = 4b+3

  Strassen output:
  i (C[0,0] of block) = M1 - M5 + M7
  j (C[0,1] of block) = M3 + M5
  k (C[1,0] of block) = M2 + M4
  l (C[1,1] of block) = M1 - M3 + M6

  So W vectors (indexed by [C[0,0], C[0,1], C[1,0], C[1,1]] = [i, j, k, l]):
  M1: [1, 0, 0, 1]    (contributes to i and l)
  M2: [0, 0, 1, 0]    (contributes to k)
  M3: [0, 1, 0, -1]   (contributes to j and -l)
  M4: [0, 0, 1, 0]    (contributes to k)
  M5: [-1, 1, 0, 0]   (contributes to -i and j)
  M6: [0, 0, 0, 1]    (contributes to l)
  M7: [1, 0, 0, 0]    (contributes to i)

Now let's code this up.
"""

import time


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = 2, 2, 8
    block_rank = 7  # Strassen rank for 2x2x2
    num_blocks = p // 2  # 4 blocks of 2 columns each
    total_rank = num_blocks * block_rank  # 28

    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]   # (4, 28)
    V = [[0] * total_rank for _ in range(m * p)]   # (16, 28)
    W = [[0] * total_rank for _ in range(p * n)]   # (16, 28)

    # Strassen U vectors: coefficients for A = [a, b, c, d] (flat: A[i*2+j])
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    u_strassen = [
        [1, 0, 0, 1],    # M1: a + d
        [0, 0, 1, 1],    # M2: c + d
        [1, 0, 0, 0],    # M3: a
        [0, 0, 0, 1],    # M4: d
        [1, 1, 0, 0],    # M5: a + b
        [-1, 0, 1, 0],   # M6: c - a
        [0, 1, 0, -1],   # M7: b - d
    ]

    # Strassen V vectors: coefficients for B_block = [e, f, g, h]
    # e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
    v_strassen = [
        [1, 0, 0, 1],    # M1: e + h
        [1, 0, 0, 0],    # M2: e
        [0, 1, 0, -1],   # M3: f - h
        [0, 0, 1, -1],   # M4: g - e
        [0, 0, 0, 1],    # M5: h
        [1, 1, 0, 0],    # M6: e + f
        [0, 0, 1, 1],    # M7: g + h
    ]

    # Strassen W vectors: coefficients for C_block = [i, j, k, l]
    # i=C[0,2b], j=C[0,2b+1], k=C[1,2b], l=C[1,2b+1]
    w_strassen = [
        [1, 0, 0, 1],    # M1: i + l
        [0, 0, 1, 0],    # M2: k
        [0, 1, 0, -1],   # M3: j - l
        [0, 0, 1, 0],    # M4: k
        [-1, 1, 0, 0],   # M5: -i + j
        [0, 0, 0, 1],    # M6: l
        [1, 0, 0, 0],    # M7: i
    ]

    # Fill the matrices for each block
    for b in range(num_blocks):
        if time.time() > deadline:
            break  # Return what we have; but this loop is tiny so unlikely

        col_base = b * block_rank

        for s in range(block_rank):
            col = col_base + s

            # Fill U: U[i*m + j, col] = u_strassen[s][i*2+j]
            for idx_a in range(n * m):
                U[idx_a][col] = u_strassen[s][idx_a]

            # Fill V: V[j*p + k, col]
            # For block b, the B entries are:
            #   e = B[0, 2b]   -> V index: 0*p + 2b = 2b
            #   f = B[0, 2b+1] -> V index: 0*p + 2b+1 = 2b+1
            #   g = B[1, 2b]   -> V index: 1*p + 2b = p + 2b
            #   h = B[1, 2b+1] -> V index: 1*p + 2b+1 = p + 2b + 1
            v_indices = [2 * b, 2 * b + 1, p + 2 * b, p + 2 * b + 1]
            for local_idx, global_idx in enumerate(v_indices):
                V[global_idx][col] = v_strassen[s][local_idx]

            # Fill W: W[k*n + i, col]
            # For block b, the C entries are:
            #   i = C[0, 2b]   -> W index: 2b*n + 0 = 4b
            #   j = C[0, 2b+1] -> W index: (2b+1)*n + 0 = 4b+2
            #   k = C[1, 2b]   -> W index: 2b*n + 1 = 4b+1
            #   l = C[1, 2b+1] -> W index: (2b+1)*n + 1 = 4b+3
            w_indices = [4 * b, 4 * b + 2, 4 * b + 1, 4 * b + 3]
            for local_idx, global_idx in enumerate(w_indices):
                W[global_idx][col] = w_strassen[s][local_idx]

    return {"U": U, "V": V, "W": W}
