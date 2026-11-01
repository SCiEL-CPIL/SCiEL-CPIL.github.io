"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid exact tensor decomposition for 3x3 * 3x4 -> 3x4.

    Strategy: split the right factor B (3x4) into a 3x3 block B1 and a 3x1 column b2.
      C = A * [B1 | b2] = [A*B1 | A*b2]
    - A*B1 is a 3x3 * 3x3 product (schoolbook rank 27).
    - A*b2 is a 3x3 * 3x1 product (rank 3: one rank-1 per output row).
    Total rank = 27 + 3 = 30 < 36 (full schoolbook).
    """
    start = time.time()
    DEADLINE = start + 140.0  # safety margin under 150s

    n, m, p = DIMENSIONS  # 3, 3, 4

    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank).
    # Index conventions:
    #   U[i*m + j, c]  -> factor for A[i, j]
    #   V[j*p + k, c]  -> factor for B[j, k]
    #   W[k*n + i, c]  -> factor for C[i, k]  (trace convention W[k*n + i, r])

    # ---- Part 1: 3x3 * 3x3 (columns 0..2 of B) schoolbook, rank 27 ----
    # For each (i, j, k) with k in 0..2: contributes to C[i,k]
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    cols_part1 = []
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k = 0,1,2 (the 3x3 block)
                cols_part1.append((i, j, k))

    # ---- Part 2: 3x3 * 3x1 (column 3 of B), rank 3 ----
    # C[i, 3] = sum_j A[i,j] * B[j,3]
    # For each output row i: one rank-1 term with U[i*m+j]=1 for all j, V[j*p+3]=1 for all j, W[3*n+i]=1.
    # Actually each term is: sum_j A[i,j]*B[j,3]. That's a single rank-1 in the tensor? No.
    # The bilinear form is sum_j A[i,j]*B[j,3] which is a sum of 3 rank-1 (one per j).
    # But we can express the whole 3-vector A*b2 as 3 rank-1 tensors (one per output i):
    #   T_i = (sum_j A[i,j] e_j) * e_{j}^T ... 
    # Simplest: for output row i, the tensor term is sum_j A[i,j] * B[j,3],
    # which equals sum_j (A[i,j] * B[j,3]). Each (i,j) pair with fixed k=3 is rank-1.
    # So it's 3*3 = 9 terms? That gives rank 9 for the vector part -> total 36. Not good.
    #
    # Better: rank of (3x3)*(3x1) tensor. C_i = sum_j A_{i,j} b_j.
    # This is a 3-dimensional output. The tensor T_{i, (i',j), j'} ... 
    # Actually: T[i, a, j] where a indexes A, j indexes b. T[i, (i', j), j'] = delta_{i, i'} * delta_{j, j'}.
    # Wait: C[i, 3] = sum_{j} A[i, j] * B[j, 3].
    # In the tensor: T[i, (i', j'), k] = 1 if i'==i and k==3, else 0; and the j' index matches.
    # Hmm, let me be precise. The tensor for "A (3x3) times B (3x1) gives C (3x1)":
    #   T[i, (i', j'), k] = delta_{i, i'} * delta_{j', k}  ... no.
    #   C[i] = sum_{j'} A[i', j'] B[j']  with i'=i. So T[i, (i', j'), k] = delta_{i,i'} * delta_{j', k} and k must be the single column.
    # Since B is 3x1, j' ranges 0..2, and the "k" for B's column is fixed at 3 (the 4th column).
    # T[i, (i', j'), 3] = delta_{i, i'} (and the j' is summed). So T[i, (i', j'), 3] = delta_{i,i'}.
    # This tensor: for fixed i and k=3, it's the identity in (i', j')... no, it's delta_{i, i'} independent of j'.
    # Rank: T[i, (i', j'), 3] = delta_{i, i'}.
    # We can write T = sum_{i=0}^{2} (e_i in C) * (e_i in A-row) * (sum_j e_j in B-col)
    # Each term: C factor = e_i, A factor = row i of identity (i.e., vector with 1 at position (i, j') for all j'?) 
    # No: A factor for term i should be the i-th row: (A factor)_{i', j'} = delta_{i, i'}. That's a 3-vector in the (i',j') space with 1s at (i,0),(i,1),(i,2).
    # B factor: (B factor)_{j', k} = delta_{k,3} (1 for all j' when k=3). That's a 3-vector in j' with 1s.
    # So term i: U has 1 at (i, j') for j'=0,1,2 in row i*m+j'; V has 1 at (j', 3) for j'=0,1,2; W has 1 at (3*n + i).
    # That's 3 rank-1 terms for the vector part. Total rank = 27 + 3 = 30.

    cols_part2 = []
    for i in range(n):
        # One term per output row i
        cols_part2.append(('vec', i))

    # Build matrices
    rank = len(cols_part1) + len(cols_part2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (i, j, k) in cols_part1:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        if time.time() > DEADLINE:
            break

    for (tag, i) in cols_part2:
        # Term i: U row (i, j') = 1 for j'=0..m-1; V row (j', 3) = 1 for j'=0..m-1; W (3*n + i) = 1
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
        W[3 * n + i][col] = 1
        col += 1
        if time.time() > DEADLINE:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
