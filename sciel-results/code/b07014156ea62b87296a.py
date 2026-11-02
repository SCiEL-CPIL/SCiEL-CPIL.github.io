import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Use overlapping Strassen blocks to cover more of the tensor.
    # Blocks cover 2x2x2 subcubes. We place them at all valid (i0,j0,k0)
    # where i0 in {0,1}, j0 in {0,2}, k0 in {0,2} — but also try i0=1.
    #
    # Strategy: place blocks at every valid position to maximize coverage,
    # then do schoolbook for uncovered terms. Overlapping blocks are fine
    # because each block independently contributes to the tensor; the sum
    # of all contributions must equal the target.
    #
    # Valid positions: i0 in {0,1} (since n=3, i0+1 <= 2), j0 in {0,2,3} (j0+1 <= 4), k0 in {0,2,3}
    # But we want 2x2 blocks, so j0 can be 0,2 (covering 0-1, 2-3) or 3 (covering 3-4).
    # Similarly k0 can be 0,2 or 3.
    
    # Try a set of blocks that maximizes coverage:
    # i0=0: covers i=0,1
    # i0=1: covers i=1,2
    # j0=0: covers j=0,1; j0=2: covers j=2,3; j0=3: covers j=3,4
    # k0=0: covers k=0,1; k0=2: covers k=2,3; k0=3: covers k=3,4
    
    # To minimize total rank, we want to minimize (num_blocks * 7 + num_uncovered_terms).
    # Each block covers 8 terms with rank 7, saving 1 per block.
    # But overlapping blocks may cover the same terms multiple times.
    
    # Let's try: all combinations of i0 in {0,1}, j0 in {0,2}, k0 in {0,2}
    # That's 2*2*2 = 8 blocks.
    blocks = []
    for i0 in [0, 1]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                blocks.append((i0, j0, k0))
    
    # Also try adding blocks at j0=3, k0=3 to cover the edges
    # j0=3 covers j=3,4; k0=3 covers k=3,4
    # Add (0,3,3) and (1,3,3) to cover the j=3,4; k=3,4 corner
    blocks.append((0, 3, 3))
    blocks.append((1, 3, 3))
    
    num_blocks = len(blocks)
    n_strassen = num_blocks * 7
    
    # Determine which terms are covered (appear in at least one block)
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di < n and j0+dj < m and k0+dk < p:
                        covered[i0+di][j0+dj][k0+dk] = True
    
    num_uncovered = sum(covered[i][j][k] == False for i in range(n) for j in range(m) for k in range(p))
    total_rank = n_strassen + num_uncovered
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Schoolbook for uncovered terms only
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1

    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    if not np.allclose(T_decomp, target, atol=1e-10):
        # Fallback to schoolbook
        rank_s = n * m * p
        U_f = [[0]*rank_s for _ in range(n*m)]
        V_f = [[0]*rank_s for _ in range(m*p)]
        W_f = [[0]*rank_s for _ in range(p*n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c2] = 1
                    V_f[j*p+k][c2] = 1
                    W_f[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
