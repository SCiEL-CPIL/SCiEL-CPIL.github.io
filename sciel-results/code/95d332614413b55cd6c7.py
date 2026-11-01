import numpy as np
import time

# Matrix multiplication tensor for C[2,10] = A[2,3] * B[3,10]
# Index convention: U indexed by (i*m + j), V by (j*p + k), W by (k*n + i)
# T[i,j,k] = 1 (scalar factor), so sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

def _build_block(U, V, W, i0, i1, j0, j1, k0, k1, c0, n, m, p):
    """
    Embed a 2x2x2 Strassen block for submatrices:
      rows i0..i1 (2 rows), cols j0..j1 (2 cols) in A,
      rows j0..j1 in B, cols k0..k1 in B,
      output rows i0..i1, cols k0..k1.
    Returns next free column index.
    """
    # 7 Strassen terms for 2x2x2 matmul
    # Each term: (U_pattern, V_pattern, W_pattern)
    # U_pattern: dict mapping (li, lj) -> coeff  where li in {0,1}, lj in {0,1}
    # V_pattern: dict mapping (lj, lk) -> coeff
    # W_pattern: dict mapping (lk, li) -> coeff
    strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1, (0,1):-1}, {(1,0):1}, {(1,0):1, (1,1):-1}),
        ({(1,0):-1, (1,1):1}, {(0,0):1, (0,1):-1}, {(0,0):1, (0,1):-1}),
        ({(0,0):1, (0,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1}),
        ({(0,0):1, (0,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1}),
        ({(0,1):-1, (1,0):1}, {(0,0):1, (1,1):1}, {(1,0):1, (1,1):-1}),
    ]
    
    for u_pat, v_pat, w_pat in strassen:
        c = c0
        for (li, lj), cv in u_pat.items():
            gi = i0 + li
            gj = j0 + lj
            U[gi * m + gj][c] += cv
        for (lj, lk), cv in v_pat.items():
            gj = j0 + lj
            gk = k0 + lk
            V[gj * p + gk][c] += cv
        for (lk, li), cv in w_pat.items():
            gk = k0 + lk
            gi = i0 + li
            W[gk * n + gi][c] += cv
        c0 += 1
    
    return c0

def _build_schoolbook_row(U, V, W, i, j, k0, k1, c0, n, m, p):
    """
    Schoolbook: compute C[i, k] += A[i, j] * B[j, k] for k in k0..k1.
    Rank = (k1 - k0) per (i,j) pair.
    Returns next free column.
    """
    for gk in range(k0, k1):
        U[i * m + j][c0] = 1.0
        V[j * p + gk][c0] = 1.0
        W[gk * n + i][c0] = 1.0
        c0 += 1
    return c0

def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin before 120s limit

    n, m, p = 2, 3, 10
    
    # Strategy:
    # For the 2x3x10 matmul, we split columns of B into:
    #   - First 8 columns: process as 4 pairs using 2x2x2 Strassen blocks
    #   - Last 2 columns: schoolbook for each row i
    #
    # But actually m=3, so we can do:
    #   Block 1: rows 0,1 of A; cols 0,1 of B (2x2x2 Strassen) -> for k pairs
    #   Block 2: rows 0,1 of A; col 2 of B (2x1) -> schoolbook
    #
    # Let's process columns of B in pairs for the first 8 columns:
    #   For k_pairs = 0..3 (covering k=0,1; 2,3; 4,5; 6,7):
    #     Use Strassen for A[:, 0:2] * B[0:2, k_pair:k_pair+2]
    #   For k=8,9 (last 2 columns):
    #     Schoolbook for all 3 rows of A
    #
    # Total rank:
    #   4 Strassen blocks * 7 = 28
    #   Last 2 columns * 3 rows * 2 rows of output = 12
    #   Plus we still need to handle row 2 of A for the first 8 columns
    #
    # Better approach:
    #   Process columns 0-7 with Strassen on first 2 rows of B, schoolbook for row 2
    #   Process columns 8-9 with full schoolbook
    #
    # For columns 0-7 (4 pairs):
    #   Strassen for A[0:2, 0:2] * B[0:2, pair] : 4*7 = 28
    #   Schoolbook for A[0:2, 2] * B[2, pair] : 4*2*2 = 16
    #   Total for cols 0-7: 44
    # For columns 8-9:
    #   Schoolbook for all: 2*3*2 = 12
    # Total: 56
    #
    # Even better: use Strassen for the full 2x3 by treating it as 2x2 + 2x1
    # but that's what we have. Let me try a cleaner decomposition.
    #
    # Alternative: process all 10 columns with 5 pairs, using Strassen on 2x2 
    # subblock and schoolbook on the remaining row.
    
    total_rank = 55  # We'll aim for this
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    c = 0
    
    # Process columns in pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    for pair_idx in range(5):
        k0 = pair_idx * 2
        k1 = k0 + 2
        
        if time.time() - start_time > DEADLINE:
            break
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1]
        c = _build_block(U, V, W, 0, 2, 0, 2, k0, k1, c, n, m, p)
        
        # Schoolbook for A[0:2, 2] * B[2, k0:k1]
        c = _build_schoolbook_row(U, V, W, 0, 2, k0, k1, c, n, m, p)
        c = _build_schoolbook_row(U, V, W, 1, 2, k0, k1, c, n, m, p)
    
    actual_rank = c
    
    # Trim to actual rank
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for cc in range(actual_rank):
                    s += U[i * m + j, cc] * V[j * p + k, cc] * W[k * n + i, cc]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        raise ValueError(f"Verification failed with max error {max_err}")
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        raise TimeoutError(f"Time exceeded: {elapsed}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
