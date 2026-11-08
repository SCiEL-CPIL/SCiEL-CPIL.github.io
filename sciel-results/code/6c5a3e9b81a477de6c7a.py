import numpy as np
import time

def _make_strassen_block(n, p):
    """
    Return (U, V, W) for multiplying a 2x2 matrix by a 2xp matrix.
    U: (4, 7), V: (2*p, 7), W: (p*2, 7)
    W uses trace convention: W[k*n + i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=np.int32)
    V = np.zeros((2 * p, rank), dtype=np.int32)
    W = np.zeros((p * 2, rank), dtype=np.int32)

    u_a, u_b, u_c, u_d = 0, 1, 2, 3

    # M1 = (a+d)(e+h)
    U[u_a, 0] += 1
    U[u_d, 0] += 1
    V[0 * p + 0, 0] += 1  # e = B[0,0]
    V[1 * p + 1, 0] += 1  # h = B[1,1]

    # M2 = (c+d)e
    U[u_c, 1] += 1
    U[u_d, 1] += 1
    V[0 * p + 0, 1] += 1

    # M3 = a(f-h)
    U[u_a, 2] += 1
    V[0 * p + 1, 2] += 1  # f = B[0,1]
    V[1 * p + 1, 2] -= 1

    # M4 = d(g-e)
    U[u_d, 3] += 1
    V[1 * p + 0, 3] += 1  # g = B[1,0]
    V[0 * p + 0, 3] -= 1

    # M5 = (a+b)h
    U[u_a, 4] += 1
    U[u_b, 4] += 1
    V[1 * p + 1, 4] += 1

    # M6 = (c-a)(e+f)
    U[u_c, 5] += 1
    U[u_a, 5] -= 1
    V[0 * p + 0, 5] += 1
    V[0 * p + 1, 5] += 1

    # M7 = (b-d)(g+h)
    U[u_b, 6] += 1
    U[u_d, 6] -= 1
    V[1 * p + 0, 6] += 1
    V[1 * p + 1, 6] += 1

    # W indices: C[k, i] = sum_r U[i,j,r] * V[j,k,r] * W[k*n+i, r]
    # C[0,0] = M1 + M4 - M5 + M7
    w_00 = 0 * n + 0
    W[w_00, 0] += 1
    W[w_00, 3] += 1
    W[w_00, 4] -= 1
    W[w_00, 6] += 1

    # C[0,1] = M3 + M5
    w_01 = 1 * n + 0
    W[w_01, 2] += 1
    W[w_01, 4] += 1

    # C[1,0] = M2 + M4
    w_10 = 0 * n + 1
    W[w_10, 1] += 1
    W[w_10, 3] += 1

    # C[1,1] = M1 - M2 + M3 + M6
    w_11 = 1 * n + 1
    W[w_11, 0] += 1
    W[w_11, 1] -= 1
    W[w_11, 2] += 1
    W[w_11, 5] += 1

    return U, V, W


def construct():
    n, m, p = 2, 3, 9
    start_time = time.time()

    # We want to multiply A (n x m) = (2x3) by B (m x p) = (3x9) to get C (n x p) = (2x9)
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (27, rank)
    # W: (p*n, rank) = (18, rank)

    # Strategy: Split the middle dimension m=3 into blocks.
    # Option A: Split into 1+1+1. Each block is 2x1x9.
    #   2x1x9 can be done with rank 2*9 = 18 (outer product of 2-vector and 9-vector).
    #   Total: 3 * 18 = 54.
    #
    # Option B: Split into 2+1.
    #   2x2x9 block. Split 9 into 4+4+1.
    #   2x2x4: Use Strassen on pairs? No, Strassen is for square.
    #   Actually, we can split 9 into 2+2+2+2+1.
    #   2x2x2 uses Strassen (rank 7). 4 of them = 28.
    #   2x2x1 uses direct (rank 4). 1 of them = 4.
    #   Total for 2x2x9: 28 + 4 = 32.
    #   2x1x9 uses direct (rank 18).
    #   Total: 32 + 18 = 50.
    #
    # Let's try Option B.

    rank_total = 50
    U = np.zeros((n * m, rank_total), dtype=np.int32)
    V = np.zeros((m * p, rank_total), dtype=np.int32)
    W = np.zeros((p * n, rank_total), dtype=np.int32)

    col_offset = 0

    # --- Block 1: 2x2x9 (A rows 0:2, cols 0:2; B rows 0:2, cols 0:9) ---
    # Sub-blocks of 2x2x2 (4 blocks) and 2x2x1 (1 block)
    num_strassen = 4
    for s in range(num_strassen):
        # Strassen block for columns 2*s and 2*s+1
        Us, Vs, Ws = _make_strassen_block(n, 2)  # Actually p=2 for the block B submatrix
        # Map indices
        # U: A rows 0:2, cols 0:2. U indices are just 0:4.
        U[0:4, col_offset:col_offset+7] += Us
        # V: B rows 0:2, cols 2*s:2*s+2.
        # V is flattened (m x p). Row j, col k is j*p + k.
        for r in range(2):
            for c in range(2):
                v_idx = r * p + (2 * s + c)
                V[v_idx, col_offset:col_offset+7] += Vs[r * 2 + c, :]
        # W: C rows 0:2, cols 2*s:2*s+2.
        # W is flattened (p x n). C[k, i] is at index k*n + i.
        for k in range(2):
            for i in range(2):
                w_idx = (2 * s + k) * n + i
                W[w_idx, col_offset:col_offset+7] += Ws[k * 2 + i, :]
        col_offset += 7

    # Last Strassen-like block is actually just direct 2x2x1 for col 8
    # Wait, I said 4 blocks of 2, so cols 0,1; 2,3; 4,5; 6,7. That's 8 cols.
    # Col 8 remains.
    # Direct 2x2x1 for col 8.
    U[0, col_offset] += 1   # a
    V[0 * p + 8, col_offset] += 1 # e
    W[8 * n + 0, col_offset] += 1 # C[8,0]
    
    U[1, col_offset+1] += 1 # b
    V[1 * p + 8, col_offset+1] += 1 # h
    W[8 * n + 0, col_offset+1] += 1 # C[8,0]
    
    U[2, col_offset+2] += 1 # c
    V[0 * p + 8, col_offset+2] += 1 # e
    W[8 * n + 1, col_offset+2] += 1 # C[8,1]
    
    U[3, col_offset+3] += 1 # d
    V[1 * p + 8, col_offset+3] += 1 # h
    W[8 * n + 1, col_offset+3] += 1 # C[8,1]
    
    col_offset += 4

    # --- Block 2: 2x1x9 (A rows 0:2, col 2; B row 2, cols 0:9) ---
    # A indices in U: row i, col 2 -> i*m + 2. So U[2] and U[5].
    # B indices in V: row 2, col k -> 2*p + k.
    # C indices in W: row k, col i -> k*n + i.
    
    # C[i,k] = A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx, col_offset] += 1
            V[v_idx, col_offset] += 1
            W[w_idx, col_offset] += 1
            col_offset += 1

    assert col_offset == rank_total, f"Rank mismatch: {col_offset} vs {rank_total}"

    # Quick check for time (though this is fast)
    if time.time() - start_time > 140:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
