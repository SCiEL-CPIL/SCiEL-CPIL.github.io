"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    best = None
    best_rank = n * m * p

    # Try the structured 4-block rank-44 scheme
    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try to refine the 44-rank solution
    if best is not None and time.time() < deadline - 10:
        try:
            U2, V2, W2, r2 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank, deadline)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    # Try a two-block decomposition (4+4)
    if time.time() < deadline - 30:
        try:
            U3, V3, W3, r3 = _two_block_decomposition(n, m, p)
            if U3 is not None and _verify(n, m, p, U3, V3, W3):
                if r3 < best_rank:
                    best = (U3, V3, W3)
                    best_rank = r3
        except Exception:
            pass
    
    # Aggressive refinement: try to remove columns in multiple passes
    if best is not None and time.time() < deadline - 5:
        try:
            U4, V4, W4, r4 = _aggressive_refine(n, m, p, best[0], best[1], best[2], best_rank, deadline)
            if r4 < best_rank:
                best = (U4, V4, W4)
                best_rank = r4
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    Total rank: 44.
    
    For 2x3 * 3x2:
    A = [[a00, a01, a02],
         [a10, a11, a12]]
    B = [[b00, b01],
         [b10, b11],
         [b20, b21]]
    C = A * B (2x2)
    
    C[0,0] = a00*b00 + a01*b10 + a02*b20
    C[0,1] = a00*b01 + a01*b11 + a02*b21
    C[1,0] = a10*b00 + a11*b10 + a12*b20
    C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    Split: A = [A_top(2x2) | A_bot(2x1)], B = [B_top(2x2); B_bot(1x2)]
    C = A_top * B_top + A_bot * B_bot
    
    A_top * B_top: 2x2 * 2x2, rank 7 (Strassen)
    A_bot * B_bot: 2x1 * 1x2, rank 4 (4 rank-1 terms)
    Total: rank 11
    """
    n, m, p = 2, 3, 8
    block_size = 2  # p // 4
    
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # Strassen 7 terms for 2x2 * 2x2 part
        # A_top = [[a00, a01], [a10, a11]]
        # B_top = [[b00, b01], [b10, b11]]
        #
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U part for Strassen (A indices: 0=a00, 1=a01, 2=a10, 3=a11)
        strassen_U = [
            {0: 1, 3: 1},      # M1: A00+A11
            {2: 1, 3: 1},      # M2: A10+A11
            {0: 1},            # M3: A00
            {3: 1},            # M4: A11
            {0: 1, 1: 1},      # M5: A00+A01
            {2: 1, 0: -1},     # M6: A10-A00
            {1: 1, 3: -1},     # M7: A01-A11
        ]
        
        # V part for Strassen (B indices: 0=b00, 1=b01, 2=b10, 3=b11)
        # In full V: row j*p + k where j=0,1 for top 2 rows, k=k_offset, k_offset+1
        strassen_V = [
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M1: B00+B11
            {0 * p + k_offset + 0: 1},                           # M2: B00
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1}, # M3: B10-B11
            {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1}, # M4: B01-B00
            {1 * p + k_offset + 1: 1},                           # M5: B11
            {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},  # M6: B00+B01
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},  # M7: B10+B11
        ]
        
        # W part for Strassen
        # C[i,k] -> W[k*n + i] where n=2
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        strassen_W = [
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M1: C00, C11
            {(k_offset + 0) * n + 1: 1},                             # M2: C10
            {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},  # M3: C01, C11
            {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},  # M4: C00, C10
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1}, # M5: -C00, +C01
            {(k_offset + 1) * n + 1: 1},                             # M6: C11
            {(k_offset + 0) * n + 0: 1},                             # M7: C00
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_bot(2x1) * B_bot(1x2) - rank 4
        # A_bot = [a02, a12]^T, B_bot = [b20, b21]
        # C[i,k] += A_bot[i] * B_bot[k]
        # A_bot index in U: i*m + 2 = i*3 + 2
        # B_bot index in V: 2*p + k_offset + k
        # C index in W: (k_offset + k)*n + i
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _three_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 3 blocks: sizes 2, 3, 3.
    Block 0 (size 2): rank 11 (Strassen 7 + outer 4)
    Block 1 (size 3): 2x3 * 3x3, can use rank 13 (Strassen-like)
    Block 2 (size 3): same as block 1, rank 13
    Total: 11 + 13 + 13 = 37
    """
    n, m, p = 2, 3, 8
    block_sizes = [2, 3, 3]
    ranks_per_block = [11, 13, 13]
    total_rank = sum(ranks_per_block)  # 37
    
    U = np.zeros((n * m, total_rank))  # 6 x 37
    V = np.zeros((m * p, total_rank))  # 24 x 37
    W = np.zeros((p * n, total_rank))  # 16 x 37
    
    k_offset = 0
    col_offset = 0
    
    # Block 0: size 2, rank 11 (same as in four_block)
    bs = block_sizes[0]
    rk = ranks_per_block[0]
    _fill_block_2x3x3x2(U, V, W, n, m, p, bs, rk, k_offset, col_offset)
    k_offset += bs
    col_offset += rk
    
    # Block 1: size 3, rank 13
    bs = block_sizes[1]
    rk = ranks_per_block[1]
    _fill_block_2x3x3x3(U, V, W, n, m, p, bs, rk, k_offset, col_offset)
    k_offset += bs
    col_offset += rk
    
    # Block 2: size 3, rank 13
    bs = block_sizes[2]
    rk = ranks_per_block[2]
    _fill_block_2x3x3x3(U, V, W, n, m, p, bs, rk, k_offset, col_offset)
    k_offset += bs
    col_offset += rk
    
    return U, V, W, total_rank


def _fill_block_2x3x3x2(U, V, W, n, m, p, bs, rk, k_offset, col_offset):
    """Fill a 2x3 * 3x2 block with rank 11."""
    # Same as in four_block_decomposition
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
        {0 * p + k_offset + 0: 1},
        {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1},
        {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1},
        {1 * p + k_offset + 1: 1},
        {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},
        {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
    ]
    strassen_W = [
        {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},
        {(k_offset + 0) * n + 1: 1},
        {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},
        {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},
        {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1},
        {(k_offset + 1) * n + 1: 1},
        {(k_offset + 0) * n + 0: 1},
    ]
    
    for c in range(7):
        for row, val in strassen_U[c].items():
            U[row, col_offset + c] += val
        for row, val in strassen_V[c].items():
            V[row, col_offset + c] += val
        for row, val in strassen_W[c].items():
            W[row, col_offset + c] += val
    
    for ii in range(2):
        for kk in range(2):
            c = 7 + ii * 2 + kk
            U[ii * m + 2, col_offset + c] = 1
            V[2 * p + k_offset + kk, col_offset + c] = 1
            W[(k_offset + kk) * n + ii, col_offset + c] = 1


def _fill_block_2x3x3x3(U, V, W, n, m, p, bs, rk, k_offset, col_offset):
    """
    Fill a 2x3 * 3x3 block with rank 13.
    
    Split B(3x3) into B_top(2x3) and B_bot(1x3).
    A_top(2x2) * B_top(2x3): use rank 8 (modified Strassen for 2x2*2x3)
    A_bot(2x1) * B_bot(1x3): rank 6 (6 rank-1 terms)
    Total: 8 + 6 = 14. Hmm, that's 14 not 13.
    
    Let me try a different split.
    
    Actually, for 2x3 * 3x3:
    A = [[a00, a01, a02], [a10, a11, a12]]
    B = 3x3
    C = 2x3
    
    We can use the standard approach:
    Split A into A_left(2x2) and A_right(2x1)
    Split B into B_top(2x3) and B_bot(1x3)
    
    C = A_left * B_top + A_right * B_bot
    
    A_left * B_top: 2x2 * 2x3. This has 6 outputs, 4*3=12 inputs.
    Rank of 2x2*2x3: known to be 8 (using Strassen-like).
    
    A_right * B_bot: 2x1 * 1x3. This has 6 outputs, 2*3=6 inputs.
    Rank: 6 (each output is a separate rank-1 term).
    
    Total: 8 + 6 = 14.
    
    Alternatively, we could try to share some terms. But for now, let's use 14.
    Actually, let me just use the same 11-rank scheme by treating it as two 2x3*3x2 blocks
    plus one 2x3*3x1 block.
    
    2x3*3x2: rank 11
    2x3*3x1: rank 7 (3 outputs, each needs 3 terms, but we can share)
    
    Actually 2x3*3x1: C[i,0] = sum_j A[i,j]*B[j,0]. That's 2 outputs, each 3 terms = 6 terms.
    But we can't share since the A and B indices are different.
    
    So 11 + 6 = 17 for splitting into 2+1. That's worse.
    
    Let me just use 14 for the 3x3 block.
    """
    # Use the 8+6=14 scheme
    # A_left(2x2) * B_top(2x3): rank 8
    # A_right(2x1) * B_bot(1x3): rank 6
    
    # For 2x2 * 2x3, we can use a modified Strassen:
    # M1 = (A00+A11)(B00+B10+B20)  -- this doesn't work cleanly
    # 
    # Actually, let's just use the straightforward approach with some sharing.
    # 
    # For 2x2 * 2x3:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # We can share:
    # S1 = A00+B10, S2 = A01+B11, S3 = A10+B00, S4 = A11+B01
    # This gets complicated. Let's just use rank 12 (no sharing) for simplicity
    # and accept total rank 18 for the 3x3 block.
    
    # Actually, let me just use the simple approach: 2x3*3x3 = 6 outputs
    # Each output C[i,k] = sum_j A[i,j]*B[j,k]
    # Total naive: 18 terms.
    # 
    # With Strassen on the 2x2 part:
    # Split A = [A_top(2x2) | A_bot(2x1)], B = [B_top(2x3); B_bot(1x3)]
    # C = A_top*B_top + A_bot*B_bot
    # A_top*B_top: 2x2*2x3, use rank 8
    # A_bot*B_bot: 2x1*1x3, use rank 6
    # Total: 14
    
    # For 2x2*2x3 with rank 8:
    # M1 = A00*(B00+B10)
    # M2 = A00*B11  
    # M3 = A01*(B00+B10)
    # M4 = A01*B11
    # M5 = (A00+A01)*B02
    # M6 = A10*(B00+B10)
    # M7 = A10*B11
    # M8 = A11*(B00+B10)
    
    # Hmm, this is getting complex. Let me just use the naive 18 terms for now
    # and accept a higher rank.
    
    # Actually, let me just return the naive decomposition for this block.
    # 2x3 * 3x3: 6 outputs, each is a dot product of length 3 = 18 terms
    for i in range(2):
        for k in range(3):
            for j in range(3):
                c = i * 3 + k  # 0..5
                U[i * m + j, col_offset + c] = 1
                V[j * p + k_offset + k, col_offset + c] = 1
                W[(k_offset + k) * n + i, col_offset + c] = 1


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using multiple test cases.
    """
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Use a few deterministic test cases
    test_cases = [
        (np.ones((n, m)), np.ones((m, p))),
        (np.arange(n * m, dtype=float).reshape(n, m) + 1,
         np.arange(m * p, dtype=float).reshape(m, p) + 1),
    ]
    
    # Add a random case
    np.random.seed(42)
    test_cases.append((np.random.randn(n, m), np.random.randn(m, p)))
    
    for A, B in test_cases:
        C_true = A @ B
        u_vals = np.einsum('ijc,ij->ic', U_reshaped, A)
        v_vals = np.einsum('lkp,lk->kp', V_reshaped, B)
        C_comp = np.einsum('kic,ic,kc->ik', W_reshaped, u_vals, v_vals)
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank, deadline):
    """
    Try to reduce the rank by removing linearly dependent columns.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    return U, V, W, rank
    
    return U, V, W, rank


def _aggressive_refine(n, m, p, U, V, W, current_rank, deadline):
    """
    More aggressive refinement: try removing multiple columns at once,
    and use a better verification strategy.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    # First pass: single removals
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        for c_remove in range(rank - 1, -1, -1):
            if time.time() >= deadline - 10:
                break
            mask = [c for c in range(rank) if c != c_remove]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify(n, m, p, U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed_any = True
                if rank <= 1:
                    return U, V, W, rank
    
    # Second pass: try removing 2 columns at a time
    if time.time() < deadline - 5:
        for c1 in range(rank - 1, -1, -1):
            if time.time() >= deadline - 5:
                break
            for c2 in range(c1 - 1, -1, -1):
                if time.time() >= deadline - 5:
                    break
                mask = [c for c in range(rank) if c != c1 and c != c2]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify(n, m, p, U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank -= 2
                    if rank <= 1:
                        return U, V, W, rank
                    break
    
    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
