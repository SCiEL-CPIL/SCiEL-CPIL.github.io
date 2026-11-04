"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p):
    """Verify decomposition using matrix multiplication check."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for c in range(rank):
        u_col = U_arr[:, c].reshape(n, m)
        v_col = V_arr[:, c].reshape(m, p)
        w_col = W_arr[:, c].reshape(p, n)
        prod = u_col @ v_col
        if not np.allclose(prod, w_col.T, atol=1e-9):
            return False
    return True


def _strassen_2x2x2(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2x2 block. Returns new column index."""
    # Strassen's 7 multiplications for 2x2 matrix multiply
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U index: (di, dj) in 2x2 block -> global (i0+di)*m + (j0+dj)
    # V index: (dj, dk) in 2x2 block -> global (j0+dj)*p + (k0+dk)
    # W index: (dk, di) in 2x2 block -> global (k0+dk)*n + (i0+di)

    terms = [
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # M2: (A10+A11)B00 -> C10+=1, C11-=1
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # M3: A00(B10-B11) -> C01+=1, C11+=1
        {
            'U': [(0, 0, 1)],
            'V': [(1, 0, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # M4: A11(B01-B00) -> C00+=1, C10+=1
        {
            'U': [(1, 1, 1)],
            'V': [(0, 1, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        # M6: (A10-A00)(B00+B01) -> C11+=1
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # M7: (A01-A11)(B10+B11) -> C00+=1
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        },
    ]

    for t in terms:
        c_idx = col
        for (di, dj, sign) in t['U']:
            U[(i0 + di) * m + (j0 + dj)][c_idx] += sign
        for (dj, dk, sign) in t['V']:
            V[(j0 + dj) * p + (k0 + dk)][c_idx] += sign
        for (dk, di, sign) in t['W']:
            W[(k0 + dk) * n + (i0 + di)][c_idx] += sign
        col += 1

    return col


def _winograd_1x3x3(i0, j0, k0, n, m, p, U, V, W, col):
    """
    Apply Winograd's 1x3x3 convolution trick:
      c0 = a0*b0 + a1*b1 + a2*b2
      c1 = a0*b1 + a1*b2 + a2*b0
      c2 = a0*b2 + a1*b0 + a2*b1
    using 5 multiplications (FFT of length 3).
    U: 3 columns (one per input a_j), V: 3 columns (one per input b_k), W: 3 columns (one per output c_l)
    Total: 5 columns.
    """
    # FFT coefficients for length 3 (using primitive cube root of unity omega)
    # omega = exp(2*pi*i/3) = -1/2 + i*sqrt(3)/2
    # We use the real-valued Winograd scheme:
    #   A = [1, 1, 1]       (DC)
    #   A = [1, -1, 0]      (Nyquist)
    #   A = [1, 0, -1]      (odd)
    # Actually for 1x3x3 convolution (circular), the standard Winograd uses:
    #   m = 3, r = 3, k = 1
    #   A = [[1,1,1],[1,-1,0],[1,0,-1]]  (3x3)
    #   B = same
    #   G = I (since k=1)
    #   M = I
    #   F = I
    # This gives 3 multiplications? No, that's not right for linear convolution.
    #
    # For LINEAR 1x3x3 convolution, we need 5 multiplications:
    #   c0 = a0*b0 + a1*b1 + a2*b2
    #   c1 = a0*b1 + a1*b2 + a2*b0
    #   c2 = a0*b2 + a1*b0 + a2*b1
    #
    # Winograd's 5-mult scheme:
    #   M1 = (a0+a1)(b0+b1)
    #   M2 = (a1+a2)(b1+b2)
    #   M3 = (a0-a2)(b0-b2)
    #   M4 = a0*b2
    #   M5 = a2*b0
    #   c0 = M1 + M2 - M3 - M4 - M5 + a1*b1 ... this is getting complex.
    #
    # Let's just use the 5-mult FFT-based scheme:
    # Evaluate at 3 points, but we need 5 for linear conv. Actually, for 3-point circular conv, 3 mults suffice.
    # But we need LINEAR conv, which is 3-point circular conv on a 5-point zero-padded signal.
    # That requires 5 mults.
    #
    # Simpler: just use schoolbook for this block. 3*3*1 = 9 mults.
    # But we can do better. Let's use the fact that this is a 1x3x3 tensor.
    # The minimal rank of 1x3x3 matrix mult tensor is 5 (Winograd).
    #
    # Winograd 1x3x3:
    #   c0 = a0*b0 + a1*b1 + a2*b2
    #   c1 = a0*b1 + a1*b2 + a2*b0
    #   c2 = a0*b2 + a1*b0 + a2*b1
    #
    # 5-mult scheme:
    #   M1 = (a0 + a1) * (b0 + b1)
    #   M2 = (a1 + a2) * (b1 + b2)
    #   M3 = (a0 - a2) * (b0 - b2)
    #   M4 = a0 * b2
    #   M5 = a2 * b0
    #
    #   c0 = M1 + M2 - M3 - M4 - M5 + a1*b1
    #       = M1 + M2 - M3 - M4 - M5 + (M1 - M2 - M3 + M4 + M5)/... 
    #
    # This is getting too complex. Let's just use schoolbook for this block.
    # 3*3*1 = 9. But we can save by noting the structure.
    #
    # Actually, for this task, let's just use schoolbook for the 1x3x3 block.
    pass

    # Schoolbook for 1x3x3: 9 multiplications
    for di in range(1):
        for dj in range(3):
            for dk in range(3):
                U[(i0 + di) * m + (j0 + dj)][col] = 1
                V[(j0 + dj) * p + (k0 + dk)][col] = 1
                W[(k0 + dk) * n + (i0 + di)][col] = 1
                col += 1

    return col


def _schoolbook_block(i0, j0, k0, hi, hj, hk, n, m, p, U, V, W, col):
    """Schoolbook for hi x hj x hk block. Returns new column index."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] = 1
                V[(j0 + dj) * p + (k0 + dk)][col] = 1
                W[(k0 + dk) * n + (i0 + di)][col] = 1
                col += 1
    return col


def _construct_structured():
    """
    Structured algebraic decomposition:
    - 2x2x2 core at (0,0,0): Strassen, rank 7
    - 1x3x3 slab at (2,0,0): schoolbook, rank 9
    - 1x2x7 slab at (2,3,0): schoolbook, rank 14
    - 1x2x3 slab at (1,3,0): schoolbook, rank 6
    Wait, let's partition properly.
    
    n=3, m=5, p=7
    Partition i: [2, 1]
    Partition j: [2, 3]
    Partition k: [7]
    
    Blocks:
    (0,0,0): 2x2x7 -> can we do better?
    (0,1,0): 2x3x7
    (1,0,0): 1x2x7
    (1,1,0): 1x3x7
    
    Total schoolbook: 2*2*7 + 2*3*7 + 1*2*7 + 1*3*7 = 28 + 42 + 14 + 21 = 105.
    
    Let's try a different partition:
    i: [2, 1]
    j: [2, 2, 1]
    k: [2, 2, 3]
    
    This is getting complex. Let's just use the block approach with Strassen for 2x2x2.
    """
    n, m, p = DIMENSIONS
    
    # Use the block decomposition approach but with a fixed, good partition
    # i: [2, 1], j: [2, 3], k: [2, 5]
    splits_i = [2, 1]
    splits_j = [2, 3]
    splits_k = [2, 5]
    
    i_offsets = [0, 2]
    j_offsets = [0, 2]
    k_offsets = [0, 2]
    
    total_rank = 0
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                if hi == 2 and hj == 2 and hk == 2:
                    total_rank += 7
                else:
                    total_rank += hi * hj * hk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                
                if hi == 2 and hj == 2 and hk == 2:
                    col = _strassen_2x2x2(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _schoolbook_block(i0, j0, k0, hi, hj, hk, n, m, p, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Start with schoolbook
    best_result = _construct_schoolbook()
    best_rank = 3 * 5 * 7

    # Try structured block decomposition
    if time.time() < deadline:
        structured_result = _construct_structured()
        structured_rank = len(structured_result['U'][0])
        if _verify(structured_result['U'], structured_result['V'], structured_result['W'], 3, 5, 7):
            if structured_rank < best_rank:
                best_result = structured_result
                best_rank = structured_rank

    return best_result
# EVOLVE-BLOCK-END
