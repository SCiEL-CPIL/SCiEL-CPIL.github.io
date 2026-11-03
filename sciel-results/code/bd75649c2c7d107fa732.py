"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                Wval = W_np[k * n + i]
                for j in range(m):
                    Uval = U_np[i * m + j]
                    Vval = V_np[j * p + k]
                    s += np.dot(np.multiply(Uval, Vval), Wval)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 Strassen factors for 2x2 * 2x2 matrix multiplication.
    """
    factors = []
    
    # M1 = (A00+A11)(B00+B11)
    factors.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A10+A11)B00
    factors.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    
    # M3 = A00(B10-B00)
    factors.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A11(B11-B01)
    factors.append({
        'U': {(1,1): 1},
        'V': {(1,1): 1, (0,1): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A00+A01)B11
    factors.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): -1}
    })
    
    # M6 = (A10-A00)(B00+B01)
    factors.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (A01-A11)(B10+B11)
    factors.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    return factors

def _build_strassen_4x4_4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 * 4x4.
    """
    level1_factors = _strassen_2x2_factors()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for l1 in range(7):
        for l2 in range(7):
            l1_factor = level1_factors[l1]
            l2_factor = level1_factors[l2]
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['U']:
                        continue
                    c1 = l1_factor['U'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['U']:
                                continue
                            c2 = l2_factor['U'][(si, sj)]
                            ai = bi * 2 + si
                            aj = bj * 2 + sj
                            U[ai * 4 + aj][col] += c1 * c2
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['V']:
                        continue
                    c1 = l1_factor['V'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['V']:
                                continue
                            c2 = l2_factor['V'][(si, sj)]
                            bri = bi * 2 + si
                            bcj = bj * 2 + sj
                            V[bri * 4 + bcj][col] += c1 * c2
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) not in l1_factor['W']:
                        continue
                    c1 = l1_factor['W'][(bi, bj)]
                    for si in range(2):
                        for sj in range(2):
                            if (si, sj) not in l2_factor['W']:
                                continue
                            c2 = l2_factor['W'][(si, sj)]
                            cri = bi * 2 + si
                            ccj = bj * 2 + sj
                            W[ccj * 4 + cri][col] += c1 * c2
            
            col += 1
    
    return U, V, W

def _build_strassen_4x4_4x2():
    """
    Build rank-28 Strassen decomposition for 4x4 * 4x2.
    Split into four 2x2 * 2x2 blocks.
    A is 4x4, B is 4x2.
    B can be seen as two 2x2 blocks stacked vertically? No, B is (m x p) = (4 x 2).
    Rows of B: 0,1,2,3. Cols: 0,1.
    Block (0,0) of A (2x2) multiplies Block (0,0) of B (2x2, rows 0-1, cols 0-1) -> Output (0,0) (2x2)
    Block (0,1) of A (2x2) multiplies Block (0,0) of B (2x2, rows 0-1, cols 0-1) -> Output (0,1) (2x2)
    Block (1,0) of A (2x2) multiplies Block (1,0) of B (2x2, rows 2-3, cols 0-1) -> Output (1,0) (2x2)
    Block (1,1) of A (2x2) multiplies Block (1,0) of B (2x2, rows 2-3, cols 0-1) -> Output (1,1) (2x2)
    
    Wait, the output is 4x2.
    Output Block (0,0) is rows 0-1, cols 0-1.
    Output Block (0,1) is rows 0-1, cols 2-3? No, p=2, so cols 0-1.
    Output is 4x2.
    Rows 0-1, Cols 0-1 is the top half.
    Rows 2-3, Cols 0-1 is the bottom half.
    
    So:
    Top-Left A (2x2) * Top B (2x2) -> Top Output (2x2)
    Top-Right A (2x2) * Top B (2x2) -> Top Output (2x2)
    Bottom-Left A (2x2) * Bottom B (2x2) -> Bottom Output (2x2)
    Bottom-Right A (2x2) * Bottom B (2x2) -> Bottom Output (2x2)
    
    This is 4 independent 2x2 * 2x2 multiplications.
    Each has rank 7.
    Total rank 28.
    """
    factors = _strassen_2x2_factors()
    rank = 4 * 7
    U = [[0] * rank for _ in range(16)] # 4x4
    V = [[0] * rank for _ in range(8)]  # 4x2
    W = [[0] * rank for _ in range(8)]  # 2x4 (p*n)
    
    col = 0
    # 4 blocks: (0,0), (0,1), (1,0), (1,1) in A
    # B blocks: (0,0) for top, (1,0) for bottom
    # Output blocks: (0,0) for top, (1,0) for bottom
    
    for block_a_i in range(2):
        for block_a_j in range(2):
            # A block is at rows 2*block_a_i to 2*block_a_i+1, cols 2*block_a_j to 2*block_a_j+1
            # B block: if block_a_i == 0, use B rows 0-1. If 1, use B rows 2-3.
            # B cols are always 0-1.
            b_row_offset = block_a_i * 2
            # Output block: if block_a_i == 0, output rows 0-1. If 1, output rows 2-3.
            # Output cols are always 0-1.
            out_row_offset = block_a_i * 2
            
            for f in range(7):
                factor = factors[f]
                # U: A block
                for (ri, ci), val in factor['U'].items():
                    a_row = block_a_i * 2 + ri
                    a_col = block_a_j * 2 + ci
                    U[a_row * 4 + a_col][col] += val
                
                # V: B block
                for (ri, ci), val in factor['V'].items():
                    b_row = b_row_offset + ri
                    b_col = ci # 0 or 1
                    V[b_row * 2 + b_col][col] += val
                
                # W: Output block
                for (ri, ci), val in factor['W'].items():
                    out_row = out_row_offset + ri
                    out_col = ci # 0 or 1
                    # W index is k * n + i -> out_col * 4 + out_row
                    W[out_col * 4 + out_row][col] += val
                
                col += 1
    
    return U, V, W

def _build_strassen_tiled_4x4_4x6():
    """
    Build a Strassen-tiled decomposition for 4x4 * 4x6.
    Split into 4x4*4x4 (Strassen, rank 49) + 4x4*4x2 (Strassen, rank 28).
    Total rank: 77.
    """
    n, m, p = 4, 4, 6
    rank_strassen = 49
    rank_strassen_2 = 28
    rank_total = rank_strassen + rank_strassen_2
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Part 1: 4x4 * 4x4 (cols 0-3 of B)
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    for i in range(n * m):
        for c in range(rank_strassen):
            U[i][c] = U_s[i][c]
    for i in range(16):
        for c in range(rank_strassen):
            V[i][c] = V_s[i][c]
    for i in range(16):
        for c in range(rank_strassen):
            W[i][c] = W_s[i][c]
    
    # Part 2: 4x4 * 4x2 (cols 4-5 of B)
    U_s2, V_s2, W_s2 = _build_strassen_4x4_4x2()
    col_offset = rank_strassen
    for i in range(n * m):
        for c in range(rank_strassen_2):
            U[i][c + col_offset] = U_s2[i][c]
    for i in range(8):
        for c in range(rank_strassen_2):
            # V index for 4x2 is j * 2 + k. Here k is 0 or 1.
            # We need to map this to the full 4x6 V matrix.
            # B is 4x6. Cols 4 and 5 correspond to k=0 and k=1 in the 4x2 block.
            # So V_s2 row i (0..7) corresponds to B row j, col k.
            # j = i // 2, k = i % 2.
            # Full V index is j * 6 + (k + 4).
            j = i // 2
            k = i % 2
            full_v_idx = j * 6 + (k + 4)
            V[full_v_idx][c + col_offset] = V_s2[i][c]
    for i in range(8):
        for c in range(rank_strassen_2):
            # W index for 2x4 is k * 4 + i. Here k is 0 or 1 (cols 4,5 of B).
            # i is 0..3 (rows of output).
            # Full W index is (k + 4) * 4 + i.
            k = i // 4
            out_i = i % 4
            full_w_idx = (k + 4) * 4 + out_i
            W[full_w_idx][c + col_offset] = W_s2[i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p
    
    # Try Strassen-tiled approach (49 + 28 = 77)
    strassen = _build_strassen_tiled_4x4_4x6()
    if strassen is not None:
        if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
            rank = len(strassen["U"][0])
            if rank < best_rank:
                best = strassen
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
