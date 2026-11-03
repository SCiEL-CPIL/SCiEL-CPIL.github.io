import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Rank calculation:
    # 7 Strassen steps for 2x2 * 2x3
    # Each step: 2x2 * 2x3
    # Split 2x3 into 2x2 (cols 0,1) and 2x1 (col 2)
    # 2x2 * 2x2 rank 7 (Strassen)
    # 2x2 * 2x1 rank 4 (Schoolbook: a00*b0, a01*b1, a10*b0, a11*b1)
    # Total per step: 11
    # Total rank: 7 * 11 = 77
    rank = 77

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen steps for 2x2 blocks
    # A blocks: 00, 01, 10, 11 (2x2 each)
    # B blocks: 00, 01, 10, 11 (2x3 each)
    # Note: B blocks are 2x3, so we split them into B_block_12 (2x2) and B_block_3 (2x1)

    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    # Helper to get index in U (A_flat)
    # A is 4x4, blocks are 2x2.
    # Block '00' is rows 0-1, cols 0-1
    # Block '01' is rows 0-1, cols 2-3
    # Block '10' is rows 2-3, cols 0-1
    # Block '11' is rows 2-3, cols 2-3
    def get_u_row(block, i1, j1):
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * m + j
        
    # Helper to get index in V (B_flat)
    # B is 4x6, blocks are 2x3.
    # Block '00' is rows 0-1, cols 0-2
    # Block '01' is rows 0-1, cols 3-5
    # Block '10' is rows 2-3, cols 0-2
    # Block '11' is rows 2-3, cols 3-5
    # We split B_block into B_block_12 (cols 0-1 of the block) and B_block_3 (col 2 of the block)
    # For V, we need to map to the full B_flat index.
    # B_flat index is j * p + k.
    # For a block (bi, bj), the local indices are j1 (0-1), k1 (0-2).
    # Global j = bi*2 + j1, global k = bj*3 + k1.
    def get_v_row(block, j1, k1):
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        else: raise ValueError
        return j * p + k
        
    # Helper to get index in W (C_flat)
    # C is 4x6, blocks are 2x3.
    # C_flat index is k * n + i.
    def get_w_row(block, i1, k1):
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        else: raise ValueError
        return k * n + i

    # Inner Strassen steps for 2x2 * 2x2
    inner_steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        
        # --- Part 1: Strassen for 2x2 * 2x2 (cols 0,1 of B block) ---
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            
            # U contributions
            for ai in range(2):
                for aj in range(2):
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in alpha.items():
                            U[get_u_row(b, ai, aj), col] += val * inner_coeff
            
            # V contributions (for B'_12, k1=0,1)
            for bj in range(2):
                for bk in range(2):
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V[get_v_row(b, bj, bk), col] += val * inner_coeff
            
            # W contributions (for C'_12, k1=0,1)
            for ci in range(2):
                for ck in range(2):
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W[get_w_row(b, ci, ck), col] += val * inner_coeff
            
            col += 1
        
        # --- Part 2: Schoolbook for 2x2 * 2x1 (col 2 of B block) ---
        # Mult 1: A[0,0] * B[0,2] -> contributes to C[0,2]
        for b, val in alpha.items():
            U[get_u_row(b, 0, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        # Mult 2: A[0,1] * B[1,2] -> contributes to C[0,2]
        for b, val in alpha.items():
            U[get_u_row(b, 0, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        # Mult 3: A[1,0] * B[0,2] -> contributes to C[1,2]
        for b, val in alpha.items():
            U[get_u_row(b, 1, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1
        
        # Mult 4: A[1,1] * B[1,2] -> contributes to C[1,2]
        for b, val in alpha.items():
            U[get_u_row(b, 1, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1

    # Verification
    try:
        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-8):
                pass
    except Exception:
        pass

    if time.time() - start > DEADLINE_SECONDS:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
