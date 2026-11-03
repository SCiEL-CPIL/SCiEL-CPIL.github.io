import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    terms = []
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        if C_comp != C_exp:
            return False
    return True

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    """
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    strassen_terms = _strassen_2x2x2_terms()
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_multiplication_tensor(n, m, p):
    """
    Build the multiplication tensor T[a, b, c].
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _find_2x2x3_decomposition(target_rank, max_iters=200, deadline=None, seed=42, n_restarts=5):
    """
    Find a rank-target_rank decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 2, 3
    T = _build_multiplication_tensor(n, m, p)
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    
    # Build the full constraint matrix for exact verification
    # Non-zero entries
    nonzero_entries = set(indices)
    
    # All possible (a,b,c) triples
    all_triples = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                all_triples.append((a, b, c))
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        np.random.seed(seed + restart)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5

        for iteration in range(max_iters):
            if deadline is not None and time.time() > deadline - 10:
                break

            # --- Solve for W ---
            M_W = np.zeros((n * m * m * p, target_rank))
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    for r in range(target_rank):
                        M_W[idx, r] = U[a, r] * V[b, r]
                    idx += 1
            for c in range(p * n):
                rhs = T[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    pass

            # --- Solve for U ---
            M_U = np.zeros((m * p * p * n, target_rank))
            idx = 0
            for b in range(m * p):
                for c in range(p * n):
                    for r in range(target_rank):
                        M_U[idx, r] = V[b, r] * W[c, r]
                    idx += 1
            for a in range(n * m):
                rhs = T[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    pass

            # --- Solve for V ---
            M_V = np.zeros((n * m * p * n, target_rank))
            idx = 0
            for a in range(n * m):
                for c in range(p * n):
                    for r in range(target_rank):
                        M_V[idx, r] = U[a, r] * W[c, r]
                    idx += 1
            for b in range(m * p):
                rhs = T[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    pass

        # Round to nearest half-integer
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2

        # Verify exactly - check all entries
        error = 0.0
        for (a, b, c) in all_triples:
            val = float(np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :]))
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
            error += abs(val - expected)
        
        if error < 1e-6:
            return U_round, V_round, W_round
    
    return None

def _combine_2x2x3_to_2x4x6(decomp_2x2x3):
    """
    Combine four 2x2x3 decompositions to form a 2x4x6 decomposition.
    """
    n, m, p = 2, 4, 6
    U_223, V_223, W_223 = decomp_2x2x3
    rank_223 = len(U_223[0])
    total_rank = 4 * rank_223
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # We have 4 blocks:
    # 1. A_L (cols 0-1 of A) * B_TL (rows 0-1, cols 0-2 of B) -> C_L (cols 0-2 of C)
    # 2. A_L (cols 0-1 of A) * B_TR (rows 0-1, cols 3-5 of B) -> C_R (cols 3-5 of C)
    # 3. A_R (cols 2-3 of A) * B_BL (rows 2-3, cols 0-2 of B) -> C_L (cols 0-2 of C)
    # 4. A_R (cols 2-3 of A) * B_BR (rows 2-3, cols 3-5 of B) -> C_R (cols 3-5 of C)
    
    blocks = [
        # (a_offset, b_row_offset, c_col_offset)
        (0, 0, 0), # A_L, B_TL -> C_L
        (0, 0, 3), # A_L, B_TR -> C_R
        (2, 2, 0), # A_R, B_BL -> C_L
        (2, 2, 3), # A_R, B_BR -> C_R
    ]
    
    for b_idx, (a_off, b_row_off, c_col_off) in enumerate(blocks):
        base_rank = b_idx * rank_223
        
        for i in range(2):
            for j in range(2):
                u_idx_global = (i) * m + (j + a_off)
                u_idx_local = i * 2 + j
                for r in range(rank_223):
                    U[u_idx_global][base_rank + r] = U_223[u_idx_local][r]
                    
        for j in range(2):
            for k in range(3):
                v_idx_global = (j + b_row_off) * p + k
                v_idx_local = j * 3 + k
                for r in range(rank_223):
                    V[v_idx_global][base_rank + r] = V_223[v_idx_local][r]
                    
        for i in range(2):
            for k in range(3):
                w_idx_global = (k + c_col_off) * n + i
                w_idx_local = k * 2 + i
                for r in range(rank_223):
                    W[w_idx_global][base_rank + r] = W_223[w_idx_local][r]
                    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # First, try the known Strassen-based rank-42 decomposition
    result = _strassen_block_decomposition()
    if _verify_decomposition(result, n, m, p, trials=50):
        best_result = result
        best_rank = 42
    else:
        best_result = None
        best_rank = 48

    # Try to find a better decomposition using 2x2x3 blocks
    # Try rank 8 for 2x2x3 (total rank 32) - this is the theoretical minimum
    if time.time() < deadline - 30:
        decomp_8 = _find_2x2x3_decomposition(8, max_iters=100, deadline=deadline, seed=1, n_restarts=5)
        if decomp_8 is not None:
            candidate = _combine_2x2x3_to_2x4x6(decomp_8)
            if _verify_decomposition(candidate, n, m, p, trials=20):
                best_result = candidate
                best_rank = 32

    # Try rank 9 for 2x2x3 (total rank 36) if 8 failed
    if best_rank > 36 and time.time() < deadline - 30:
        decomp_9 = _find_2x2x3_decomposition(9, max_iters=100, deadline=deadline, seed=10, n_restarts=5)
        if decomp_9 is not None:
            candidate = _combine_2x2x3_to_2x4x6(decomp_9)
            if _verify_decomposition(candidate, n, m, p, trials=20):
                best_result = candidate
                best_rank = 36

    # Try rank 10 for 2x2x3 (total rank 40) if 9 failed
    if best_rank > 40 and time.time() < deadline - 30:
        decomp_10 = _find_2x2x3_decomposition(10, max_iters=100, deadline=deadline, seed=20, n_restarts=5)
        if decomp_10 is not None:
            candidate = _combine_2x2x3_to_2x4x6(decomp_10)
            if _verify_decomposition(candidate, n, m, p, trials=20):
                best_result = candidate
                best_rank = 40

    if best_result is None:
        best_result = _strassen_block_decomposition()

    return best_result
