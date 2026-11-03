import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication."""
    return [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]

def _verify_decomposition(result, n=2, m=4, p=6, trials=30):
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
    """Construct a rank-42 decomposition using Strassen's algorithm on 2x2 blocks."""
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

def _build_2x2x3_tensor():
    """Build the 2x2x3 multiplication tensor as a list of (a,b,c) with value 1."""
    n, m, p = 2, 2, 3
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _try_rank_r_2x2x3(target_rank, deadline, seed_offset=0, n_restarts=3):
    """
    Try to find a rank-target_rank decomposition for 2x2x3.
    Uses alternating least squares with rounding to half-integers.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 2, 2, 3
    indices = _build_2x2x3_tensor()
    
    # Build constraint structure
    # For each (a,b,c) in indices, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # For all other (a,b,c), we need sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    all_a = range(n * m)  # 0..3
    all_b = range(m * p)  # 0..5
    all_c = range(p * n)  # 0..5
    
    target_set = set(indices)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(1000 + seed_offset + restart)
        
        # Initialize
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Alternating least squares
        max_iters = 50
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for W: for each c, find W[c,:] that best fits
            # sum_a,b U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # This is: for each c, we have equations for all (a,b)
            # M_W[c] is a system: for each (a,b), sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
            
            # Build M_W: shape (n*m*m*p, target_rank)
            M_W = np.zeros((n * m * m * p, target_rank))
            idx = 0
            for a in all_a:
                for b in all_b:
                    for r in range(target_rank):
                        M_W[idx, r] = U[a, r] * V[b, r]
                    idx += 1
            
            # RHS for each c: T[:, :, c].flatten()
            for c in all_c:
                rhs = np.zeros(n * m * m * p)
                for a in all_a:
                    for b in all_b:
                        if (a, b, c) in target_set:
                            rhs[a * m * p + b] = 1.0
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    pass
            
            # Solve for U: for each a, find U[a,:] that best fits
            # sum_b,c V[b,r]*W[c,r]*U[a,r] = T[a,b,c]
            M_U = np.zeros((m * p * p * n, target_rank))
            idx = 0
            for b in all_b:
                for c in all_c:
                    for r in range(target_rank):
                        M_U[idx, r] = V[b, r] * W[c, r]
                    idx += 1
            
            for a in all_a:
                rhs = np.zeros(m * p * p * n)
                for b in all_b:
                    for c in all_c:
                        if (a, b, c) in target_set:
                            rhs[b * p * n + c] = 1.0
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    pass
            
            # Solve for V: for each b, find V[b,:] that best fits
            # sum_a,c U[a,r]*W[c,r]*V[b,r] = T[a,b,c]
            M_V = np.zeros((n * m * p * n, target_rank))
            idx = 0
            for a in all_a:
                for c in all_c:
                    for r in range(target_rank):
                        M_V[idx, r] = U[a, r] * W[c, r]
                    idx += 1
            
            for b in all_b:
                rhs = np.zeros(n * m * p * n)
                for a in all_a:
                    for c in all_c:
                        if (a, b, c) in target_set:
                            rhs[a * p * n + c] = 1.0
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    pass
        
        # Round to nearest half-integer
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        # Verify exactly
        error = 0.0
        for a in all_a:
            for b in all_b:
                for c in all_c:
                    val = float(np.sum(U_round[a, :] * V_round[b, :] * W_round[c, :]))
                    expected = 1.0 if (a, b, c) in target_set else 0.0
                    error += abs(val - expected)
        
        if error < 1e-6:
            return U_round, V_round, W_round
    
    return None

def _combine_2x2x3_to_2x4x6(decomp_2x2x3):
    """Combine four 2x2x3 decompositions to form a 2x4x6 decomposition."""
    n, m, p = 2, 4, 6
    U_223, V_223, W_223 = decomp_2x2x3
    rank_223 = U_223.shape[1]
    total_rank = 4 * rank_223
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    blocks = [
        (0, 0, 0),   # A_L (cols 0-1), B_TL (rows 0-1, cols 0-2) -> C_L (cols 0-2)
        (0, 0, 3),   # A_L (cols 0-1), B_TR (rows 0-1, cols 3-5) -> C_R (cols 3-5)
        (2, 2, 0),   # A_R (cols 2-3), B_BL (rows 2-3, cols 0-2) -> C_L (cols 0-2)
        (2, 2, 3),   # A_R (cols 2-3), B_BR (rows 2-3, cols 3-5) -> C_R (cols 3-5)
    ]
    
    for b_idx, (a_off, b_row_off, c_col_off) in enumerate(blocks):
        base_rank = b_idx * rank_223
        
        for i in range(2):
            for j in range(2):
                u_idx_global = i * m + (j + a_off)
                u_idx_local = i * 2 + j
                for r in range(rank_223):
                    U[u_idx_global][base_rank + r] = U_223[u_idx_local, r]
        
        for j in range(2):
            for k in range(3):
                v_idx_global = (j + b_row_off) * p + k
                v_idx_local = j * 3 + k
                for r in range(rank_223):
                    V[v_idx_global][base_rank + r] = V_223[v_idx_local, r]
        
        for i in range(2):
            for k in range(3):
                w_idx_global = (k + c_col_off) * n + i
                w_idx_local = k * 2 + i
                for r in range(rank_223):
                    W[w_idx_global][base_rank + r] = W_223[w_idx_local, r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, get the reliable rank-42 Strassen decomposition
    best_result = _strassen_block_decomposition()
    best_rank = 42
    
    if not _verify_decomposition(best_result, n, m, p, trials=20):
        # Fallback to schoolbook rank-48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 48
    
    # Try to find rank-9 2x2x3 decomposition (total rank 36)
    if time.time() < deadline - 10:
        decomp_9 = _try_rank_r_2x2x3(9, deadline, seed_offset=10, n_restarts=3)
        if decomp_9 is not None:
            candidate = _combine_2x2x3_to_2x4x6(decomp_9)
            if _verify_decomposition(candidate, n, m, p, trials=20):
                best_result = candidate
                best_rank = 36
    
    # Try rank-8 2x2x3 (total rank 32) if time remains
    if best_rank > 32 and time.time() < deadline - 10:
        decomp_8 = _try_rank_r_2x2x3(8, deadline, seed_offset=20, n_restarts=2)
        if decomp_8 is not None:
            candidate = _combine_2x2x3_to_2x4x6(decomp_8)
            if _verify_decomposition(candidate, n, m, p, trials=20):
                best_result = candidate
                best_rank = 32
    
    return best_result
