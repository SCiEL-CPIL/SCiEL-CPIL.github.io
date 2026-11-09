"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def _remaining_time():
    return _TIMEOUT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    T_approx = np.zeros_like(T)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for r in range(rank):
        T_approx += U_np[:, r][:, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
                    
    return np.allclose(T_approx, T)

def _construct_strassen_238():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: j = 2 (2x1x2)
        a2_idx = 0 * m + 2
        c2_idx = 1 * m + 2
        
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """
    Constructs rank 48 schoolbook decomposition.
    """
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

def _try_reduce_rank(base_U, base_V, base_W, n, m, p, target_rank, max_iters=5000):
    """
    Try to reduce the rank by merging terms using local search.
    """
    if target_rank >= len(base_U[0]):
        return None
    
    # Start from the base solution
    U = [row[:] for row in base_U]
    V = [row[:] for row in base_V]
    W = [row[:] for row in base_W]
    
    current_rank = len(U[0])
    
    # We want to reduce to target_rank by setting some columns to zero
    # and redistributing their contributions
    
    # For each pair of columns, try to merge them
    for _ in range(max_iters):
        if _remaining_time() < 30:
            break
        
        if current_rank <= target_rank:
            # Verify and return
            if _verify_decomposition(U, V, W, n, m, p):
                # Trim to target_rank
                U_trimmed = [row[:target_rank] for row in U]
                V_trimmed = [row[:target_rank] for row in V]
                W_trimmed = [row[:target_rank] for row in W]
                return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
            break
        
        # Pick two random columns to try merging
        c1, c2 = random.sample(range(current_rank), 2)
        
        # Try to express columns c1 and c2 as a single column
        # This is only possible if they are proportional, which is rare
        # Instead, try a local perturbation
        
        # Perturb one of the columns
        col = random.choice([c1, c2])
        dim = len(U)
        for _ in range(3):
            # Small random perturbation
            if random.random() < 0.3:
                U[random.randint(0, dim-1)][col] += random.choice([-1, 1])
            if random.random() < 0.3:
                V[random.randint(0, len(V)-1)][col] += random.choice([-1, 1])
            if random.random() < 0.3:
                W[random.randint(0, len(W)-1)][col] += random.choice([-1, 1])
        
        # Check if we can now set one column to zero
        # This is a heuristic; in practice this is very unlikely to work
        # So this approach is not very effective
        
        pass
    
    return None

def _greedy_rank_search(n, m, p, start_rank=44, min_rank=30):
    """
    Greedy search for lower rank by trying to remove terms.
    """
    base = _construct_strassen_238()
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    
    current_rank = len(U[0])
    best = {"U": U, "V": V, "W": W}
    
    for target_rank in range(current_rank - 1, min_rank - 1, -1):
        if _remaining_time() < 20:
            break
        
        # Try to find a valid decomposition of rank target_rank
        # by randomly zeroing out columns and checking
        found = False
        for attempt in range(100):
            if _remaining_time() < 10:
                break
            
            # Randomly select target_rank columns to keep
            cols = random.sample(range(current_rank), target_rank)
            cols.sort()
            
            U_test = [row[c] for row in U for c in cols]
            V_test = [row[c] for row in V for c in cols]
            W_test = [row[c] for row in W for c in cols]
            
            # Reshape
            U_test = [U_test[i*target_rank:(i+1)*target_rank] for i in range(len(U))]
            V_test = [V_test[i*target_rank:(i+1)*target_rank] for i in range(len(V))]
            W_test = [W_test[i*target_rank:(i+1)*target_rank] for i in range(len(W))]
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                best = {"U": U_test, "V": V_test, "W": W_test}
                current_rank = target_rank
                found = True
                break
        
        if not found:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: Rank 44 Strassen-based
    cand1 = _construct_strassen_238()
    if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], n, m, p):
        best = cand1
        best_rank = len(cand1["U"][0])
    else:
        best = _construct_schoolbook()
        best_rank = len(best["U"][0])
    
    # Try to improve with greedy search
    if _remaining_time() > 30:
        improved = _greedy_rank_search(n, m, p, start_rank=best_rank, min_rank=30)
        if improved and len(improved["U"][0]) < best_rank:
            best = improved
            best_rank = len(best["U"][0])
    
    return best
