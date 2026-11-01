"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    max_rank = 60
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= max_rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Correct Strassen block for 2x2x2 sub-problem
    # Sub-problem: A[2x2] * B[2x2] -> C[2x2]
    # A indices: (i, ja) and (i, jb) for i in {0,1}
    # B indices: (ja, ka), (ja, kb), (jb, ka), (jb, kb)
    # C indices: (ka, i), (kb, i) for i in {0,1}
    #
    # u_idx(i, j) = i*m + j
    # v_idx(j, k) = j*p + k
    # w_idx(k, i) = k*n + i
    def strassen_2x2x2(ja, jb, ka, kb):
        # U indices
        u_0ja = 0 * m + ja  # row 0, col ja
        u_1ja = 1 * m + ja  # row 1, col ja
        u_0jb = 0 * m + jb  # row 0, col jb
        u_1jb = 1 * m + jb  # row 1, col jb

        # V indices
        v_jaka = ja * p + ka  # row ja, col ka
        v_jakb = ja * p + kb  # row ja, col kb
        v_jbka = jb * p + ka  # row jb, col ka
        v_jbkb = jb * p + kb  # row jb, col kb

        # W indices
        w_ka0 = ka * n + 0  # row ka, col 0
        w_ka1 = ka * n + 1  # row ka, col 1
        w_kb0 = kb * n + 0  # row kb, col 0
        w_kb1 = kb * n + 1  # row kb, col 1

        # Strassen's 7 products for 2x2*2x2:
        # M1 = (A00+A11)*(B00+B11)
        add_term([(u_0ja, 1), (u_1jb, 1)], [(v_jaka, 1), (v_jbkb, 1)], [(w_ka0, 1), (w_ka1, 1), (w_kb0, 1), (w_kb1, 1)])

        # M2 = (A10+A11)*B00
        add_term([(u_1ja, 1), (u_1jb, 1)], [(v_jaka, 1)], [(w_ka0, 1), (w_ka1, 1)])

        # M3 = A00*(B10-B00)
        add_term([(u_0ja, 1)], [(v_jbka, 1), (v_jaka, -1)], [(w_ka0, -1), (w_ka1, -1), (w_kb0, 1), (w_kb1, 1)])

        # M4 = A11*(B01-B10)
        add_term([(u_1jb, 1)], [(v_jakb, 1), (v_jbka, -1)], [(w_ka0, 1), (w_ka1, 1), (w_kb0, -1), (w_kb1, -1)])

        # M5 = (A00+A01)*B11
        add_term([(u_0ja, 1), (u_0jb, 1)], [(v_jbkb, 1)], [(w_kb0, 1), (w_kb1, 1)])

        # M6 = (A10-A00)*(B00+B01)
        add_term([(u_1ja, 1), (u_0ja, -1)], [(v_jaka, 1), (v_jakb, 1)], [(w_ka0, 1), (w_ka1, 1)])

        # M7 = (A01-A11)*(B10+B11)
        add_term([(u_0jb, 1), (u_1jb, -1)], [(v_jbka, 1), (v_jbkb, 1)], [(w_kb0, 1), (w_kb1, 1)])

    # Try different block decompositions and find the best one
    best_result = None
    best_rank = max_rank

    # Strategy 1: Two Strassen 2x2x2 blocks + schoolbook for rest
    # Block 1: j={0,1}, k={0,1}
    # Block 2: j={2,3}, k={2,3}
    # Remaining: j={0,1,2,3} x k={2,3,4,5} + j={4} x k={0,1,2,3,4,5}
    
    def try_decomposition(strassen_blocks, label=""):
        nonlocal col
        col = 0
        U.fill(0)
        V.fill(0)
        W.fill(0)
        
        # Apply Strassen blocks
        for (ja, jb, ka, kb) in strassen_blocks:
            strassen_2x2x2(ja, jb, ka, kb)
        
        # Track which (i,j,k) are covered
        covered = set()
        for (ja, jb, ka, kb) in strassen_blocks:
            for i in range(n):
                for j in [ja, jb]:
                    for k in [ka, kb]:
                        covered.add((i, j, k))
        
        # Add schoolbook terms for uncovered entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, col] = 1
                        V[v_idx, col] = 1
                        W[w_idx, col] = 1
                        col += 1
        
        # Verify
        U_used = U[:, :col]
        V_used = V[:, :col]
        W_used = W[:, :col]
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_used[i * m + j, :]
                    v_row = V_used[j * p + k, :]
                    w_row = W_used[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and col < best_rank:
            best_rank = col
            best_result = {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
        
        return col if valid else max_rank

    # Try various block configurations
    configs = [
        # Two Strassen blocks on j={0,1}x k={0,1} and j={2,3}x k={2,3}
        [(0, 1, 0, 1), (2, 3, 2, 3)],
        # Two Strassen blocks on j={0,1}x k={0,1} and j={2,3}x k={0,1}
        [(0, 1, 0, 1), (2, 3, 0, 1)],
        # Two Strassen blocks on j={0,1}x k={4,5} and j={2,3}x k={4,5}
        [(0, 1, 4, 5), (2, 3, 4, 5)],
        # One Strassen block on j={0,1}x k={0,1}
        [(0, 1, 0, 1)],
        # Three Strassen blocks
        [(0, 1, 0, 1), (0, 1, 2, 3), (2, 3, 0, 1)],
        [(0, 1, 0, 1), (2, 3, 0, 1), (2, 3, 2, 3)],
        [(0, 1, 0, 1), (0, 1, 4, 5), (2, 3, 0, 1)],
        # Four Strassen blocks
        [(0, 1, 0, 1), (0, 1, 2, 3), (2, 3, 0, 1), (2, 3, 2, 3)],
        [(0, 1, 0, 1), (0, 1, 4, 5), (2, 3, 0, 1), (2, 3, 4, 5)],
    ]
    
    for config in configs:
        if time.time() > deadline - 10:
            break
        try_decomposition(config)
    
    # If no valid decomposition found, fallback to schoolbook
    if best_result is None:
        U_fallback = np.zeros((n * m, n * m * p), dtype=float)
        V_fallback = np.zeros((m * p, n * m * p), dtype=float)
        W_fallback = np.zeros((p * n, n * m * p), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
