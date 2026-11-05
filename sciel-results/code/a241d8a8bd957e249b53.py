"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via local search with exact rational arithmetic.
     - Exact duplicate-column removal.
     - Single-column linear-dependency elimination (solve residual over remaining columns, round to half-integers, validate).
     - Two-column simultaneous elimination (solve 2x system, round, validate).
  3. Validate any candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the known rank-44 decomposition.
"""
import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # explicit guard: 150s kill, keep safety margin

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact integer validation: sum_c U*V*W == 1 for every (i,a,k) ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    acc = 0
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    # ---- Helper: convert matrix of ints to list of column vectors (as tuples) ----
    def _to_cols(mat):
        if not mat:
            return []
        num_rows = len(mat)
        num_cols = len(mat[0])
        cols = []
        for c in range(num_cols):
            col_vec = tuple(mat[r][c] for r in range(num_rows))
            cols.append(col_vec)
        return cols

    # ---- Remove exact duplicate columns ----
    def _remove_duplicates(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        
        U_cols = _to_cols(U)
        V_cols = _to_cols(V)
        W_cols = _to_cols(W)
        
        seen = set()
        new_U = []
        new_V = []
        new_W = []
        
        for c in range(rank):
            key = (U_cols[c], V_cols[c], W_cols[c])
            if key not in seen:
                seen.add(key)
                new_U.append(U_cols[c])
                new_V.append(V_cols[c])
                new_W.append(W_cols[c])
        
        new_rank = len(new_U)
        if new_rank == rank:
            return candidate
            
        # Reconstruct matrices
        new_U_mat = [[0] * new_rank for _ in range(n * m)]
        new_V_mat = [[0] * new_rank for _ in range(m * p)]
        new_W_mat = [[0] * new_rank for _ in range(p * n)]
        
        for c in range(new_rank):
            for r in range(n * m):
                new_U_mat[r][c] = new_U[c][r]
            for r in range(m * p):
                new_V_mat[r][c] = new_V[c][r]
            for r in range(p * n):
                new_W_mat[r][c] = new_W[c][r]
                
        return {"U": new_U_mat, "V": new_V_mat, "W": new_W_mat}

    # ---- Try to eliminate a single column c by expressing it as a linear combo of others ----
    def _try_eliminate_single(candidate, c_elim):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        
        # We want to find coefficients x_j (j != c_elim) such that for all (i,a,k):
        # sum_{j != c_elim} x_j * U[i,a,j]*V[a,k,j]*W[k,i,j] == U[i,a,c_elim]*V[a,k,c_elim]*W[k,i,c_elim]
        
        # This is a linear system in x_j.
        # Variables: x_0, ..., x_{rank-1} excluding x_{c_elim}. Total rank-1 variables.
        # Constraints: num_terms = 48.
        
        # Build A (48 x (rank-1)) and b (48,)
        var_indices = [j for j in range(rank) if j != c_elim]
        num_vars = len(var_indices)
        
        A = np.zeros((num_terms, num_vars), dtype=float)
        b = np.zeros(num_terms, dtype=float)
        
        idx = 0
        for i in range(n):
            for a in range(m):
                u_elim = U[i*m + a][c_elim]
                for k in range(p):
                    v_elim = V[a*p + k][c_elim]
                    w_elim = W[k*n + i][c_elim]
                    b[idx] = u_elim * v_elim * w_elim
                    
                    for vi, j in enumerate(var_indices):
                        u_j = U[i*m + a][j]
                        v_j = V[a*p + k][j]
                        w_j = W[k*n + i][j]
                        A[idx, vi] = u_j * v_j * w_j
                    idx += 1
        
        # Solve A x = b
        try:
            # Use lstsq for robustness, but we need exact integers/halves.
            # Since the system is likely underdetermined or square, there may be no solution or infinite.
            # We check if b is in the column space of A.
            x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if solution is exact (within floating point error)
            if residuals.size > 0 and residuals[0] > 1e-9:
                return None
                
            # Round x to nearest half-integer
            x_rounded = np.round(x * 2) / 2
            
            # Verify exactness with Fractions
            x_frac = [Fraction(int(v * 2), 2) for v in x_rounded]
            
            # Check if sum matches exactly
            for idx_t in range(num_terms):
                i = idx_t // (m * p)
                rem = idx_t % (m * p)
                a = rem // p
                k = rem % p
                
                acc = Fraction(0)
                for vi, j in enumerate(var_indices):
                    acc += x_frac[vi] * Fraction(U[i*m + a][j]) * Fraction(V[a*p + k][j]) * Fraction(W[k*n + i][j])
                
                target = Fraction(U[i*m + a][c_elim]) * Fraction(V[a*p + k][c_elim]) * Fraction(W[k*n + i][c_elim])
                
                if acc != target:
                    return None
                    
            # Construct new candidate by removing column c_elim
            new_U = []
            new_V = []
            new_W = []
            for r in range(n*m):
                new_U.append([U[r][j] for j in var_indices])
            for r in range(m*p):
                new_V.append([V[r][j] for j in var_indices])
            for r in range(p*n):
                new_W.append([W[r][j] for j in var_indices])
                
            return {"U": new_U, "V": new_V, "W": new_W}
            
        except Exception:
            return None

    # ---- Search for rank reduction ----
    current = base
    current_rank = len(current["U"][0])
    
    # 1. Remove duplicates
    deduped = _remove_duplicates(current)
    if len(deduped["U"][0]) < current_rank:
        current = deduped
        current_rank = len(current["U"][0])
        
    if not _ok():
        return current

    # 2. Try eliminating single columns
    improved = True
    while improved and _ok():
        improved = False
        for c in range(current_rank):
            if not _ok():
                break
            candidate = _try_eliminate_single(current, c)
            if candidate is not None:
                if _validate(candidate):
                    current = candidate
                    current_rank -= 1
                    improved = True
                    break # Restart loop with new rank
                    
    if not _ok():
        return current

    # 3. Try eliminating pairs of columns
    # This is more complex. We solve for x_j, y_j for two columns c1, c2.
    # A x + B y = b
    # Where A, B are matrices formed by the remaining columns.
    # This is a system of 48 equations with 2*(rank-2) variables.
    # We can stack them: [A B] [x; y] = b.
    
    def _try_eliminate_pair(candidate, c1, c2):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        
        var_indices = [j for j in range(rank) if j != c1 and j != c2]
        num_vars = len(var_indices)
        
        # Total variables: 2 * num_vars
        # A_mat is 48 x (2*num_vars)
        A_mat = np.zeros((num_terms, 2 * num_vars), dtype=float)
        b = np.zeros(num_terms, dtype=float)
        
        idx = 0
        for i in range(n):
            for a in range(m):
                u1 = U[i*m + a][c1]
                u2 = U[i*m + a][c2]
                for k in range(p):
                    v1 = V[a*p + k][c1]
                    v2 = V[a*p + k][c2]
                    w1 = W[k*n + i][c1]
                    w2 = W[k*n + i][c2]
                    
                    b[idx] = u1 * v1 * w1 + u2 * v2 * w2
                    
                    for vi, j in enumerate(var_indices):
                        u_j = U[i*m + a][j]
                        v_j = V[a*p + k][j]
                        w_j = W[k*n + i][j]
                        prod = u_j * v_j * w_j
                        A_mat[idx, vi] = prod
                        A_mat[idx, vi + num_vars] = prod
                    idx += 1
                    
        try:
            x, residuals, rank_A, s = np.linalg.lstsq(A_mat, b, rcond=None)
            if residuals.size > 0 and residuals[0] > 1e-9:
                return None
                
            x_rounded = np.round(x * 2) / 2
            x_frac = [Fraction(int(v * 2), 2) for v in x_rounded]
            
            # Verify
            for idx_t in range(num_terms):
                i = idx_t // (m * p)
                rem = idx_t % (m * p)
                a = rem // p
                k = rem % p
                
                acc = Fraction(0)
                for vi, j in enumerate(var_indices):
                    term = Fraction(U[i*m + a][j]) * Fraction(V[a*p + k][j]) * Fraction(W[k*n + i][j])
                    acc += x_frac[vi] * term + x_frac[vi + num_vars] * term
                
                target = (Fraction(U[i*m + a][c1]) * Fraction(V[a*p + k][c1]) * Fraction(W[k*n + i][c1]) +
                          Fraction(U[i*m + a][c2]) * Fraction(V[a*p + k][c2]) * Fraction(W[k*n + i][c2]))
                
                if acc != target:
                    return None
                    
            # Construct new candidate
            new_U = []
            new_V = []
            new_W = []
            for r in range(n*m):
                new_U.append([U[r][j] for j in var_indices])
            for r in range(m*p):
                new_V.append([V[r][j] for j in var_indices])
            for r in range(p*n):
                new_W.append([W[r][j] for j in var_indices])
                
            return {"U": new_U, "V": new_V, "W": new_W}
            
        except Exception:
            return None

    # Limit pair search to avoid combinatorial explosion if rank is high
    # But rank is ~44, pairs are ~900. Each solve is 48x84. Should be fast.
    for c1 in range(current_rank):
        if not _ok():
            break
        for c2 in range(c1 + 1, current_rank):
            if not _ok():
                break
            candidate = _try_eliminate_pair(current, c1, c2)
            if candidate is not None:
                if _validate(candidate):
                    current = candidate
                    current_rank -= 2
                    # Restart pair search with new rank
                    break
        if current_rank < len(base["U"][0]) - 2:
            # If we found a pair, we should restart the single elimination loop too
            # But for simplicity and time, we just break out and return
            # Actually, we should continue searching.
            # Let's just break the pair loop and go back to single elimination?
            # No, let's just return the current best if we found any improvement.
            # The probability of finding multiple independent pairs is low.
            break

    return current

# EVOLVE-BLOCK-END
