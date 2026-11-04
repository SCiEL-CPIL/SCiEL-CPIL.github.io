"""
Strategy: 
1. Base: 44-term decomposition (4 Strassen blocks for 2x2x8 + schoolbook for 2x1x8).
2. Laser Method: Construct a systematic linear constraint system using the specific
   2x3x8 geometry. For each candidate term, build the full linear system of equations
   (one per (i,a,k) entry) and solve via exact Gaussian elimination. Use multiple
   passes with different term orderings to find all possible eliminations.
3. Final polish: After each elimination, re-run the search to catch cascading dependencies.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Verify 44-term
    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    best_rank = 44
    best_candidate = candidate_44

    # Laser Method: Build the full linear constraint system for rank reduction.
    # For each term t, we want to find if there exist coefficients c_s (s != t)
    # such that for ALL (i,a,k): T_t[i,a,k] = sum_s c_s * T_s[i,a,k]
    # where T_t[i,a,k] = U[i,a,t]*V[a,k,t]*W[k,i,t]
    
    # Precompute all (i,a,k) indices
    all_ijk = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]
    
    def _gaussian_eliminate(mat, num_vars):
        """Solve linear system using exact Gaussian elimination.
        mat is a list of rows, each row has num_vars coefficients + 1 rhs.
        Returns (coeffs, is_consistent, num_pivots)"""
        num_eqs = len(mat)
        if num_eqs == 0:
            return [Fraction(0)] * num_vars, True, 0
        
        # Work on a copy
        A = [row[:] for row in mat]
        
        pivot_cols = []
        for col in range(num_vars):
            # Find pivot in this column
            pivot_row = -1
            for row_idx in range(len(pivot_cols), num_eqs):
                if A[row_idx][col] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            
            # Swap
            if pivot_row != len(pivot_cols):
                A[len(pivot_cols)], A[pivot_row] = A[pivot_row], A[len(pivot_cols)]
            
            pivot_cols.append(col)
            pv = A[len(pivot_cols) - 1][col]
            
            # Eliminate below
            for row_idx in range(len(pivot_cols), num_eqs):
                if A[row_idx][col] != 0:
                    factor = A[row_idx][col] / pv
                    for j in range(col, num_vars + 1):
                        A[row_idx][j] -= factor * A[len(pivot_cols) - 1][j]
        
        # Check consistency
        for row_idx in range(len(pivot_cols), num_eqs):
            all_zero = all(A[row_idx][j] == 0 for j in range(num_vars))
            if all_zero and A[row_idx][num_vars] != 0:
                return None, False, len(pivot_cols)
        
        # Back substitution
        coeffs = [Fraction(0)] * num_vars
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            row = A[i]
            rhs = row[num_vars]
            for j in range(col + 1, num_vars):
                if row[j] != 0:
                    rhs -= row[j] * coeffs[j]
            if row[col] != 0:
                coeffs[col] = rhs / row[col]
            else:
                # Should not happen if pivot was found
                if rhs != 0:
                    return None, False, len(pivot_cols)
        
        return coeffs, True, len(pivot_cols)
    
    def _try_eliminate(U_mat, V_mat, W_mat, eliminated_set, term_idx):
        """Try to eliminate term_idx using all other non-eliminated terms."""
        if not _elapsed_ok():
            return False
        
        active = [t for t in range(rank_44) if t not in eliminated_set and t != term_idx]
        if not active:
            return False
        
        num_vars = len(active)
        
        # Build the full linear system: one equation per (i,a,k)
        # Equation: sum_s c_s * U[i,a,s]*V[a,k,s]*W[k,i,s] = U[i,a,term]*V[a,k,term]*W[k,i,term]
        rows = []
        for (i, a, k) in all_ijk:
            rhs = Fraction(U_mat[i * m + a][term_idx]) * \
                   Fraction(V_mat[a * p + k][term_idx]) * \
                   Fraction(W_mat[k * n + i][term_idx])
            
            row = []
            for s in active:
                coeff = Fraction(U_mat[i * m + a][s]) * \
                         Fraction(V_mat[a * p + k][s]) * \
                         Fraction(W_mat[k * n + i][s])
                row.append(coeff)
            row.append(rhs)
            rows.append(row)
        
        if not rows:
            return False
        
        # Solve
        coeffs, consistent, num_pivots = _gaussian_eliminate(rows, num_vars)
        if not consistent:
            return False
        
        if coeffs is None:
            return False
        
        # Verify solution against ALL entries
        for (i, a, k) in all_ijk:
            lhs = Fraction(U_mat[i * m + a][term_idx]) * \
                   Fraction(V_mat[a * p + k][term_idx]) * \
                   Fraction(W_mat[k * n + i][term_idx])
            rhs = Fraction(0)
            for idx, s in enumerate(active):
                if coeffs[idx] != 0:
                    rhs += coeffs[idx] * Fraction(U_mat[i * m + a][s]) * \
                           Fraction(V_mat[a * p + k][s]) * \
                           Fraction(W_mat[k * n + i][s])
            if lhs != rhs:
                return False
        
        # Check half-integer constraint
        for c in coeffs:
            if c.denominator > 2:
                return False
        
        return True
    
    # Multiple passes with different orderings (Laser Method: try all orderings)
    eliminated_set = set()
    U_cur = [row[:] for row in U_44]
    V_cur = [row[:] for row in V_44]
    W_cur = [row[:] for row in W_44]
    
    # Pass 1: Simplest terms first
    term_order_1 = sorted(range(rank_44), key=lambda r: 
        sum(1 for row in U_cur for x in row if x != 0) +
        sum(1 for row in V_cur for x in row if x != 0) +
        sum(1 for row in W_cur for x in row if x != 0))
    
    for r in term_order_1:
        if not _elapsed_ok():
            break
        if r in eliminated_set:
            continue
        if _try_eliminate(U_cur, V_cur, W_cur, eliminated_set, r):
            eliminated_set.add(r)
            best_rank = rank_44 - len(eliminated_set)
    
    # Pass 2: Reverse order
    term_order_2 = list(reversed(term_order_1))
    for r in term_order_2:
        if not _elapsed_ok():
            break
        if r in eliminated_set:
            continue
        if _try_eliminate(U_cur, V_cur, W_cur, eliminated_set, r):
            eliminated_set.add(r)
            best_rank = rank_44 - len(eliminated_set)
    
    # Pass 3: Random-ish order (deterministic pseudo-random)
    term_order_3 = sorted(range(rank_44), key=lambda r: (r * 7) % rank_44)
    for r in term_order_3:
        if not _elapsed_ok():
            break
        if r in eliminated_set:
            continue
        if _try_eliminate(U_cur, V_cur, W_cur, eliminated_set, r):
            eliminated_set.add(r)
            best_rank = rank_44 - len(eliminated_set)
    
    # Final polish: try again in original order
    for r in range(rank_44):
        if not _elapsed_ok():
            break
        if r in eliminated_set:
            continue
        if _try_eliminate(U_cur, V_cur, W_cur, eliminated_set, r):
            eliminated_set.add(r)
            best_rank = rank_44 - len(eliminated_set)
    
    # Rebuild final candidate
    if eliminated_set:
        active_terms = [t for t in range(rank_44) if t not in eliminated_set]
        U_new = [[U_cur[i * m + a][t] for t in active_terms] for i in range(n) for a in range(m)]
        V_new = [[V_cur[a * p + k][t] for t in active_terms] for a in range(m) for k in range(p)]
        W_new = [[W_cur[k * n + i][t] for t in active_terms] for k in range(p) for i in range(n)]
        
        candidate = {"U": U_new, "V": V_new, "W": W_new}
        if _validate(candidate):
            best_candidate = candidate
        else:
            # Fallback to 44-term if validation fails
            best_candidate = candidate_44
    else:
        best_candidate = candidate_44
    
    return best_candidate

# EVOLVE-BLOCK-END
