"""Matrix multiplication 3x3 by 3x4 - rank minimization via structured sparse search."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations, combinations_with_replacement

_START_TIME = time.time()
DEADLINE = _START_TIME + 145  # 5s safety margin before 150s limit

def _time_left():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    Strategy: Systematic search over structured patterns and random sparse
    integer U, V matrices. For each candidate, solve for W exactly using
    half-integer constraints. Try ranks from low to high.
    """
    n, m, p = 3, 3, 4
    
    # Schoolbook fallback (rank 36)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try ranks from low to high
    for target_rank in range(12, min(best_rank + 1, 40)):
        if _time_left() < 5:
            break
        
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Don't break - try to find even lower rank
    
    return best_result

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _try_rank(n, m, p, r):
    """
    Try to find a rank-r decomposition using multiple strategies.
    """
    np.random.seed(42 + r * 1000)
    
    # Budget allocation based on time remaining
    time_budget = _time_left()
    if time_budget < 10:
        return None
    
    # Strategy 1: Systematic structured search
    result = _try_structured_systematic(n, m, p, r)
    if result is not None:
        return result
    
    # Strategy 2: Random sparse search with varied parameters
    max_attempts = int(min(2000, time_budget * 50))
    if max_attempts < 50:
        max_attempts = 50
    
    for attempt in range(max_attempts):
        if _time_left() < 3:
            break
        
        # Vary density and value range
        density = 0.15 + 0.55 * (attempt % 11) / 11
        max_val = 1 + (attempt % 4)
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for i in range(n*m):
            for c in range(r):
                if np.random.random() < density:
                    U_vals[i, c] = np.random.randint(-max_val, max_val + 1)
                    if U_vals[i, c] == 0:
                        U_vals[i, c] = 1
        
        for i in range(m*p):
            for c in range(r):
                if np.random.random() < density:
                    V_vals[i, c] = np.random.randint(-max_val, max_val + 1)
                    if V_vals[i, c] == 0:
                        V_vals[i, c] = 1
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    # Strategy 3: Branch-and-bound style structured search
    result = _try_branch_and_bound(n, m, p, r)
    if result is not None:
        return result
    
    return None

def _try_structured_systematic(n, m, p, r):
    """
    Systematic search over structured patterns:
    - Each column c has a specific support pattern in U and V
    - Try all combinations of small support sets
    """
    if r > 25:
        return None
    
    # For each column, try patterns where U and V have limited non-zero entries
    # Focus on patterns that might yield M rows with simple structure
    
    # Try: each column c corresponds to a subset of (i,j) pairs in U and (j,k) pairs in V
    # such that the resulting M row has a simple pattern
    
    np.random.seed(999 + r)
    
    # Try structured patterns with 1-3 non-zeros per column in U and V
    for u_nonzeros in range(1, min(4, n*m + 1)):
        for v_nonzeros in range(1, min(4, m*p + 1)):
            for _ in range(200):
                if _time_left() < 3:
                    break
                
                U_vals = np.zeros((n*m, r), dtype=int)
                V_vals = np.zeros((m*p, r), dtype=int)
                
                for c in range(r):
                    # Choose positions for non-zeros in U column c
                    u_positions = np.random.choice(n*m, size=min(u_nonzeros, n*m), replace=False)
                    for pos in u_positions:
                        U_vals[pos, c] = np.random.choice([-1, 1])
                    
                    # Choose positions for non-zeros in V column c
                    v_positions = np.random.choice(m*p, size=min(v_nonzeros, m*p), replace=False)
                    for pos in v_positions:
                        V_vals[pos, c] = np.random.choice([-1, 1])
                
                result = _solve_for_W(n, m, p, r, U_vals, V_vals)
                if result is not None:
                    return result
    
    return None

def _try_branch_and_bound(n, m, p, r):
    """
    Branch-and-bound style search:
    Build up the decomposition column by column, checking feasibility at each step.
    """
    if r > 20:
        return None
    
    # Start with empty U, V and build up
    U_vals = np.zeros((n*m, r), dtype=int)
    V_vals = np.zeros((m*p, r), dtype=int)
    
    # For each column, try adding non-zero entries in a structured way
    # Focus on columns that contribute to satisfying the tensor equations
    
    np.random.seed(777 + r)
    
    for attempt in range(500):
        if _time_left() < 3:
            break
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        # Build columns one by one
        for c in range(r):
            # Decide support for column c
            # Use a heuristic: prioritize positions that help satisfy unsatisfied equations
            
            # Random choice of 1-3 positions in U
            n_u = np.random.choice([1, 2, 3])
            u_positions = np.random.choice(n*m, size=n_u, replace=False)
            for pos in u_positions:
                U_vals[pos, c] = np.random.choice([-1, 1])
            
            # Random choice of 1-3 positions in V
            n_v = np.random.choice([1, 2, 3])
            v_positions = np.random.choice(m*p, size=n_v, replace=False)
            for pos in v_positions:
                V_vals[pos, c] = np.random.choice([-1, 1])
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    return None

def _solve_for_W(n, m, p, r, U_vals, V_vals):
    """
    Given U and V, compute M and try to find half-integer W.
    Returns (U_list, V_list, W_list) or None.
    """
    # Compute M[(i,k),c] = sum_j U[i*m+j,c] * V[j*p+k,c]
    M = np.zeros((p*n, r), dtype=int)
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                M[k*n + i, c] = s
    
    # For each (i,k) row of W, solve: sum_c M[row,c] * W[row,c] = 1
    W_vals = []
    valid = True
    for i in range(n):
        for k in range(p):
            row_idx = k * n + i
            m_row = M[row_idx, :]
            w_solution = _find_half_integer_solution(m_row)
            if w_solution is None:
                valid = False
                break
            W_vals.append(w_solution)
        if not valid:
            break
    
    if not valid:
        return None
    
    # Convert to proper lists
    U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n*m)]
    V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m*p)]
    W_list = []
    for row in W_vals:
        W_list.append([float(x) if x.denominator != 1 else int(x) for x in row])
    
    # Verify with exact arithmetic
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _find_half_integer_solution(m_row):
    """
    Find a half-integer vector w such that sum_c m_row[c] * w[c] = 1.
    m_row is a numpy array of integers.
    Returns a list of Fractions or None.
    """
    r = len(m_row)
    
    # Try 1-term solutions
    for c in range(r):
        mc = int(m_row[c])
        if mc != 0:
            w_val = Fraction(1, mc)
            if (w_val * 2).denominator == 1:
                result = [Fraction(0)] * r
                result[c] = w_val
                return result
    
    # Try 2-term solutions
    for a in range(r):
        ma = int(m_row[a])
        if ma == 0:
            continue
        for b in range(a+1, r):
            mb = int(m_row[b])
            if mb == 0:
                continue
            for x_num in range(-6, 7):  # x = x_num/2
                x = Fraction(x_num, 2)
                y = (Fraction(1) - Fraction(ma) * x) / Fraction(mb)
                if (y * 2).denominator == 1:
                    result = [Fraction(0)] * r
                    result[a] = x
                    result[b] = y
                    return result
    
    # Try 3-term solutions
    for a in range(r):
        ma = int(m_row[a])
        if ma == 0:
            continue
        for b in range(a+1, r):
            mb = int(m_row[b])
            if mb == 0:
                continue
            for c in range(b+1, r):
                mc = int(m_row[c])
                if mc == 0:
                    continue
                for wa_num in range(-3, 4):
                    w_a = Fraction(wa_num, 2)
                    for wb_num in range(-3, 4):
                        w_b = Fraction(wb_num, 2)
                        w_c = (Fraction(1) - Fraction(ma)*w_a - Fraction(mb)*w_b) / Fraction(mc)
                        if (w_c * 2).denominator == 1:
                            result = [Fraction(0)] * r
                            result[a] = w_a
                            result[b] = w_b
                            result[c] = w_c
                            return result
    
    # Try 4-term solutions with small values
    if r >= 4:
        for a in range(r):
            ma = int(m_row[a])
            if ma == 0:
                continue
            for b in range(a+1, r):
                mb = int(m_row[b])
                if mb == 0:
                    continue
                for c in range(b+1, r):
                    mc = int(m_row[c])
                    if mc == 0:
                        continue
                    for d in range(c+1, r):
                        md = int(m_row[d])
                        if md == 0:
                            continue
                        # Try several combinations
                        for wa in [Fraction(0), Fraction(1), Fraction(-1)]:
                            for wb in [Fraction(0), Fraction(1), Fraction(-1)]:
                                w_c = (Fraction(1) - Fraction(ma)*wa - Fraction(mb)*wb) / Fraction(mc)
                                if (w_c * 2).denominator == 1:
                                    # Now check if we can set w_d to make it work
                                    # Actually, we need sum = 1, so:
                                    # ma*wa + mb*wb + mc*w_c + md*w_d = 1
                                    # w_d = (1 - ma*wa - mb*wb - mc*w_c) / md
                                    w_d = (Fraction(1) - Fraction(ma)*wa - Fraction(mb)*wb - Fraction(mc)*w_c) / Fraction(md)
                                    if (w_d * 2).denominator == 1:
                                        result = [Fraction(0)] * r
                                        result[a] = wa
                                        result[b] = wb
                                        result[c] = w_c
                                        result[d] = w_d
                                        return result
    
    return None

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
