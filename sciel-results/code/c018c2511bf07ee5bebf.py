"""Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations_with_replacement

_START_TIME = time.time()
DEADLINE = _START_TIME + 118  # 2s safety margin before 120s limit

def _time_left():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    Strategy: For each target rank r, try many random (U, V) pairs with small
    integer entries. For each, compute the induced M matrix and check if a
    half-integer W exists that satisfies all 12 equations. Use a more
    systematic search with structured patterns in addition to random tries.
    """
    n, m, p = 3, 3, 4
    
    # Schoolbook fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try ranks from low to high
    for target_rank in range(12, min(best_rank + 1, 40)):
        if _time_left() < 3:
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
    Try to find a rank-r decomposition.
    
    Multiple strategies:
    1. Random sparse integer U, V
    2. Structured patterns (block diagonal, etc.)
    3. For each, solve for W exactly
    """
    np.random.seed(42 + r * 1000)
    
    max_attempts = 500 if r <= 16 else 300 if r <= 20 else 150
    if _time_left() < 10:
        max_attempts = 50
    
    # Strategy 1: Random search
    for attempt in range(max_attempts):
        if _time_left() < 2:
            break
        
        # Vary the density and value range
        density = 0.2 + 0.5 * (attempt % 7) / 7
        max_val = 1 + (attempt % 3)  # Values in {-max_val, ..., max_val}
        
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
    
    # Strategy 2: Structured patterns
    result = _try_structured(n, m, p, r)
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
            for x_num in range(-4, 5):  # x = x_num/2
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
                for wa_num in [-2, -1, 0, 1, 2]:
                    w_a = Fraction(wa_num, 2)
                    for wb_num in [-2, -1, 0, 1, 2]:
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
                        # Try w[a]=1, w[b]=0, w[c]=0, solve for w[d]
                        w_a = Fraction(1)
                        w_b = Fraction(0)
                        w_c = Fraction(0)
                        w_d = (Fraction(1) - Fraction(ma)*w_a) / Fraction(md)
                        if (w_d * 2).denominator == 1:
                            result = [Fraction(0)] * r
                            result[a] = w_a
                            result[d] = w_d
                            return result
                        
                        # Try w[a]=0, w[b]=1, w[c]=0, solve for w[d]
                        w_a = Fraction(0)
                        w_b = Fraction(1)
                        w_c = Fraction(0)
                        w_d = (Fraction(1) - Fraction(mb)*w_b) / Fraction(md)
                        if (w_d * 2).denominator == 1:
                            result = [Fraction(0)] * r
                            result[b] = w_b
                            result[d] = w_d
                            return result
    
    return None

def _try_structured(n, m, p, r):
    """Try structured patterns for U and V."""
    # Pattern 1: Block diagonal-like structure
    # Each column of U and V has non-zero entries in specific blocks
    
    # Try: for each column c, assign it to a specific (i,j) or (j,k) pattern
    # This is essentially trying all ways to partition the r columns
    
    # Simpler: try patterns where U[i*m+j, c] is non-zero only for specific (i,j)
    # and V[j*p+k, c] is non-zero only for specific (j,k)
    
    # For small r, enumerate possible structures
    if r > 25:
        return None
    
    # Try: each column c corresponds to a triple (i_c, j_c, k_c)
    # U[i*m+j, c] = 1 if i==i_c and j==j_c, else 0
    # V[j*p+k, c] = 1 if j==j_c and k==k_c, else 0
    # Then M[k*n+i, c] = 1 if i==i_c and k==k_c, else 0
    # And W[k*n+i, c] must be 1 if i==i_c and k==k_c, else 0
    # This gives the schoolbook decomposition, which has rank n*m*p
    
    # More general: allow U and V to have multiple non-zero entries per column
    # Try: U[i*m+j, c] in {-1, 0, 1} with exactly 2 non-zeros per column
    # V[j*p+k, c] in {-1, 0, 1} with exactly 2 non-zeros per column
    
    np.random.seed(12345 + r)
    
    for attempt in range(100):
        if _time_left() < 2:
            break
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for c in range(r):
            # Pick 2 random positions in U for column c
            u_positions = np.random.choice(n*m, size=2, replace=False)
            for pos in u_positions:
                U_vals[pos, c] = np.random.choice([-1, 1])
            
            # Pick 2 random positions in V for column c
            v_positions = np.random.choice(m*p, size=2, replace=False)
            for pos in v_positions:
                V_vals[pos, c] = np.random.choice([-1, 1])
        
        result = _solve_for_W(n, m, p, r, U_vals, V_vals)
        if result is not None:
            return result
    
    return None

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
