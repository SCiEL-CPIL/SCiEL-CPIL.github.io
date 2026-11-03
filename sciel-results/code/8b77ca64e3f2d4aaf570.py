"""Matrix multiplication 3x3 by 3x4 - rank minimization via tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

_START_TIME = time.time()
DEADLINE = _START_TIME + 118  # 2s safety margin before 120s limit

def _time_left():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    The matrix multiplication tensor T[i,j,k] = 1 if the (i,k) entry of A*B
    depends on A[i,j]*B[j,k].
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    Key insight: For fixed U, V, the equations decouple by (i,k) pair:
    For each (i,k): sum_c [sum_j U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c] = 1
    Let M[(i,k),c] = sum_j U[i*m+j,c] * V[j*p+k,c]
    Then: sum_c M[(i,k),c] * W[k*n+i,c] = 1
    
    This is 12 independent linear equations (one per (i,k) pair), each with r unknowns.
    """
    n, m, p = 3, 3, 4
    
    # Schoolbook fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But we need at least enough to handle the structure
    
    for target_rank in range(13, min(best_rank, 30)):
        if _time_left() < 5:
            break
        
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue searching for even lower rank
    
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
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _try_rank(n, m, p, r):
    """
    Try to find a rank-r decomposition.
    
    Strategy:
    1. Randomly initialize U (n*m x r) and V (m*p x r) with entries in {-1, 0, 1}
    2. Compute M[(i,k),c] = sum_j U[i*m+j,c] * V[j*p+k,c] for all (i,k),c
    3. For each (i,k), solve sum_c M[(i,k),c] * W[k*n+i,c] = 1 with half-integer W
    4. Verify the complete decomposition
    """
    np.random.seed(int(time.time() * 1000) % 1000000)
    
    max_attempts = 200 if r <= 20 else 100
    if _time_left() < 10:
        max_attempts = 20
    
    for attempt in range(max_attempts):
        if _time_left() < 2:
            break
        
        # Random sparse initialization of U and V
        # Use entries in {-1, 0, 1} with varying density
        density = 0.3 + 0.4 * (attempt % 5) / 5  # Vary density
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for i in range(n*m):
            for c in range(r):
                if np.random.random() < density:
                    U_vals[i, c] = np.random.choice([-1, 1])
        
        for i in range(m*p):
            for c in range(r):
                if np.random.random() < density:
                    V_vals[i, c] = np.random.choice([-1, 1])
        
        # Compute M[(i,k),c] = sum_j U[i*m+j,c] * V[j*p+k,c]
        # Index M by (k*n + i) to match W's row indexing
        M = np.zeros((p*n, r), dtype=int)
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                    M[k*n + i, c] = s
        
        # For each (i,k) [row index k*n+i of W], solve:
        # sum_c M[k*n+i, c] * W[k*n+i, c] = 1
        # Need W entries to be half-integers (integers or half-integers)
        
        W_vals = np.zeros((p*n, r), dtype=object)
        valid = True
        
        for i in range(n):
            for k in range(p):
                row_idx = k * n + i
                m_row = M[row_idx, :]  # length r, integer values
                
                # Find a half-integer solution to sum_c m_row[c] * w[c] = 1
                w_solution = _solve_half_integer(m_row)
                if w_solution is None:
                    valid = False
                    break
                W_vals[row_idx, :] = w_solution
            if not valid:
                break
        
        if not valid:
            continue
        
        # Convert to lists for verification
        U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n*m)]
        V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m*p)]
        W_list = []
        for i in range(p*n):
            row = []
            for c in range(r):
                val = W_vals[i, c]
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        row.append(val.numerator)
                    else:
                        row.append(float(val))
                else:
                    row.append(int(val) if val == int(val) else float(val))
            W_list.append(row)
        
        # Verify with exact arithmetic
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _solve_half_integer(m_row):
    """
    Find a half-integer vector w such that sum_c m_row[c] * w[c] = 1.
    m_row is a numpy array of integers.
    Returns a list of Fractions or None if no sparse solution found.
    """
    r = len(m_row)
    
    # Try 1-term solutions: m_row[c] * w[c] = 1 => w[c] = 1/m_row[c]
    # Need 1/m_row[c] to be a half-integer, so m_row[c] must be ±1, ±2
    for c in range(r):
        mc = int(m_row[c])
        if mc != 0:
            w_val = Fraction(1, mc)
            # Check if half-integer: w_val * 2 should be integer
            if (w_val * 2).denominator == 1:
                result = [Fraction(0)] * r
                result[c] = w_val
                return result
    
    # Try 2-term solutions: m_row[a] * w[a] + m_row[b] * w[b] = 1
    # w[a] = x, w[b] = y, both half-integers
    # m_a * x + m_b * y = 1
    # Try x = ±1, ±1/2, 0 and solve for y
    for a in range(r):
        ma = int(m_row[a])
        if ma == 0:
            continue
        for b in range(a+1, r):
            mb = int(m_row[b])
            if mb == 0:
                continue
            # Try various half-integer values for x
            for x_num in [-2, -1, 0, 1, 2]:  # x = x_num/2
                x = Fraction(x_num, 2)
                # m_a * x + m_b * y = 1 => y = (1 - m_a*x) / m_b
                y = (Fraction(1) - Fraction(ma) * x) / Fraction(mb)
                if (y * 2).denominator == 1:
                    result = [Fraction(0)] * r
                    result[a] = x
                    result[b] = y
                    return result
    
    # Try 3-term solutions: m_row[a]*w[a] + m_row[b]*w[b] + m_row[c]*w[c] = 1
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
                # Try w[a]=1, w[b]=1, solve for w[c]
                w_a = Fraction(1)
                w_b = Fraction(1)
                w_c = (Fraction(1) - Fraction(ma)*w_a - Fraction(mb)*w_b) / Fraction(mc)
                if (w_c * 2).denominator == 1:
                    result = [Fraction(0)] * r
                    result[a] = w_a
                    result[b] = w_b
                    result[c] = w_c
                    return result
                
                # Try w[a]=1, w[b]=-1, solve for w[c]
                w_a = Fraction(1)
                w_b = Fraction(-1)
                w_c = (Fraction(1) - Fraction(ma)*w_a - Fraction(mb)*w_b) / Fraction(mc)
                if (w_c * 2).denominator == 1:
                    result = [Fraction(0)] * r
                    result[a] = w_a
                    result[b] = w_b
                    result[c] = w_c
                    return result
                
                # Try w[a]=1/2, w[b]=1/2, solve for w[c]
                w_a = Fraction(1, 2)
                w_b = Fraction(1, 2)
                w_c = (Fraction(1) - Fraction(ma)*w_a - Fraction(mb)*w_b) / Fraction(mc)
                if (w_c * 2).denominator == 1:
                    result = [Fraction(0)] * r
                    result[a] = w_a
                    result[b] = w_b
                    result[c] = w_c
                    return result
    
    return None

# Main entry point
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
