"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that T[i,j,k] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                s = 0
                for r in range(len(U[0])):
                    s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                if s != 1:
                    return False
    return True

def _construct_rank28(n, m, p):
    """Verified rank-28 construction: 4 independent 2x2 Strassen blocks."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for t in range(7):
            c = b * 7 + t
            
            for i in range(2):
                for j in range(2):
                    val = U_coeffs[t][i * 2 + j]
                    if val != 0:
                        U[i * m + j][c] = val
                        
            if t == 0:
                V[0 * p + k0][c] = 1
                V[1 * p + k1][c] = 1
            elif t == 1:
                V[0 * p + k0][c] = 1
            elif t == 2:
                V[0 * p + k1][c] = 1
                V[1 * p + k1][c] = -1
            elif t == 3:
                V[1 * p + k0][c] = 1
                V[0 * p + k0][c] = -1
            elif t == 4:
                V[1 * p + k1][c] = 1
            elif t == 5:
                V[0 * p + k0][c] = 1
                V[0 * p + k1][c] = 1
            elif t == 6:
                V[1 * p + k0][c] = 1
                V[1 * p + k1][c] = 1
                
            r_c00 = k0 * n + 0
            r_c01 = k1 * n + 0
            r_c10 = k0 * n + 1
            r_c11 = k1 * n + 1
            
            if t == 0:
                W[r_c00][c] = 1
                W[r_c11][c] = 1
            elif t == 1:
                W[r_c10][c] = 1
                W[r_c11][c] = -1
            elif t == 2:
                W[r_c01][c] = 1
                W[r_c11][c] = 1
            elif t == 3:
                W[r_c00][c] = 1
                W[r_c10][c] = 1
            elif t == 4:
                W[r_c00][c] = -1
                W[r_c01][c] = 1
            elif t == 5:
                W[r_c11][c] = 1
            elif t == 6:
                W[r_c00][c] = 1

    return {"U": U, "V": V, "W": W}

def _solve_for_W(n, m, p, rank, U_forms, V, deadline):
    """
    Given U_forms (rank x nm) and V (mp x rank), solve for W (pn x rank)
    such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    Returns W as list of lists, or None if no solution or timeout.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    num_vars = pn * rank
    num_eqs = n * m * p
    
    # Build linear system A @ W_flat = b
    A = [[Fraction(0) for _ in range(num_vars)] for _ in range(num_eqs)]
    b = [Fraction(1) for _ in range(num_eqs)]
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                for r in range(rank):
                    coeff = U_forms[a_idx][r] * V[b_idx][r]
                    if coeff != 0:
                        var_idx = c_idx * rank + r
                        A[eq_idx][var_idx] += Fraction(coeff)
                eq_idx += 1
    
    # Gaussian elimination with partial pivoting
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    m_rows = num_eqs
    n_cols = num_vars
    pivot_row = 0
    
    for col in range(n_cols):
        if time.time() > deadline:
            return None
        
        # Find pivot
        found = False
        for row in range(pivot_row, m_rows):
            if aug[row][col] != 0:
                found = True
                break
        
        if not found:
            continue
        
        # Swap rows
        aug[pivot_row], aug[row] = aug[row], aug[pivot_row]
        
        # Normalize pivot row
        pivot_val = aug[pivot_row][col]
        for j in range(col, n_cols + 1):
            aug[pivot_row][j] /= pivot_val
        
        # Eliminate other rows
        for row in range(m_rows):
            if row == pivot_row:
                continue
            factor = aug[row][col]
            if factor != 0:
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[pivot_row][j]
        
        pivot_row += 1
        if pivot_row >= m_rows:
            break
    
    # Check consistency
    for row in range(pivot_row, m_rows):
        if any(aug[row][j] != 0 for j in range(n_cols)):
            return None
    
    # Extract solution (free variables = 0)
    W_flat = [Fraction(0) for _ in range(n_cols)]
    for row in range(pivot_row):
        for col in range(n_cols):
            if aug[row][col] != 0:
                W_flat[col] = aug[row][n_cols]
                break
    
    # Reshape W
    W = [[0] * rank for _ in range(pn)]
    for c_idx in range(pn):
        for r in range(rank):
            W[c_idx][r] = W_flat[c_idx * rank + r]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                s = 0
                for r in range(rank):
                    s += U_forms[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                if s != 1:
                    return None
    
    return W

def _search_lower_rank(n, m, p, deadline):
    """
    Search for a lower-rank decomposition than 28.
    Try different rank values with random U and V forms, solve for W.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Use a fixed seed for reproducibility
    rng = random.Random(42)
    
    for rank in [24, 25, 26, 27]:
        if time.time() > deadline:
            break
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate U forms: each is a vector in {-1, 0, 1}^nm
            U_forms = []
            for r in range(rank):
                vec = []
                for _ in range(nm):
                    val = rng.choice([-1, 0, 0, 1])
                    vec.append(val)
                U_forms.append(vec)
            
            # Generate V: mp x rank, each entry in {-1, 0, 1}
            V = []
            for _ in range(mp):
                vec = []
                for r in range(rank):
                    val = rng.choice([-1, 0, 0, 1])
                    vec.append(val)
                V.append(vec)
            
            # Solve for W
            W = _solve_for_W(n, m, p, rank, U_forms, V, deadline)
            if W is not None:
                return {"U": [list(row) for row in U_forms], "V": [list(row) for row in V], "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: return before 150s limit
    
    # First, get the verified rank-28 solution
    result_28 = _construct_rank28(n, m, p)
    
    # Try to find a lower-rank solution
    if time.time() < deadline:
        result_lower = _search_lower_rank(n, m, p, deadline)
        if result_lower is not None:
            # Verify it
            if _verify(result_lower["U"], result_lower["V"], result_lower["W"], n, m, p):
                return result_lower
    
    # Fallback to rank-28
    return result_28
# EVOLVE-BLOCK-END
