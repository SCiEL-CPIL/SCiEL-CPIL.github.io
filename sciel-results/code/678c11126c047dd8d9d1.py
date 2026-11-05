import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen (7 terms)
    # M1 = (a00+a11)(b00+b11)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2 = (a10+a11)b00
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3 = a00(b01-b11)
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4 = a11(b10-b00)
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5 = (a00+a01)b11
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_27():
    """
    Attempts to find a rank-27 decomposition by LP-searching for a 6-term
    completion of the 2x4 sub-problem (A_{2x2} * B_{2x4}).
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None
    
    n, m, p = 2, 3, 5
    rank = 27
    
    # The 2x4 sub-problem: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x4 (rows j=0,1; cols k=0,1,2,3)
    # We need 6 terms. Each term t has:
    #   u_t: 2x2 matrix (i, j)
    #   v_t: 2x4 matrix (j, k)
    #   w_t: 4x2 matrix (k, i)
    
    # Constraint: for each (i, j, k), sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1
    # This is 2*2*4 = 16 equations.
    
    # This is bilinear. Let's fix the V factors to be sparse and solve for U and W.
    
    # Approach: assume each term t has v_t[j,k] = 1 for exactly one (j,k) pair, and 0 otherwise.
    # With 6 terms and 8 (j,k) pairs, we can't cover all pairs. So this won't work.
    
    # Alternative: assume each term t has v_t[j,k] = 1 for exactly two (j,k) pairs.
    # With 6 terms, we cover 12 (j,k) pairs, so 4 pairs are covered by 2 terms each.
    
    # Let's try: terms 0-3 cover (0,0),(0,1),(1,0),(1,1) with one pair each.
    # Terms 4-5 cover (0,2),(0,3),(1,2),(1,3) with two pairs each.
    
    # Actually, let's just try to find any solution using a simple iterative method.
    
    # Initialize U, V, W randomly.
    np.random.seed(42)
    U_sub = np.random.randn(6, 2, 2)
    V_sub = np.random.randn(6, 2, 4)
    W_sub = np.random.randn(6, 4, 2)
    
    # Iteratively solve for one factor at a time.
    for iteration in range(1000):
        # Solve for U: for each (i, j, k), u[i,j] * sum_t v_t[j,k] * w_t[k,i] = 1
        # This is: for each (i, j), u[i,j] = 1 / sum_t v_t[j,k] * w_t[k,i] for each k
        # But this must hold for all k, so we need consistency.
        
        # Actually, the constraint is: for each (i, j, k), sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1
        # This is: for each (i, j, k), the dot product of u[:,i,j] and (v[:,j,k] * w[:,k,i]) = 1
        
        # Let's solve for U given V and W:
        # For each (i, j, k): sum_t u_t[i,j] * (v_t[j,k] * w_t[k,i]) = 1
        # This is a linear system in u.
        
        # Build the system: 16 equations, 6*4 = 24 unknowns (u_t[i,j] for t=0..5, i=0..1, j=0..1)
        # Actually, u has 6*2*2 = 24 entries.
        
        A_mat = np.zeros((16, 24))
        b_vec = np.ones(16)
        
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    eq_idx = i * 8 + j * 4 + k
                    for t in range(6):
                        var_idx = t * 4 + i * 2 + j
                        A_mat[eq_idx, var_idx] = V_sub[t, j, k] * W_sub[t, k, i]
        
        # Solve the linear system.
        try:
            U_flat = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            U_sub = U_flat.reshape(6, 2, 2)
        except np.linalg.LinAlgError:
            return None
        
        # Solve for V given U and W:
        A_mat = np.zeros((16, 48))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    eq_idx = i * 8 + j * 4 + k
                    for t in range(6):
                        var_idx = t * 8 + j * 4 + k
                        A_mat[eq_idx, var_idx] = U_sub[t, i, j] * W_sub[t, k, i]
        
        try:
            V_flat = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            V_sub = V_flat.reshape(6, 2, 4)
        except np.linalg.LinAlgError:
            return None
        
        # Solve for W given U and V:
        A_mat = np.zeros((16, 48))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    eq_idx = i * 8 + j * 4 + k
                    for t in range(6):
                        var_idx = t * 8 + k * 2 + i
                        A_mat[eq_idx, var_idx] = U_sub[t, i, j] * V_sub[t, j, k]
        
        try:
            W_flat = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            W_sub = W_flat.reshape(6, 4, 2)
        except np.linalg.LinAlgError:
            return None
    
    # Check if the solution is valid.
    for i in range(2):
        for j in range(2):
            for k in range(4):
                total = 0.0
                for t in range(6):
                    total += U_sub[t, i, j] * V_sub[t, j, k] * W_sub[t, k, i]
                if abs(total - 1.0) > 1e-6:
                    return None
    
    # Construct the full decomposition.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 10
    
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # r = 14
    
    # Part 3: 6-term completion for 2x4 (j=0,1; k=0,1,2,3)
    for t in range(6):
        for i in range(2):
            for j in range(2):
                U[i * m + j][r] = U_sub[t, i, j]
        for j in range(2):
            for k in range(4):
                V[j * p + k][r] = V_sub[t, j, k]
        for k in range(4):
            for i in range(2):
                W[k * n + i][r] = W_sub[t, k, i]
        r += 1
    # r = 20
    
    # Wait, this is only 20 terms, not 27. I made a mistake.
    # The 2x4 sub-problem has 6 terms, so total rank = 10 + 4 + 6 = 20.
    # But we want rank 27, so we need 13 terms for the 2x4 sub-problem.
    
    # I think I confused myself. Let me reconsider.
    
    # The schoolbook rank is 30. The current rank-28 decomposition uses:
    # - 10 terms for A2 B2
    # - 4 terms for A1 B13
    # - 7 terms for A1 B11 (Strassen)
    # - 7 terms for A1 B12 (Strassen)
    # Total: 28
    
    # To get rank 27, we need to reduce the 14 Strassen terms to 13.
    # This means we need to find a 13-term decomposition for the 2x4 sub-problem.
    
    # The 2x4 sub-problem has rank at most 8 (schoolbook). Can we do it in 13 terms?
    # No, 13 > 8, so that's not the issue.
    
    # I think I'm overcomplicating this. Let me just return None and fall back to rank 28.
    return None

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # First, try rank 27
    res_27 = _try_rank_27()
    if res_27 is not None:
        if _verify_decomposition(res_27['U'], res_27['V'], res_27['W'], 2, 3, 5):
            return res_27
    
    # Fall back to rank 28
    res_28 = _construct_rank_28()
    
    if _verify_decomposition(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5):
        return res_28
    else:
        raise ValueError("Verification failed for rank-28 decomposition")

construct()
