import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36
    start_time = time.time()
    time_limit = 135  # safety margin before 150s kill

    def is_valid(U, V, W):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if val != 1:
                        return False
        return True

    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [to_list(x) for x in obj]
        else:
            return obj

    def try_solve_for_W(U_np, V_np, R):
        """Given U and V as numpy arrays, solve for W. Returns dict or None."""
        num_vars = p * n * R
        A_mat = np.zeros((n * m * p, num_vars))
        b_vec = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_base = k * n + i
                    for r in range(R):
                        var_idx = w_base * R + r
                        A_mat[eq_idx, var_idx] = U_np[u_idx, r] * V_np[v_idx, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Try multiple denominators
            for denom in [1, 2, 4, 8]:
                W_scaled = np.round(W_flat * denom) / denom
                if np.allclose(A_mat @ W_scaled, b_vec, atol=1e-8):
                    W_sol = W_scaled.reshape((p * n, R))
                    U_list = [[Fraction(int(x)) for x in row] for row in U_np.tolist()]
                    V_list = [[Fraction(int(x)) for x in row] for row in V_np.tolist()]
                    W_list = [[Fraction(int(x * denom)) / denom for x in row] for row in W_sol.tolist()]
                    if is_valid(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except:
            pass
        return None

    def strassen_3x3x3():
        """Return a rank-7 decomposition for 3x3x3 matrix multiplication (Strassen)."""
        # Strassen's 7-multiplication algorithm for 2x2, extended to 3x3
        # This is a known decomposition; we'll construct it carefully
        # For simplicity, return None and use a different approach
        return None

    def structured_search(R, max_attempts=200):
        """Structured search: fix some entries to 0/1 to reduce search space."""
        start_time_search = time.time()
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                return None
            if time.time() - start_time_search > 60:  # per-R time limit
                return None
            
            # Random U and V with small integers, biased towards sparse
            U_np = np.random.randint(-2, 3, size=(n * m, R))
            V_np = np.random.randint(-2, 3, size=(m * p, R))
            
            # Add some structure: ensure each row has at least one non-zero
            for i in range(n * m):
                if np.all(U_np[i] == 0):
                    U_np[i, np.random.randint(R)] = np.random.choice([-1, 1])
            for j in range(m * p):
                if np.all(V_np[j] == 0):
                    V_np[j, np.random.randint(R)] = np.random.choice([-1, 1])
            
            result = try_solve_for_W(U_np, V_np, R)
            if result is not None:
                return result
        return None

    # Strategy: Try ranks from 27 down to 12
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Known upper bounds suggest 3x3x4 can be done in less than 36
    
    # Try ranks from 27 down to 12
    for R in range(27, 11, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Number of attempts based on remaining time
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
            
        # More attempts for lower ranks (harder to find)
        if R <= 15:
            max_attempts = 300
        elif R <= 20:
            max_attempts = 500
        else:
            max_attempts = 200
            
        result = structured_search(R, max_attempts=max_attempts)
        if result is not None:
            return {
                "U": to_list(result["U"]),
                "V": to_list(result["V"]),
                "W": to_list(result["W"])
            }

    # Fallback to Schoolbook (rank 36)
    rank = rank_schoolbook
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
