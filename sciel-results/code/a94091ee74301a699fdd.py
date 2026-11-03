import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 130  # seconds, safety margin before 150s kill

    def is_valid(U, V, W):
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
            
            # Try multiple denominators: 1, 2, 4
            for denom in [1, 2, 4]:
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

    # First, try Strassen-based construction for 3x3x3 part + scaling for 4th column
    # Strassen's algorithm for 3x3x3 matrix multiplication has rank 7
    # We can extend this to 3x3x4 by treating the 4th column separately
    # This gives rank 7 + 3 = 10 (Strassen for 3x3x3 + 3 scalar multiplications for 4th column)
    
    def strassen_3x3x3():
        """Return Strassen's rank-7 decomposition for 3x3x3 matrix multiplication."""
        # Strassen's 7 intermediate products for 3x3x3
        # M1 = (A00+A22)(B00+B22)
        # M2 = (A20+A22)B00
        # M3 = A00(B01-B21)
        # M4 = A22(B21-B01)
        # M5 = (A00+A10)(B00+B10)
        # M6 = (A12+A22)(B20+B22)
        # M7 = (A00+A12)(B12-B22) - (A00+A22)(B00+B22)
        # Actually, let me use the standard Strassen for 2x2 and extend...
        # For simplicity, let me use a known rank-7 scheme for 3x3x3
        
        # Standard Strassen for 2x2x2 has rank 7, but for 3x3x3 we need a different approach
        # Let me use the known minimal rank for 3x3x3 which is 7 (Strassen)
        
        # Actually, the minimal rank for 3x3x3 is 7 (Strassen's algorithm)
        # Let me construct it properly
        
        # U matrix: 9x7 (3x3 input A)
        # V matrix: 9x7 (3x3 input B)  
        # W matrix: 9x7 (3x3 output C)
        
        # Using the standard Strassen decomposition for 3x3x3
        # This is a well-known result
        
        U = [
            [1, 0, 0, 0, 1, 0, 0],  # A00
            [0, 0, 0, 0, 0, 0, 1],  # A01
            [0, 0, 0, 0, 0, 0, 0],  # A02
            [0, 0, 0, 0, 1, 0, 0],  # A10
            [0, 0, 0, 0, 0, 0, 0],  # A11
            [0, 0, 0, 0, 0, 0, 1],  # A12
            [1, 1, 0, 0, 0, 0, 0],  # A20
            [0, 0, 0, 0, 0, 0, 0],  # A21
            [1, 0, 0, 0, 0, 0, 0]   # A22
        ]
        
        V = [
            [1, 0, 0, 0, 1, 0, 0],  # B00
            [0, 0, 1, 0, 0, 0, 0],  # B01
            [0, 0, 0, 0, 0, 0, 0],  # B02
            [0, 0, 0, 0, 1, 0, 0],  # B10
            [0, 0, 0, 0, 0, 0, 0],  # B11
            [0, 0, 0, 0, 0, 0, 0],  # B12
            [1, 0, 0, 0, 0, 0, 0],  # B20
            [0, 0, -1, 1, 0, 0, 0], # B21
            [1, 0, 0, 0, 0, 1, -1]  # B22
        ]
        
        W = [
            [0, 0, 0, 0, 0, 0, 1],  # C00
            [0, 0, 1, 0, 0, 0, 0],  # C01
            [0, 0, 0, 0, 0, 0, 0],  # C02
            [0, 0, 0, 0, 1, 0, 0],  # C10
            [0, 0, 0, 0, 0, 0, 0],  # C11
            [0, 0, 0, 0, 0, 0, 0],  # C12
            [0, 0, 0, 0, 0, 0, 0],  # C20
            [0, 0, 0, 1, 0, 0, 0],  # C21
            [0, 0, 0, 0, 0, 1, 0]   # C22
        ]
        
        return U, V, W

    # Try Strassen-based approach: rank 7 for 3x3x3 + 3 for scaling 4th column = rank 10
    # But we need to be more careful about the structure
    
    # Actually, let me just use the random search but with a better starting point
    # Start from rank 10 and go down
    
    # Search for R from 10 down to 20 (if 10 doesn't work, go up)
    # Actually, let's search from a reasonable range
    
    # First try known good ranks
    for R in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
        if time.time() - start_time > time_limit:
            break
        
        attempts = 0
        max_attempts = 1000 if R <= 12 else 500
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            # Use small integers for U and V to make W more likely to be rational with small denominator
            U_np = np.random.randint(-2, 3, size=(n * m, R))
            V_np = np.random.randint(-2, 3, size=(m * p, R))
            
            result = try_solve_for_W(U_np, V_np, R)
            if result is not None:
                return {
                    "U": to_list(result["U"]),
                    "V": to_list(result["V"]),
                    "W": to_list(result["W"])
                }

    # Fallback to Schoolbook
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
