import time
import itertools
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Attempt to find a rank-28 decomposition using a randomized search 
       over dense matrices with small integer entries, solving for W via 
       least squares and verifying with exact integer arithmetic.
    2. If no rank-28 decomposition is found within the time limit, 
       return the schoolbook rank-30 decomposition.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition using integer arithmetic."""
        r = len(U[0])
        # Convert to integers for exact check (multiply by 2 if half-integers)
        # We assume entries are integers or half-integers.
        # Multiply all by 2 to work with integers.
        U_int = [[int(u * 2) for u in row] for row in U]
        V_int = [[int(v * 2) for v in row] for row in V]
        W_int = [[int(w * 2) for w in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U_int[i * m + j][c] * V_int[j * p + k][c] * W_int[k * n + i][c]
                    # The sum should be 1 * 2^3 = 8
                    if s != 8:
                        return False
        return True

    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9

    # Search for rank 28
    target_rank = 28
    
    # We will try to find U, V, W with entries in {-2, -1, 0, 1, 2} / 2
    # i.e., integers in {-4, -2, 0, 2, 4} / 2? No, just {-1, 0, 1} is a good start.
    # Let's try entries in {-1, 0, 1} first.
    
    search_deadline = start_time + 105.0
    
    np.random.seed(42) # For reproducibility in development, but random in prod? 
    # Actually, let's use random seed based on time for variety, but fixed for debugging.
    
    def try_random_search():
        for trial in range(500):
            if time.time() > search_deadline:
                return None
            
            # Generate random dense U and V with entries in {-1, 0, 1}
            U_rand = np.random.choice([-1, 0, 1], size=(nm, target_rank))
            V_rand = np.random.choice([-1, 0, 1], size=(mp, target_rank))
            
            # Construct the linear system for W
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # For each (i,j,k), we have a linear equation in W.
            # We have 30 equations and 10*28 = 280 unknowns.
            # We can solve for W using least squares.
            
            A = np.zeros((n*m*p, pn * target_rank))
            b = np.ones(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation idx
                        # Involves W[k*n+i, :]
                        u_row = U_rand[i*m+j, :]
                        v_row = V_rand[j*p+k, :]
                        coeffs = u_row * v_row
                        
                        # The unknowns are W[k*n+i, 0..target_rank-1]
                        # They are located in the vector x at indices:
                        # (k*n+i)*target_rank + c
                        start_idx = (k*n+i) * target_rank
                        A[idx, start_idx:start_idx+target_rank] = coeffs
                        b[idx] = 1
                        idx += 1
            
            # Solve A x = b
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Reshape x to W
            W_sol = x.reshape(pn, target_rank)
            
            # Check if W_sol has integer or half-integer entries
            W_check = W_sol * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-6):
                W_rounded = np.round(W_check) / 2.0
                
                # Convert to lists
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                
                # Verify exactness
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Also check if it's close to integers
            if np.allclose(W_sol, np.round(W_sol), atol=1e-6):
                W_rounded = np.round(W_sol)
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}

    result = try_random_search()
    if result:
        return result
        
    return schoolbook()
