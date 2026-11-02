import numpy as np
import random
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 115  # seconds
    
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def is_half_integer_matrix(matrix, tol=1e-7):
        """Check if matrix entries are close to k/2 and return rounded version."""
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) > tol:
                    return None
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return result

    def solve_for_w(U, V, R):
        """
        Given U (nm x R) and V (mp x R), solve for W (pn x R) such that
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        Uses least squares. Returns W as list of lists or None if not half-integer.
        """
        # Build A and b
        # Equations indexed by (i, j, k) -> eq_idx = i * m * p + j * p + k
        # Unknowns indexed by (k, i, c) -> col_idx = (k * n + i) * R + c
        A = np.zeros((n * m * p, pn * R))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for c in range(R):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        w_idx = (k * n + i) * R + c
                        A[eq_idx, w_idx] = coeff
        
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
            
        W_sol = W_flat.reshape((pn, R))
        
        # Check for half-integers
        W_nice = is_half_integer_matrix(W_sol)
        if W_nice is None:
            return None
            
        # Verify with exact arithmetic if possible, or high precision float
        if verify_decomposition(U, V, W_nice, R):
            return W_nice
        return None

    # Strategy:
    # 1. Try structured random initializations for U and V.
    # 2. Solve for W via linear system.
    # 3. Check if W is "nice" (half-integers).
    # 4. Iterate ranks from lower bound upwards.
    
    # Lower bound is max(nm, mp, pn) = 20.
    # We know rank(3x4x5) is at least 20.
    # Let's try ranks from 20 to 40. If we find one, we return it immediately 
    # (since we are going in increasing order, the first valid one is minimal).
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Helper to generate random U, V with specific sparsity/density
    def gen_rand_UV(R, density=0.5):
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        
        for i in range(nm):
            for c in range(R):
                if random.random() < density:
                    U[i][c] = random.choice([-1, 1])
        for j in range(mp):
            for c in range(R):
                if random.random() < density:
                    V[j][c] = random.choice([-1, 1])
                    
        # Ensure no zero rows
        for i in range(nm):
            if all(x == 0 for x in U[i]):
                U[i][random.randint(0, R-1)] = random.choice([-1, 1])
        for j in range(mp):
            if all(x == 0 for x in V[j]):
                V[j][random.randint(0, R-1)] = random.choice([-1, 1])
                
        return U, V

    # We will try different ranks.
    # To maximize chance of finding a low rank, we start from the theoretical lower bound.
    # However, finding a valid decomposition at the very lower bound is hard.
    # We'll scan R from 20 to 55.
    
    # Optimization: Pre-compute some structures?
    # The linear system depends on U and V.
    
    # Let's try a few specific structures first.
    # Structure 1: Random dense
    # Structure 2: Random sparse
    
    # We will iterate R. For each R, we do multiple restarts.
    # If we find a valid decomposition for rank R, we return it.
    
    ranks_to_try = list(range(20, 56))
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        if R >= best_rank:
            continue
            
        # Number of restarts: allocate more time to lower ranks
        # If R is close to 20, try more.
        num_restarts = 1000 if R < 30 else 500
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random U, V
            # Vary density
            density = random.choice([0.3, 0.5, 0.7, 1.0])
            U, V = gen_rand_UV(R, density)
            
            W = solve_for_w(U, V, R)
            
            if W is not None:
                # Found a valid decomposition
                return {"U": U, "V": V, "W": W}
                
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
