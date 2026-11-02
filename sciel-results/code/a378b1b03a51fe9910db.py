import numpy as np
import random
import time
from fractions import Fraction

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
    time_limit = 115  # Leave 5s buffer for the evaluator's 120s limit
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    def verify_decomposition(U, V, W, rank):
        """
        Verifies sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def round_to_half_integer(val):
        """Round a value to the nearest half-integer (multiple of 0.5)."""
        return round(val * 2) / 2.0

    def solve_and_round(U, V, R):
        """
        Solves for W given U and V using least squares, then rounds to half-integers.
        Returns the rounded W if it exactly satisfies the decomposition, else None.
        """
        num_eqs = n * m * p
        num_unk = R * p * n
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            w_idx = (k * n + i) * R + c
                            A[eq_idx, w_idx] = coeff
                    eq_idx += 1
        
        try:
            # Use lstsq
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # If the system is not consistent, residuals will be high.
            # Even if consistent, the solution might not be rational.
            # We round to half-integers and check.
            
            W_sol = W_flat.reshape((p * n, R))
            W_rounded = [[round_to_half_integer(val) for val in row] for row in W_sol]
            
            # Verify the rounded solution
            if verify_decomposition(U, V, W_rounded, R):
                return W_rounded
            else:
                return None
        except:
            return None

    def local_search(U, V, R, max_iters=100):
        """
        Performs local search on U and V to find a valid W.
        Starts with current U, V.
        Tries to flip signs or change 0s to 1s/-1s in U and V.
        """
        current_U = [row[:] for row in U]
        current_V = [row[:] for row in V]
        
        # First try the initial state
        W_sol = solve_and_round(current_U, current_V, R)
        if W_sol is not None:
            return current_U, current_V, W_sol
            
        for _ in range(max_iters):
            # Randomly perturb U or V
            if random.random() < 0.5:
                # Perturb U
                i = random.randint(0, nm - 1)
                c = random.randint(0, R - 1)
                old_val = current_U[i][c]
                # Choose a new value from {-1, 0, 1}
                new_val = random.choice([-1, 0, 1])
                current_U[i][c] = new_val
                
                W_sol = solve_and_round(current_U, current_V, R)
                if W_sol is not None:
                    return current_U, current_V, W_sol
            else:
                # Perturb V
                j = random.randint(0, mp - 1)
                c = random.randint(0, R - 1)
                old_val = current_V[j][c]
                new_val = random.choice([-1, 0, 1])
                current_V[j][c] = new_val
                
                W_sol = solve_and_round(current_U, current_V, R)
                if W_sol is not None:
                    return current_U, current_V, W_sol
                    
        return None, None, None

    # Search ranks
    # We try ranks from a lower bound up to schoolbook-1
    # Lower bound for 3x4x5 is not trivial, but let's start at 30.
    ranks_to_try = list(range(30, 60))
    
    # If we have time, we can also try to improve upon the schoolbook by local search on the schoolbook?
    # But schoolbook is 60. We want < 60.
    
    for R in ranks_to_try:
        if R >= best_rank:
            break
            
        if time.time() - start_time > time_limit:
            break
            
        # Determine time budget for this rank
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 5:
            break
            
        # Number of restarts depends on time remaining
        # Each restart involves a linear solve (O(nmp * R^2) roughly) and local search.
        # Let's estimate cost. n=3, m=4, p=5. nmp=60.
        # If R=40, A is 60x(40*15)=60x600. Lstsq is fast.
        # Local search adds overhead.
        
        max_restarts = 2000
        if time_remaining < 20:
            max_restarts = 200
        elif time_remaining < 50:
            max_restarts = 500
            
        for r_idx in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V sparsely
            # Density: 0.2 to 0.4 is often good for sparse systems
            density = 0.3
            U_rand = [[0] * R for _ in range(nm)]
            V_rand = [[0] * R for _ in range(mp)]
            
            for i in range(nm):
                for c in range(R):
                    if random.random() < density:
                        U_rand[i][c] = random.choice([-1, 1])
                        
            for j in range(mp):
                for c in range(R):
                    if random.random() < density:
                        V_rand[j][c] = random.choice([-1, 1])
            
            # Try direct solve + round
            W_sol = solve_and_round(U_rand, V_rand, R)
            if W_sol is not None:
                best_rank = R
                best_U = U_rand
                best_V = V_rand
                best_W = W_sol
                return {"U": best_U, "V": best_V, "W": best_W}
            
            # If direct solve fails, try local search
            # Limit local search iterations to keep time bounded
            ls_iters = 10 if time_remaining < 20 else 50
            U_sol, V_sol, W_ls = local_search(U_rand, V_rand, R, max_iters=ls_iters)
            if W_ls is not None:
                best_rank = R
                best_U = U_sol
                best_V = V_sol
                best_W = W_ls
                return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
