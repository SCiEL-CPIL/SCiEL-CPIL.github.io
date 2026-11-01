"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i, j, k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    # New Strategy: Integer Linear Programming (ILP)
    # We search for matrices U, V, W with entries in {-1, 0, 1}.
    # This is a subset of valid half-integers.
    # The problem is non-linear (product of 3 variables).
    # We linearize by introducing auxiliary variables.
    
    # To keep the problem tractable within 120s, we must limit the rank r.
    # The number of variables grows as O(r * (nm + mp + pn) + r^2 * nmp).
    # For n=2, m=4, p=6:
    # nm=8, mp=24, pn=12.
    # Variables: 44*r + 24*r^2.
    # If r=20: 880 + 9600 = 10480 variables.
    # Constraints: 24 equations.
    # This might be too large for a quick solve.
    
    # Let's try a smaller rank first, e.g., r=10, 12, 15.
    # If we find a solution, we return it.
    
    def solve_ilp(r):
        """
        Solves for U, V, W of rank r with entries in {-1, 0, 1}.
        Returns dict or None.
        """
        if time.time() > deadline - 5:
            return None
            
        # Variables:
        # u[i, j, r] -> index (i*m+j)*r + rr
        # v[j, k, r] -> index (j*p+k)*r + rr
        # w[k, i, r] -> index (k*n+i)*r + rr
        # aux[eq, r] -> index eq * r + rr, where eq is (i*m+j)*p+k
        
        num_u = n * m * r
        num_v = m * p * r
        num_w = p * n * r
        num_eqs = n * m * p
        num_aux = num_eqs * r
        
        total_vars = num_u + num_v + num_w + num_aux
        
        # Bounds: -1 <= x <= 1
        lb = np.full(total_vars, -1.0)
        ub = np.full(total_vars, 1.0)
        bounds = Bounds(lb, ub)
        
        # Integrality: all variables are integers
        integrality = np.ones(total_vars)
        
        # Objective: minimize 0 (feasibility problem)
        c = np.zeros(total_vars)
        
        # Constraints:
        # For each (i, j, k), sum_r u[i,j,r] * v[j,k,r] * w[k,i,r] = 1
        # We linearize u*v*w by introducing aux variables.
        # Actually, direct linearization of x*y*z is complex.
        # Simpler approach: Fix one matrix, e.g., W, to be sparse?
        # Or use a different linearization.
        
        # Alternative: Since we only need ONE solution, we can try to enforce
        # that for each (i,j,k), exactly one term in the sum is 1, and the rest 0?
        # No, that's too restrictive.
        
        # Let's try a simpler ILP formulation:
        # We can't easily linearize the product of 3 binary-like variables in a standard LP.
        # However, we can use the fact that if entries are in {-1,0,1}, 
        # we can fix W to be a "selection" matrix?
        
        # Let's try a different angle: 
        # We know rank 48 works.
        # Can we merge columns?
        # The ILP approach above is likely too slow for r>10 due to variable count.
        
        # Let's try a heuristic search instead: Local Search / Simulated Annealing on the discrete space.
        # Start with rank 48 (schoolbook).
        # Try to merge two columns (i.e., set one column to zero and adjust others? No, that changes rank).
        # Actually, we can try to find a rank r < 48 by:
        # 1. Randomly initialize U, V, W with rank r and entries in {-1, 0, 1}.
        # 2. Check if the tensor is reconstructed.
        # 3. If not, flip random entries to minimize error.
        
        # This is a discrete optimization problem.
        
        def discrete_search(r, max_iters=5000):
            if time.time() > deadline - 5:
                return None
                
            # Initialize with random sparse matrices
            # To make it easier, let's start with a subset of the schoolbook columns.
            # Pick r random (i,j,k) tuples from the 48.
            # This gives a valid rank-r "sub-decomposition" but it only covers r of the 24 tensor entries (with value 1), and leaves the rest 0.
            # We need to adjust U, V, W so that ALL 24 entries become 1.
            
            # Let's start with the first r schoolbook columns.
            indices = list(range(48))
            random.shuffle(indices)
            selected = indices[:r]
            
            U = np.zeros((n*m, r), dtype=int)
            V = np.zeros((m*p, r), dtype=int)
            W = np.zeros((p*n, r), dtype=int)
            
            for idx, col in enumerate(selected):
                i = col // 24
                rem = col % 24
                j = rem // 6
                k = rem % 6
                
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
            
            # Now we have a decomposition that gives 1 for the 'selected' (i,j,k) and 0 for others.
            # We want to change entries to make all (i,j,k) equal to 1.
            
            # Error function: sum (T[i,j,k] - sum_r ...)^2
            # We will use a simple hill-climbing / random restart approach.
            
            def calculate_error():
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for rr in range(r):
                                s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            err += (s - 1.0)**2
                return err
            
            current_err = calculate_error()
            if current_err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            best_err = current_err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random matrix, row, and col
                mat_choice = random.choice([0, 1, 2])
                if mat_choice == 0:
                    mat = U
                    rows = n * m
                elif mat_choice == 1:
                    mat = V
                    rows = m * p
                else:
                    mat = W
                    rows = p * n
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, r - 1)
                
                # Flip value: -1 <-> 0 <-> 1
                old_val = mat[row, col]
                # Try all other values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    mat[row, col] = new_val
                    new_err = calculate_error()
                    
                    if new_err < best_err:
                        best_err = new_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                        if best_err < 1e-9:
                            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                    else:
                        # Revert
                        mat[row, col] = old_val
            
            # If we didn't find a solution, return the best found (which is likely still invalid)
            # But we only return if valid.
            return None
            
        # Try ranks from 40 down to 20
        # Start with higher ranks as they are closer to the known solution.
        target_ranks = [47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
        
        # To save time, we will limit the search per rank.
        # If we find a solution, we return it immediately.
        # We prioritize lower ranks for better score, but higher ranks are easier to find.
        # Let's try a mix: 47, 40, 30, 25, 20.
        
        # Actually, let's just try a few specific ranks with a tight time limit.
        search_ranks = [47, 45, 40, 35, 30, 25, 20]
        
        for r in search_ranks:
            if time.time() > deadline - 10:
                break
                
            res = discrete_search(r, max_iters=2000)
            if res is not None:
                best_result = res
                best_rank = r
                # If we find a low rank, break?
                # Given the difficulty, any improvement is good.
                if r < 40:
                    break

    return best_result

# EVOLVE-BLOCK-END
