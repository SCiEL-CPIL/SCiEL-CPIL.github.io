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
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def random_decomposition(r, seed=None):
        """Generate a random decomposition with entries in {-1, 0, 1}."""
        if seed is not None:
            random.seed(seed)
        
        U = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(p*n)]
        
        return {"U": U, "V": V, "W": W}

    def hill_climb(decomp, max_iters=1000, time_limit=5.0):
        """Perform hill climbing to find a valid decomposition."""
        U = [row[:] for row in decomp["U"]]
        V = [row[:] for row in decomp["V"]]
        W = [row[:] for row in decomp["W"]]
        r = len(U[0]) if U else 0
        
        def error():
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for rr in range(r):
                            s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                        err += (s - 1)**2
            return err
        
        current_err = error()
        if current_err < 1e-9:
            return decomp
        
        best_err = current_err
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random matrix, row, and column
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
            
            old_val = mat[row][col]
            # Try all other values in {-1, 0, 1}
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                
                mat[row][col] = new_val
                new_err = error()
                
                if new_err < best_err:
                    best_err = new_err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                    
                    if best_err < 1e-9:
                        return {"U": best_U, "V": best_V, "W": best_W}
                else:
                    # Revert
                    mat[row][col] = old_val
        
        # Return best found (may still be invalid)
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try structured decomposition using bilinear tricks
    # For 2x4 * 4x6, we can exploit the structure
    # Let's try rank 18-30 with structured initialization
    
    def structured_decomposition(r, seed=None):
        """
        Structured initialization:
        For 2x4 * 4x6:
        U has shape (8, r), V has shape (24, r), W has shape (12, r)
        
        Strategy: Use the fact that matrix multiplication can be decomposed
        into rank-1 terms. We'll use a semi-structured approach where
        we initialize based on Kronecker product structure.
        """
        if seed is not None:
            random.seed(seed)
        
        # Initialize with a pattern that has some structure
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # For rank r, assign columns in a structured way
        # Each column corresponds to a "term" in the decomposition
        # We'll use a round-robin assignment based on (i, j, k) indices
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    # This column "covers" the (i,j,k) term
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
        # For remaining columns, add random perturbation to help cover other terms
        while col < r:
            # Randomly assign to help cover missing terms
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            
            U[i*m+j][col] = random.choice([-1, 1])
            V[j*p+k][col] = random.choice([-1, 1])
            W[k*n+i][col] = random.choice([-1, 1])
            col += 1
        
        return {"U": U, "V": V, "W": W}

    # Search for lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) / something
    # For 2x4*4x6: nm=8, mp=24, pn=12
    # Lower bound is at least max(8,24,12) = 24? No, that's not right.
    # Actually, rank >= max(n, m, p) * min(n, m, p) / max(n, m, p) ... 
    # Let's just try ranks from 47 down to 20
    
    found = False
    for r in range(47, 19, -1):
        if time.time() > deadline - 15:
            break
        if found:
            break
        
        # Try structured initialization first
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            
            decomp = structured_decomposition(r, seed=seed*1000)
            
            if check_decomposition(decomp["U"], decomp["V"], decomp["W"]):
                best_result = decomp
                best_rank = r
                found = True
                break
            
            # Try hill climbing from this initialization
            result = hill_climb(decomp, max_iters=500, time_limit=3.0)
            if check_decomposition(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = r
                found = True
                break
        
        if found:
            break
        
        # Try pure random initialization
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            decomp = random_decomposition(r, seed=seed*777)
            result = hill_climb(decomp, max_iters=300, time_limit=2.0)
            if check_decomposition(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = r
                found = True
                break
    
    # If we didn't find anything better, return schoolbook
    if not found:
        best_result, best_rank = schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
