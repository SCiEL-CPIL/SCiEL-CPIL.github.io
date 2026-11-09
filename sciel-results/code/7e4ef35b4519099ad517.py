"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds

def _verify_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using rational arithmetic."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(rank):
                    u_val = U[i*m+j][c] if isinstance(U, list) else U[i*m+j, c]
                    v_val = V[j*p+k][c] if isinstance(V, list) else V[j*p+k, c]
                    w_val = W[k*n+i][c] if isinstance(W, list) else W[k*n+i, c]
                    # Convert to Fraction
                    if isinstance(u_val, Fraction):
                        uf = u_val
                    else:
                        uf = Fraction(u_val).limit_denominator(10000)
                    if isinstance(v_val, Fraction):
                        vf = v_val
                    else:
                        vf = Fraction(v_val).limit_denominator(10000)
                    if isinstance(w_val, Fraction):
                        wf = w_val
                    else:
                        wf = Fraction(w_val).limit_denominator(10000)
                    s += uf * vf * wf
                if s != Fraction(1, 1):
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact within tolerance."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
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

def _structured_search(rank, n, m, p, seed=42, max_iters=5000):
    """
    Search for low-rank decomposition using structured random rank-1 terms.
    Each term is a product of three sparse vectors with small integer coefficients.
    """
    random.seed(seed)
    
    # Coefficient range for structured search
    coeff_range = [-2, -1, 0, 1, 2]
    
    best_error = float('inf')
    best_solution = None
    
    for iteration in range(max_iters):
        # Generate random sparse structure for each rank-1 term
        U_terms = []
        V_terms = []
        W_terms = []
        
        for c in range(rank):
            # Random sparse vectors with small integer coefficients
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # Randomly select a few non-zero entries
            num_nonzero_u = random.randint(1, min(3, n * m))
            num_nonzero_v = random.randint(1, min(3, m * p))
            num_nonzero_w = random.randint(1, min(3, p * n))
            
            for _ in range(num_nonzero_u):
                idx = random.randint(0, n * m - 1)
                u_vec[idx] = random.choice(coeff_range)
                if u_vec[idx] == 0:
                    u_vec[idx] = 1
            
            for _ in range(num_nonzero_v):
                idx = random.randint(0, m * p - 1)
                v_vec[idx] = random.choice(coeff_range)
                if v_vec[idx] == 0:
                    v_vec[idx] = 1
            
            for _ in range(num_nonzero_w):
                idx = random.randint(0, p * n - 1)
                w_vec[idx] = random.choice(coeff_range)
                if w_vec[idx] == 0:
                    w_vec[idx] = 1
            
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
        
        # Build factor matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            for i in range(n * m):
                U[i][c] = U_terms[c][i]
            for j in range(m * p):
                V[j][c] = V_terms[c][j]
            for k in range(p * n):
                W[k][c] = W_terms[c][k]
        
        # Check if this decomposition is exact
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    error += abs(s - 1)
        
        if error == 0:
            return U, V, W
        
        if error < best_error:
            best_error = error
            best_solution = (U, V, W)
    
    return best_solution

def _polish_solution(U, V, W, n, m, p, seed=42, max_iters=2000):
    """
    Try to polish a near-solution by adjusting individual entries.
    """
    random.seed(seed)
    
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    rank = len(U[0])
    
    def compute_error(U, V, W):
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    error += abs(s - 1)
        return error
    
    best_error = compute_error(U, V, W)
    
    for iteration in range(max_iters):
        if best_error == 0:
            return U, V, W
        
        # Randomly select a matrix and entry to modify
        mat_choice = random.choice(['U', 'V', 'W'])
        
        if mat_choice == 'U':
            row = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            old_val = U[row][col]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            U[row][col] = new_val
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
            else:
                U[row][col] = old_val
        elif mat_choice == 'V':
            row = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            old_val = V[row][col]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            V[row][col] = new_val
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
            else:
                V[row][col] = old_val
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            old_val = W[row][col]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            W[row][col] = new_val
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
            else:
                W[row][col] = old_val
    
    if best_error == 0:
        return U, V, W
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 15.0  # 15s safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search for lower ranks
    # For 3x3x3x5, try ranks from 20 to 44
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds with structured search
        for seed in range(10):
            if time.time() > deadline:
                break
            
            # Use structured search with limited iterations per rank
            remaining_time = deadline - time.time()
            if remaining_time < 2.0:
                break
            
            # Estimate iterations based on remaining time
            max_iters = min(3000, int(remaining_time * 500))
            
            result = _structured_search(rank, n, m, p, seed=seed + 100, max_iters=max_iters)
            
            if result is not None:
                U, V, W = result
                # Verify with exact arithmetic
                if _verify_exact(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break  # Found a decomposition at this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
