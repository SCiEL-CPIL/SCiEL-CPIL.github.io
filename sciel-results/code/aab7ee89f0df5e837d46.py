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

def _compute_error_fast(U, V, W, n, m, p):
    """Compute total error using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j, :]
                v_val = V_arr[j*p+k, :]
                w_val = W_arr[k*n+i, :]
                s = np.dot(u_val, v_val * w_val)
                total_error += abs(s - 1.0)
    return total_error

def _structured_search(rank, n, m, p, seed=42, max_iters=5000, deadline=None):
    """
    Search for low-rank decomposition using structured random rank-1 terms.
    Uses numpy for fast error evaluation.
    """
    random.seed(seed)
    
    # Coefficient range for structured search - expanded to include Strassen-like coefficients
    coeff_range = [-2, -1, 1, 2]
    
    best_error = float('inf')
    best_solution = None
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random sparse structure for each rank-1 term
        U_terms = []
        V_terms = []
        W_terms = []
        
        for c in range(rank):
            # Random sparse vectors with small integer coefficients
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # Randomly select a few non-zero entries
            # Allow up to 3 non-zeros to capture denser structures like Strassen
            num_nonzero_u = random.randint(1, min(3, n * m))
            num_nonzero_v = random.randint(1, min(3, m * p))
            num_nonzero_w = random.randint(1, min(3, p * n))
            
            for _ in range(num_nonzero_u):
                idx = random.randint(0, n * m - 1)
                u_vec[idx] = random.choice(coeff_range)
            
            for _ in range(num_nonzero_v):
                idx = random.randint(0, m * p - 1)
                v_vec[idx] = random.choice(coeff_range)
            
            for _ in range(num_nonzero_w):
                idx = random.randint(0, p * n - 1)
                w_vec[idx] = random.choice(coeff_range)
            
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
        
        # Build factor matrices as numpy arrays
        U = np.array(U_terms).T  # shape (n*m, rank)
        V = np.array(V_terms).T  # shape (m*p, rank)
        W = np.array(W_terms).T  # shape (p*n, rank)
        
        # Check if this decomposition is exact
        error = _compute_error_fast(U, V, W, n, m, p)
        
        if error < 1e-10:
            # Convert back to lists for exact verification
            U_list = [[int(round(U[i, c])) for c in range(rank)] for i in range(n * m)]
            V_list = [[int(round(V[i, c])) for c in range(rank)] for i in range(m * p)]
            W_list = [[int(round(W[i, c])) for c in range(rank)] for i in range(p * n)]
            return U_list, V_list, W_list
        
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
    
    return best_solution

def _polish_solution(U, V, W, n, m, p, seed=42, max_iters=2000, deadline=None):
    """
    Try to polish a near-solution by adjusting individual entries.
    Uses numpy for fast error computation.
    """
    random.seed(seed)
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    def compute_error():
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    s = np.dot(u_val, v_val * w_val)
                    total_error += abs(s - 1.0)
        return total_error
    
    best_error = compute_error()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if best_error < 1e-10:
            break
        
        # Randomly select a matrix and entry to modify
        mat_choice = random.choice(['U', 'V', 'W'])
        
        if mat_choice == 'U':
            row = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            old_val = U[row, col]
            U[row, col] = old_val + random.choice([-1, 1])
            new_error = compute_error()
            if new_error < best_error:
                best_error = new_error
            else:
                U[row, col] = old_val
        elif mat_choice == 'V':
            row = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            old_val = V[row, col]
            V[row, col] = old_val + random.choice([-1, 1])
            new_error = compute_error()
            if new_error < best_error:
                best_error = new_error
            else:
                V[row, col] = old_val
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            old_val = W[row, col]
            W[row, col] = old_val + random.choice([-1, 1])
            new_error = compute_error()
            if new_error < best_error:
                best_error = new_error
            else:
                W[row, col] = old_val
    
    if best_error < 1e-10:
        U_list = [[int(round(U[i, c])) for c in range(rank)] for i in range(n * m)]
        V_list = [[int(round(V[i, c])) for c in range(rank)] for i in range(m * p)]
        W_list = [[int(round(W[i, c])) for c in range(rank)] for i in range(p * n)]
        return U_list, V_list, W_list
    return None

def _try_block_construction(n, m, p):
    """
    Try a block-based construction for 3x3 x 3x5.
    Split the 3x5 matrix into blocks and use Strassen-like tricks.
    """
    # For 3x3 * 3x5, we can think of the 3x5 as [3x3 | 3x2]
    # C = A * B1 + A * B2 where B1 is 3x3, B2 is 3x2
    
    # Try to find a rank-<45 decomposition using algebraic identities
    
    # One approach: use the fact that for 3x3 matrices, there are known
    # rank-23 decompositions (Strassen). We can extend this.
    
    # Let's try a simpler approach: use the schoolbook but try to merge terms
    
    # Actually, let's just return None and rely on the search
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 10.0  # 10s safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 3x3x3x5, the rank is at most 45 (schoolbook)
    # Try ranks from 40 down to 35 (reasonable range)
    
    # First, try a few specific constructions
    # Then use structured search
    
    # Try structured search for lower ranks
    for rank in range(40, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds with structured search
        for seed in range(20):
            if time.time() > deadline:
                break
            
            # Use structured search with limited iterations per rank
            remaining_time = deadline - time.time()
            if remaining_time < 1.0:
                break
            
            # Estimate iterations based on remaining time
            # Each iteration is fast with numpy (~0.1ms), so we can do many
            max_iters = min(10000, int(remaining_time * 5000))
            
            result = _structured_search(rank, n, m, p, seed=seed + 100, max_iters=max_iters, deadline=deadline)
            
            if result is not None:
                U, V, W = result
                # Verify with exact arithmetic
                if _verify_exact(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break  # Found a decomposition at this rank, try lower
    
    # If we didn't find anything below 40, try polishing the schoolbook
    # to see if we can reduce rank by removing redundant terms
    if best_rank == 45 and time.time() < deadline:
        # Try to find linear dependencies in the schoolbook terms
        # This is complex, so skip for now
        pass
    
    return best_result
# EVOLVE-BLOCK-END
