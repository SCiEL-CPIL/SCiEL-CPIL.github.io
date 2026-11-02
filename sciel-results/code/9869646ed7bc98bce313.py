"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    Strategy:
    1. Start with schoolbook (rank 112)
    2. Use structured Strassen-like combinations to reduce rank
    3. Use efficient vectorized verification
    4. Try to find exact integer/half-integer solutions via local search
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def verify_fast(U_np, V_np, W_np, tol=1e-9):
        """
        Efficiently verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
        Using vectorized operations.
        """
        rank = U_np.shape[1]
        # For each (i,j,k) triple, compute the sum
        # We can vectorize over r for each (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_detailed(U_np, V_np, W_np, tol=1e-9):
        """Verify and return the max error."""
        rank = U_np.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Schoolbook solution
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Strategy 1: Try to find a Strassen-like decomposition
    # For matrix multiplication, we can use the fact that:
    # If we have a rank-r decomposition for n'xmxp', we can extend to nxmxp by padding.
    # 
    # Key insight: The tensor T[i,j,k] = 1 for all i,j,k is a very special tensor.
    # A rank-1 term (u, v, w) contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k].
    # 
    # For the schoolbook, each term has u[i,j]=1 for one (i,j), v[j,k]=1 for one (j,k), w[k,i]=1 for one (k,i).
    # 
    # To reduce rank, we need to find linear combinations that still sum to 1.
    # This is equivalent to finding a low-rank decomposition of the all-ones tensor structure.
    
    # Let's try a different approach: use the structure of the problem.
    # The equation sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # can be rewritten as: for each (i,k), sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # But this doesn't directly help.
    
    # Let's try random search with better initialization.
    # The key challenge is that we need EXACT integer/half-integer solutions.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try various target ranks, starting from reasonable lower bounds
    # Lower bound considerations:
    # - The tensor has n*m*p = 112 "1" entries
    # - Each rank-1 term can "cover" at most n*m*p entries, but with structure
    # - Realistic lower bound for 4x4x7 is probably around 30-50
    
    # First, let's try to see if we can find a solution with rank < 112
    # using a smarter search.
    
    # Approach: Start with schoolbook and try to merge terms.
    # If two terms (u1,v1,w1) and (u2,v2,w2) can be replaced by one term
    # (u3,v3,w3) such that u1*v1*w1 + u2*v2*w2 = u3*v3*w3 (as tensors),
    # we reduce rank by 1.
    
    # This is hard to search directly. Instead, let's use a gradient-based approach
    # with random restarts, targeting specific ranks.
    
    def compute_error_vectorized(Ur, Vr, Wr):
        """Compute sum of squared errors for all (i,j,k)."""
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ur[i*m+j], Vr[j*p+k] * Wr[k*n+i])
                    errors.append((s - 1.0) ** 2)
        return sum(errors)
    
    # Try to find solutions for various ranks
    # We'll try ranks from 20 up to 111, with multiple restarts for lower ranks
    
    found_solution = False
    for target_rank in range(20, 112):
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Number of restarts: more for lower ranks (harder to find)
        num_restarts = max(5, min(30, 300 // target_rank))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            scale = 1.0 / (target_rank ** 0.5)
            U_r = np.random.randn(n * m, target_rank) * scale
            V_r = np.random.randn(m * p, target_rank) * scale
            W_r = np.random.randn(p * n, target_rank) * scale
            
            current_err = compute_error_vectorized(U_r, V_r, W_r)
            
            # Local search: coordinate descent with random perturbations
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if current_err < 1e-12:
                    break
                
                # Pick a random matrix and element to perturb
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    row = random.randint(0, n*m-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, 0.05)
                    old_val = U_r[row, colr]
                    U_r[row, colr] = old_val + delta
                    new_err = compute_error_vectorized(U_r, V_r, W_r)
                    if new_err > current_err:
                        U_r[row, colr] = old_val
                    else:
                        current_err = new_err
                elif mat_choice == 1:
                    row = random.randint(0, m*p-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, 0.05)
                    old_val = V_r[row, colr]
                    V_r[row, colr] = old_val + delta
                    new_err = compute_error_vectorized(U_r, V_r, W_r)
                    if new_err > current_err:
                        V_r[row, colr] = old_val
                    else:
                        current_err = new_err
                else:
                    row = random.randint(0, p*n-1)
                    colr = random.randint(0, target_rank-1)
                    delta = random.gauss(0, 0.05)
                    old_val = W_r[row, colr]
                    W_r[row, colr] = old_val + delta
                    new_err = compute_error_vectorized(U_r, V_r, W_r)
                    if new_err > current_err:
                        W_r[row, colr] = old_val
                    else:
                        current_err = new_err
            
            if current_err < 1e-10:
                if verify_fast(U_r, V_r, W_r, tol=1e-9):
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    best_rank = target_rank
                    found_solution = True
                    break
        
        if found_solution:
            # Continue searching for even lower rank
            continue
    
    # If we found a solution, try to snap to nearest half-integers and verify
    if best_rank < schoolbook_rank:
        # Try to quantize to half-integers
        def quantize(arr):
            """Quantize to nearest half-integer."""
            return np.round(arr * 2) / 2
        
        U_q = quantize(best_U)
        V_q = quantize(best_V)
        W_q = quantize(best_W)
        
        if verify_fast(U_q, V_q, W_q, tol=1e-9):
            best_U, best_V, best_W = U_q, V_q, W_q
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
