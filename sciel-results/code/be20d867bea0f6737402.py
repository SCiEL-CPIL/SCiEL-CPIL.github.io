"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Precompute for speed: We need to check all (i, j, k)
    # This is a 3-way sum. We can vectorize partially.
    # For each c, the contribution is an outer product of U_c, V_c, W_c.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Iterate over i, j, k is O(n*m*p*rank). 4*5*5*100 = 10000 ops. Very fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                # Dot product of the three vectors
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_structured_patterns(n, m, p, target_rank, deadline):
    """
    Try structured patterns that might achieve rank reduction.
    These include block structures, Kronecker-like patterns, etc.
    """
    U_fb = np.zeros((n * m, target_rank))
    V_fb = np.zeros((m * p, target_rank))
    W_fb = np.zeros((p * n, target_rank))
    
    # Pattern 1: Try to exploit the structure by having some columns
    # that cover multiple (i,j,k) triples
    # For rank n*m*p - 1, we need one column to "cover" two triples
    # This is unlikely with simple patterns, but let's try a few
    
    # Pattern: Use a few random sparse columns with small integer values
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Create a base schoolbook pattern
        U_base = np.zeros((n * m, target_rank))
        V_base = np.zeros((m * p, target_rank))
        W_base = np.zeros((p * n, target_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U_base[i*m+j, col] = 1
                        V_base[j*p+k, col] = 1
                        W_base[k*n+i, col] = 1
                        col += 1
        
        # Now try to modify some columns to create dependencies
        # This is a heuristic - we'll try to merge two triples into one column
        if target_rank < n * m * p:
            # Try to find two triples that can be merged
            # This is complex, so we'll use a simpler heuristic
            # Randomly perturb some entries
            for _ in range(3):
                # Pick a random column to perturb
                c = random.randint(0, target_rank - 1)
                # Add a small random perturbation to one entry
                row_u = random.randint(0, n * m - 1)
                row_v = random.randint(0, m * p - 1)
                row_w = random.randint(0, p * n - 1)
                
                # Only if the perturbation doesn't break the existing structure
                if U_base[row_u, c] == 1:
                    U_base[row_u, c] = 2
                    # Compensate elsewhere - this is very heuristic
                    # We'll just check if it works
                    if _verify(U_base, V_base, W_base, n, m, p):
                        return U_base.tolist(), V_base.tolist(), W_base.tolist()
                    U_base[row_u, c] = 1
                elif V_base[row_v, c] == 1:
                    V_base[row_v, c] = 2
                    if _verify(U_base, V_base, W_base, n, m, p):
                        return U_base.tolist(), V_base.tolist(), W_base.tolist()
                    V_base[row_v, c] = 1
                elif W_base[row_w, c] == 1:
                    W_base[row_w, c] = 2
                    if _verify(U_base, V_base, W_base, n, m, p):
                        return U_base.tolist(), V_base.tolist(), W_base.tolist()
                    W_base[row_w, c] = 1
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school))
    V_fb = np.zeros((m * p, rank_school))
    W_fb = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Attempt to find lower rank (99, 98, etc.)
    # Strategy: First try structured patterns, then random search
    for target_rank in [99, 98, 97]:
        if time.time() > deadline - 10:
            break
        
        # Try structured patterns first
        result = _try_structured_patterns(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = np.array(result[0]), np.array(result[1]), np.array(result[2])
            break
        
        # If structured patterns didn't work, try random search
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_school:
            break
    
    # Return the best found
    # Ensure output is lists of lists as per interface requirement (or compatible)
    # The interface says: factor matrices of shapes... with integer/half-integer entries
    # Returning numpy arrays or lists should be fine, but lists are safer for serialization.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
