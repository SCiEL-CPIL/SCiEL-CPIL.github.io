import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 56
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def is_half_integer_matrix(M, tol=1e-9):
        """Check if matrix entries are close to half-integers (k/2)."""
        M2 = M * 2
        M2_rounded = np.round(M2)
        if np.allclose(M2, M2_rounded, atol=tol):
            return True
        return False

    def try_rank_integer_search(r, max_attempts=200):
        """
        Search for a rank-r decomposition by choosing integer U, W 
        and solving for V. Check if V is half-integer.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Integer values to try for U and W
        # Using small integers increases chance of small determinant in B
        int_vals = [-2, -1, 0, 1, 2]
        
        # We can also try -1, 0, 1
        int_vals_small = [-1, 0, 1]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Alternate between small and slightly larger integer sets
            if attempt % 2 == 0:
                vals = int_vals_small
            else:
                vals = int_vals
            
            U = np.random.choice(vals, size=(nm, r))
            W = np.random.choice(vals, size=(pn, r))
            
            V = np.zeros((mp, r), dtype=float)
            valid = True
            
            for j in range(m):
                for k in range(p):
                    # B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    B = np.array([
                        [U[i*m+j, c] * W[k*n+i, c] for c in range(r)]
                        for i in range(n)
                    ], dtype=float)
                    
                    rhs = np.ones(n)
                    
                    try:
                        # Solve B v = rhs
                        # Since B is 2x r, we have 2 equations, r unknowns.
                        # We want a solution. If r >= 2, we can use pinv.
                        # But we need EXACT solution for the linear system.
                        # If rank(B) < n, no solution or infinite.
                        # If rank(B) == n, we can find a solution.
                        # Using pinv gives minimum norm solution.
                        # We must check if B @ v == rhs exactly (within float tol).
                        
                        v = np.linalg.pinv(B) @ rhs
                        
                        # Check consistency
                        if not np.allclose(B @ v, rhs, atol=1e-9):
                            valid = False
                            break
                            
                        V[j*p+k, :] = v
                        
                    except:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Verify full decomposition
            C = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(r):
                            C[i, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            if np.allclose(C, np.ones((n, p)), atol=1e-9):
                # Check if U, V, W are half-integers
                if is_half_integer_matrix(U) and is_half_integer_matrix(V) and is_half_integer_matrix(W):
                    return (U, V, W)
                else:
                    # Try to snap to nearest half-integer and re-verify
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    C_r = np.zeros((n, p), dtype=float)
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                for c in range(r):
                                    C_r[i, k] += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    
                    if np.allclose(C_r, np.ones((n, p)), atol=1e-8):
                        return (U_r, V_r, W_r)
        
        return None

    # Try ranks from low to high
    # We start lower than before because integer search is more directed
    # but we still need to cover a range.
    # Known lower bounds are higher, but let's try 30-55.
    best_result = None
    
    # It is unlikely to find rank < 40 for 2x4x7 easily.
    # Let's try a wider range but limit attempts per rank.
    for r in range(30, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        result = try_rank_integer_search(r, max_attempts=100)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        rank = U.shape[1]
        # Convert to list format
        U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
