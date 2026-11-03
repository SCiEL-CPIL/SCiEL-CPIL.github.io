"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    
    Strategy:
    1. Use a structured search: Fix random sparse patterns for U and V.
    2. Solve the linear system for W.
    3. Check if W has clean (integer/half-integer) values and if the decomposition is exact.
    4. Iterate with different patterns and ranks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Helper to check if a value is a clean half-integer
    def is_clean_half_int(val, tol=1e-9):
        # val is float
        # Check if val is close to k/2 for some integer k
        # 2*val should be close to an integer
        val2 = val * 2.0
        nearest_int = round(val2)
        if abs(val2 - nearest_int) < tol:
            return True
        return False

    def check_decomposition(U, V, W, rank):
        """
        Check if U, V, W form a valid decomposition.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
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

    # Try ranks from 24 up to 35
    # Lower bound for 3x3x3 is 23. 3x3x4 should be close.
    for target_rank in range(24, 36):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        # Number of attempts per rank
        # We can afford more attempts for lower ranks if they fail quickly
        attempts = 100 if target_rank < 28 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 10.0:
                break
            
            # Generate random sparse U and V
            # Entries in {-1, 0, 1}
            # Density control: start sparse, maybe increase density if failing?
            # For now, let's try a moderate sparsity.
            # U shape: 9 x r
            # V shape: 12 x r
            
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            
            # Randomly assign non-zeros
            # Each column c in U and V should have some structure.
            # Let's try: For each column c, pick a random subset of rows for U and V to be non-zero.
            # Or simply: Each entry is non-zero with probability p_nonzero.
            
            p_nonzero_U = 0.4
            p_nonzero_V = 0.4
            
            for c in range(target_rank):
                for i in range(n * m):
                    if random.random() < p_nonzero_U:
                        U[i][c] = random.choice([-1.0, 1.0])
                for j in range(m * p):
                    if random.random() < p_nonzero_V:
                        V[j][c] = random.choice([-1.0, 1.0])
            
            # Construct linear system for W
            # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Let M[i*k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # Then sum_c M[i*k, c] * W[k*n+i, c] = 1
            # Note: The index for W is k*n+i.
            # The system is:
            # For each (i, k) pair (output index), we have an equation.
            # But W[k*n+i, c] is the variable.
            # The coefficient of W[k*n+i, c] in the equation for (i, k) is M[i*k, c].
            # Wait, the equation for output (i, k) is:
            # sum_c ( sum_j U[i*m+j, c] V[j*p+k, c] ) * W[k*n+i, c] = 1
            # Let A[i*k, c] = sum_j U[i*m+j, c] V[j*p+k, c]
            # Then sum_c A[i*k, c] * W[k*n+i, c] = 1
            # This is a system of n*p = 12 equations.
            # The unknowns are W[k*n+i, c] for all k, i, c.
            # Total unknowns: p * n * target_rank = 12 * target_rank.
            # Total equations: 12.
            # This is under-determined! We have 12 equations and 12*r unknowns.
            # We need to find A solution W such that it is "clean".
            # Since it's under-determined, we can pick a minimum norm solution or a sparse solution.
            # However, the "clean" constraint is non-convex.
            
            # Alternative approach:
            # If we fix U, V, and W for all but one column, we can solve for the last one?
            # No, that's what the linear system does.
            
            # Let's use a different strategy:
            # Assume W is also sparse/random, and check if the residual is 0?
            # No, that's the original random search.
            
            # Better:
            # We can solve for W using least squares to get a candidate, then check if it's clean.
            # If not, try a different random U, V.
            
            # Build A matrix: shape (12, target_rank) is not quite right.
            # The variable is W_flat of size 12 * target_rank.
            # Let's define the matrix B of size (12, 12 * target_rank).
            # Row index: idx = i * p + k (0 to 11)
            # Col index: w_idx = (k * n + i) * target_rank + c ? No.
            # W is indexed by [k*n+i, c].
            # Let's flatten W into a vector w_vec of length 12 * target_rank.
            # w_vec[ (k*n+i)*target_rank + c ] = W[k*n+i][c]
            
            # The equation for (i, k) is:
            # sum_c A[i*k, c] * W[k*n+i, c] = 1
            # This couples variables from different rows of W (indexed by k*n+i) but same column c?
            # No, for a fixed (i,k), the term W[k*n+i, c] appears.
            # So for equation (i,k), the variables involved are W[k*n+i, 0], ..., W[k*n+i, r-1].
            # And the coefficients are A[i*k, 0], ..., A[i*k, r-1].
            # Wait, this means each equation (i,k) only involves the row of W corresponding to (k,i).
            # Let's re-read the tensor definition.
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # We need T[i,j,k] = 1 for all i,j,k.
            # Fix i, j, k.
            # Sum over c.
            # This is NOT separable into independent equations for each (i,k) row of W.
            # Because U[i,j,c] and V[j,k,c] depend on j.
            # For a fixed (i,k), the equation is:
            # sum_c ( sum_j U[i,j,c] V[j,k,c] ) W[k,i,c] = 1
            # Let M[i,k,c] = sum_j U[i,j,c] V[j,k,c].
            # Then sum_c M[i,k,c] W[k,i,c] = 1.
            # This IS separable!
            # For each pair (i, k), we have a linear equation:
            # M[i,k,0]*W[k,i,0] + ... + M[i,k,r-1]*W[k,i,r-1] = 1.
            # There are n*p = 12 such equations.
            # Each equation involves r variables (the row of W for that specific (k,i)).
            # So we have 12 independent linear equations, each in r variables.
            # This is extremely under-determined.
            
            # We can choose W[k,i,c] to be 0 for most c, and solve for one?
            # If we set W[k,i,c] = 0 for c != c0, then M[i,k,c0] * W[k,i,c0] = 1 => W[k,i,c0] = 1/M[i,k,c0].
            # We need W[k,i,c0] to be a clean half-integer.
            # So we need 1/M[i,k,c0] to be a clean half-integer.
            # This implies M[i,k,c0] must be a reciprocal of a half-integer (e.g., 1, 2, 1/2, etc.).
            
            # Strategy:
            # 1. Generate random U, V.
            # 2. Compute M[i,k,c] for all i,k,c.
            # 3. For each (i,k), find a column c such that 1/M[i,k,c] is a clean half-integer.
            #    If found, set W[k,i,c] = 1/M[i,k,c] and W[k,i,other] = 0.
            #    If multiple (i,k) choose the same c, it's fine, W is just a matrix.
            #    Wait, W[k,i,c] is a single entry.
            #    The constraint is that for each (i,k), we pick ONE c (or a combination) such that the sum is 1.
            #    If we force sparsity (1 non-zero per row of W), we just need to find c such that M[i,k,c] is "nice".
            
            valid_W = True
            W_candidate = [[0.0] * target_rank for _ in range(p * n)]
            
            for i in range(n):
                for k in range(p):
                    found_c = False
                    for c in range(target_rank):
                        m_val = 0.0
                        for j in range(m):
                            m_val += U[i * m + j][c] * V[j * p + k][c]
                        if abs(m_val) < 1e-9:
                            continue
                        # Check if 1/m_val is clean half-integer
                        w_val = 1.0 / m_val
                        if is_clean_half_int(w_val):
                            W_candidate[k * n + i][c] = w_val
                            found_c = True
                            break
                    if not found_c:
                        valid_W = False
                        break
                if not valid_W:
                    break
            
            if valid_W:
                # Final check
                if check_decomposition(U, V, W_candidate, target_rank):
                    # Convert to list of ints/half-ints for output
                    # The problem asks for integer/half-integer entries.
                    # We can return floats if they are exactly representable, or convert to Fraction?
                    # The interface says "integer/half-integer entries". Usually floats 0.5, 1.0 are accepted.
                    best_result = {"U": U, "V": V, "W": W_candidate}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
