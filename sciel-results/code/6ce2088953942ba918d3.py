"""Deterministic rank-reduction for 4x5 * 5x6 matrix multiplication tensor."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the schoolbook decomposition (rank = n*m*p = 120)
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=np.int64)
    V = np.zeros((m * p, rank_school), dtype=np.int64)
    W = np.zeros((p * n, rank_school), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    def verify(U_arr, V_arr, W_arr, r):
        """Verify decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
        if time.time() > deadline - 5:
            return False
        U_r = U_arr.reshape(n, m, r)
        V_r = V_arr.reshape(m, p, r)
        W_r = W_arr.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Verify schoolbook
    if not verify(U, V, W, rank_school):
        # Fallback: return schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = rank_school

    # Strategy: Fix sparse skeleton patterns for U and V, then solve for W
    # The tensor equation is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # For fixed (i,j,k), this is linear in W[k,i,c] if U[i,j,c] and V[j,k,c] are fixed.
    # We can fix U and V to have specific sparse patterns, then solve for W.
    #
    # Key insight: If we fix U[i,j,c] and V[j,k,c] for all (i,j,k,c), then
    # W[k,i,c] appears in exactly one equation (for that specific (i,j,k) where
    # U[i,j,c] != 0 and V[j,k,c] != 0).
    #
    # Better approach: For each column c, choose which (i,j) pairs have U[i,j,c] != 0
    # and which (j,k) pairs have V[j,k,c] != 0. Then W[k,i,c] is determined by
    # the equations where both U and V are non-zero.
    #
    # Let's try: For each column c, pick a single (i_c, j_c) for U and a single
    # (j_c, k_c) for V. Then W[k_c, i_c, c] = 1 / (U[i_c,j_c,c] * V[j_c,k_c,c]).
    # But this is just the schoolbook decomposition!
    #
    # To get rank reduction, we need to allow multiple non-zeros per column.
    # Let's try: For each column c, allow U to have non-zeros for multiple (i,j)
    # and V to have non-zeros for multiple (j,k). Then W must satisfy a system.
    
    np.random.seed(42)
    
    # Try to find lower rank decompositions by fixing sparse patterns
    # and solving the resulting linear system for W
    
    # Approach: For a given rank r, try random sparse patterns for U and V
    # where each column has exactly one non-zero in U (at position (i_c, j_c))
    # and exactly one non-zero in V (at position (j_c, k_c)).
    # Then W[k_c, i_c, c] must equal 1, and all other W entries must be 0
    # to avoid contributing to other (i,j,k) equations.
    # This only works if each (i,j,k) is covered exactly once, which is schoolbook.
    
    # Alternative: Allow U to have multiple non-zeros per column.
    # If U[i,j,c] = 1 for multiple (i,j) pairs, and V[j,k,c] = 1 for multiple
    # (j,k) pairs, then W must satisfy:
    #   sum_{c: U[i,j,c]*V[j,k,c] != 0} W[k,i,c] * U[i,j,c] * V[j,k,c] = 1
    # This is a linear system in W.
    
    # Let's try a structured approach: partition the indices
    # For 4x5x6, we can try to use block structure.
    
    # Try ranks from 119 down to some minimum
    # For each rank, try several random sparse patterns
    
    min_rank_to_try = 100  # Don't go too low; unlikely to find valid decompositions
    
    for target_rank in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try multiple random patterns
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Generate random sparse patterns for U and V
            # Each column c has a "center" (i_c, j_c, k_c)
            # U[i,j,c] is non-zero for (i,j) near (i_c, j_c)
            # V[j,k,c] is non-zero for (j,k) near (j_c, k_c)
            
            U_try = np.zeros((n * m, target_rank), dtype=np.int64)
            V_try = np.zeros((m * p, target_rank), dtype=np.int64)
            
            # For each column, choose a random (i_c, j_c, k_c)
            centers = []
            for c in range(target_rank):
                i_c = np.random.randint(0, n)
                j_c = np.random.randint(0, m)
                k_c = np.random.randint(0, p)
                centers.append((i_c, j_c, k_c))
                
                # U[i_c, j_c, c] = 1
                U_try[i_c * m + j_c, c] = 1
                
                # V[j_c, k_c, c] = 1
                V_try[j_c * p + k_c, c] = 1
                
                # Optionally add nearby non-zeros
                if np.random.random() < 0.3:
                    # Add a nearby (i', j') for U
                    di = np.random.choice([-1, 0, 1])
                    dj = np.random.choice([-1, 0, 1])
                    i2 = (i_c + di) % n
                    j2 = (j_c + dj) % m
                    if U_try[i2 * m + j2, c] == 0:
                        U_try[i2 * m + j2, c] = 1
                
                if np.random.random() < 0.3:
                    # Add a nearby (j', k') for V
                    dj = np.random.choice([-1, 0, 1])
                    dk = np.random.choice([-1, 0, 1])
                    j2 = (j_c + dj) % m
                    k2 = (k_c + dk) % p
                    if V_try[j2 * p + k2, c] == 0:
                        V_try[j2 * p + k2, c] = 1
            
            # Now solve for W: For each (i,j,k), we need:
            # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # This is a linear system in W.
            
            # Build the system: variables are W[k,i,c] for all k,i,c
            # Number of variables: p * n * target_rank
            num_vars = p * n * target_rank
            
            # Number of equations: n * m * p (one for each (i,j,k))
            num_eqs = n * m * p
            
            # Coefficient matrix A and right-hand side b
            # A[eq_idx, var_idx] = U[i,j,c] * V[j,k,c] where eq_idx corresponds to (i,j,k)
            # and var_idx corresponds to W[k,i,c]
            
            # For efficiency, build the system incrementally
            A = np.zeros((num_eqs, num_vars), dtype=np.int64)
            b = np.ones(num_eqs, dtype=np.int64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            coeff = U_try[i * m + j, c] * V_try[j * p + k, c]
                            if coeff != 0:
                                var_idx = k * n * target_rank + i * target_rank + c
                                A[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            # Solve the linear system
            # Use numpy's least squares or exact rational arithmetic
            # For small systems, use exact arithmetic with fractions
            
            if time.time() > deadline - 10:
                break
            
            # Try to solve using numpy first (faster)
            try:
                # Check if the system is consistent
                # Use least squares as a quick check
                W_solution, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is valid
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        continue
                
                # Verify the solution
                W_try = W_solution.reshape(p, n, target_rank)
                
                # Check if W_try is close to integer/half-integer
                # For now, just verify the decomposition
                U_r = U_try.reshape(n, m, target_rank)
                V_r = V_try.reshape(m, p, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_try)
                
                if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
                    # Found a valid decomposition!
                    # Round W to nearest integer or half-integer if possible
                    W_rounded = np.round(W_try * 2) / 2
                    T_rounded = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_rounded)
                    
                    if np.allclose(T_rounded, np.ones((n, m, p)), atol=1e-9):
                        best_U, best_V, best_W = U_try, V_try, W_rounded
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # Use the unrounded solution if it's valid
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = target_rank
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
