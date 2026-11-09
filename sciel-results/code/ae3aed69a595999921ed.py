import time
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    global START_TIME, TIME_LIMIT
    
    # Trivial rank decomposition (baseline)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    def verify_decomposition_numpy(U, V, W, n, m, p):
        """Verify using NumPy broadcasting."""
        try:
            U_reshaped = np.array(U).reshape(n, m, -1)
            V_reshaped = np.array(V).reshape(m, p, -1)
            W_reshaped = np.array(W).reshape(p, n, -1)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            return np.all(result == 1)
        except Exception:
            return False
    
    def solve_over_finite_field(r, n, m, p, prime=3, max_trials=200):
        """
        Try to find a rank-r decomposition by solving linear equations over F_prime.
        Each rank-1 term (u, v, w) contributes u[i,j]*v[j,k]*w[k,i] to the (i,j,k) entry.
        We need sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1 for all i,j,k.
        
        Strategy: Randomly fix U and V, then solve for W using linear algebra over F_prime.
        """
        num_u = n * m
        num_v = m * p
        num_w = p * n
        num_constraints = n * m * p
        
        # Build constraint matrix A where A[constraint_idx, var_idx] * w[var_idx] = b[constraint_idx]
        # For each (i,j,k) constraint and each rank term r:
        #   coefficient = U[i*m+j, r] * V[j*p+k, r]
        #   variable = W[k*n+i, r]
        
        for trial in range(max_trials):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                return None
            
            # Random U, V over F_prime
            U = np.random.randint(0, prime, size=(num_u, r)) % prime
            V = np.random.randint(0, prime, size=(num_v, r)) % prime
            
            # Build system A * w = b over F_prime
            # A is num_constraints x (r * num_w)
            # But we can structure it: for each rank term r, it contributes to constraints
            # where U[i*m+j,r] * V[j*p+k,r] != 0
            
            # Simpler: iterate over constraints and build rows
            A = np.zeros((num_constraints, r * num_w), dtype=int)
            b = np.ones(num_constraints, dtype=int)
            
            constraint_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            coeff = (U[i * m + j, r_idx] * V[j * p + k, r_idx]) % prime
                            if coeff != 0:
                                var_idx = r_idx * num_w + (k * n + i)
                                A[constraint_idx, var_idx] = coeff
                        constraint_idx += 1
            
            # Solve A * w = b over F_prime using Gaussian elimination
            w_solution = solve_linear_system_finite_field(A, b, prime)
            
            if w_solution is None:
                continue
            
            # w_solution has shape (r * num_w,)
            W_flat = w_solution.reshape(r, num_w).T  # Now (num_w, r)
            
            # Convert to lists and verify over integers
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_flat.tolist()
            
            if verify_decomposition_numpy(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def solve_linear_system_finite_field(A, b, prime):
        """
        Solve A*x = b over F_prime using Gaussian elimination.
        Returns solution vector or None if no unique solution.
        """
        A = np.array(A, dtype=int)
        b = np.array(b, dtype=int)
        rows, cols = A.shape
        
        # Augmented matrix
        M = np.hstack([A, b.reshape(-1, 1)]) % prime
        
        # Gaussian elimination
        pivot_row = 0
        for col in range(cols):
            if pivot_row >= rows:
                break
            
            # Find pivot
            pivot = None
            for row in range(pivot_row, rows):
                if M[row, col] % prime != 0:
                    pivot = row
                    break
            
            if pivot is None:
                continue
            
            # Swap rows
            if pivot != pivot_row:
                M[[pivot_row, pivot]] = M[[pivot, pivot_row]]
            
            # Normalize pivot row
            inv_pivot = pow(int(M[pivot_row, col] % prime), prime - 2, prime)
            M[pivot_row] = (M[pivot_row] * inv_pivot) % prime
            
            # Eliminate other rows
            for row in range(rows):
                if row != pivot_row and M[row, col] % prime != 0:
                    factor = int(M[row, col] % prime)
                    M[row] = (M[row] - factor * M[pivot_row]) % prime
            
            pivot_row += 1
        
        # Check for consistency
        for row in range(pivot_row, rows):
            if np.any(M[row, :cols] != 0) and M[row, cols] != 0:
                return None
        
        # Extract solution (free variables set to 0)
        x = np.zeros(cols, dtype=int)
        for row in range(pivot_row):
            # Find which column is the pivot for this row
            for col in range(cols):
                if M[row, col] % prime != 0:
                    x[col] = int(M[row, cols] % prime)
                    break
        
        return x
    
    # Search for lower rank decompositions
    # Focus on ranks just below the schoolbook rank (48)
    
    # Attempt 1: Finite field structured search for r in [47, 48]
    for r in [47, 48]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        result = solve_over_finite_field(r, n, m, p, prime=3, max_trials=500)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If not found, try random search with {-1, 0, 1} for r in [47, 48]
    if best_rank == rank_trivial:
        def try_rank_random(r, n, m, p, max_iterations=10000):
            """Try to find a rank-r decomposition with coefficients in {-1, 0, 1}."""
            np.random.seed(42)
            
            num_u = n * m
            num_v = m * p
            num_w = p * n
            
            for _ in range(max_iterations):
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    return None
                
                U = np.random.choice([-1, 0, 1], size=(num_u, r))
                V = np.random.choice([-1, 0, 1], size=(num_v, r))
                W = np.random.choice([-1, 0, 1], size=(num_w, r))
                
                if verify_decomposition_numpy(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            return None
        
        for r in [47, 48]:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            result = try_rank_random(r, n, m, p, max_iterations=10000)
            if result is not None:
                best_rank = r
                best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
