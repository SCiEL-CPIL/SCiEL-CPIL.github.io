import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: n*m*p - n - m + 1 = 42 - 2 - 3 + 1 = 38
    target_rank = n * m * p - n - m + 1
    
    # Fallback to schoolbook if search fails
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
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # We try to find a rank 38 solution
    # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
    # Equations: For each i, k, j, j': sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+j', c] == delta(j, j')
    
    # Number of equations: n * p * m * m = 2 * 7 * 3 * 3 = 126
    # Number of unknowns for W: p * n * r = 14 * 38 = 532
    # System is underdetermined, so solutions exist.
    
    # We will use Gaussian elimination over Q (Fractions) to find a solution.
    # To keep it fast, we limit the number of random restarts.
    
    max_restarts = 50
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        r = target_rank
        # Generate random U and V with small integers
        # Using -1, 0, 1 to keep numbers small
        U = np.random.choice([-1, 0, 1], size=(n * m, r))
        V = np.random.choice([-1, 0, 1], size=(m * p, r))
        
        # Build the linear system A * W_flat = b
        # A has shape (n*p*m*m, p*n*r)
        # b has shape (n*p*m*m,)
        
        # To avoid large memory usage, we solve for W row by row?
        # No, W couples all c for a fixed (k,i).
        # Let's flatten W to a vector of size p*n*r.
        
        # Construct A and b
        # Indices:
        # Eq idx: e = ((i * p + k) * m + j) * m + j'
        # Var idx: v = (k * n + i) * r + c
        
        num_eqs = n * p * m * m
        num_vars = p * n * r
        
        # We only need to solve if the system is consistent.
        # Since it's underdetermined, it's always consistent if we pick U,V well?
        # Actually, the map (U,V) -> (A,b) might have a non-zero b component in the null space of A?
        # No, the equation is linear in W. If the rows of A span the space containing b, a solution exists.
        # For random U,V, the rows of A are random. The probability that b is in the column space is 1?
        # Actually, for any U,V, the system A W = b might not have a solution if b is not in Col(A).
        # But we can use Least Squares to find the best W, then check if it's exact.
        
        # Using Fraction for exactness is slow.
        # Let's use floating point first to check feasibility, then refine?
        # Or just use Gaussian elimination with Fractions on a small subset?
        
        # Given the small size (126 eqs, 532 vars), we can use scipy to find a minimum norm solution
        # and then round? No, rounding breaks exactness.
        
        # Let's try to solve using `scipy.linalg.lstsq` and check the residual.
        # If residual is very small, we might have an integer solution?
        # Unlikely to be exact integers.
        
        # Alternative: Use the fact that we can set most of W to 0?
        
        # Let's try a different strategy:
        # Fix U and V to be the first r columns of the identity?
        # If r >= n*m + m*p, we can make U and V sparse identities.
        # n*m = 6, m*p = 21. Sum = 27.
        # If r >= 27, we can set U to have 1s on diagonal for first 6 rows, V for first 21 rows?
        # This doesn't guarantee a solution for W.
        
        # Let's stick to the random search with exact rational arithmetic using a simplified solver.
        # Since 126 equations is small, we can use `scipy.sparse`?
        
        # Actually, there is a known solution for rank 38.
        # I will return the schoolbook solution if I don't find a better one quickly.
        # The prompt asks for a "stronger algorithm".
        # A rank 38 solution is significantly better.
        
        # Let's try to construct the solution using the "Bilinear Complexity" reduction.
        # The reduction from 42 to 38 saves 4 operations.
        # This corresponds to the identity:
        # (A+B)(B+I) = AB + AI + BB + BI ...
        
        # Given the time constraint and complexity of implementing a general rank minimizer,
        # and the fact that the seed is 1.0 (rank 42),
        # I will attempt a simple random search for rank 38.
        
        # Build A matrix for the linear system
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        eq_idx = ((i * p + k) * m + j) * m + jp
                        b[eq_idx] = 1 if j == jp else 0
                        
                        # Variables are W[k*n+i, c]
                        # A[eq_idx, var_idx] = U[i*m+j, c] * V[j*p+jp, c]
                        for c in range(r):
                            var_idx = (k * n + i) * r + c
                            A[eq_idx, var_idx] = U[i * m + j, c] * V[j * p + jp, c]
        
        # Solve A W = b
        # Use least squares
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if the solution is exact
        # Residual should be 0
        if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-9:
            continue
            
        # Check if W_flat entries are close to integers or half-integers
        W_check = W_flat
        # Scale to check for half-integers
        # If W is integer, 2*W is even integer.
        # If W is half-integer, 2*W is odd integer.
        
        valid = True
        for val in W_check:
            if abs(val - round(val)) > 1e-6:
                # Try half-integer
                if abs(2*val - round(2*val)) > 1e-6:
                    valid = False
                    break
        
        if valid:
            # Convert to lists of Fractions/Ints
            U_out = [[int(x) for x in row] for row in U]
            V_out = [[int(x) for x in row] for row in V]
            W_mat = W_check.reshape((p * n, r))
            W_out = []
            for row in W_mat:
                row_out = []
                for val in row:
                    # Determine if it's integer or half-integer
                    if abs(val - round(val)) < 1e-6:
                        row_out.append(int(round(val)))
                    else:
                        # It's a half-integer
                        val2 = 2 * val
                        int_val2 = int(round(val2))
                        # Represent as Fraction(int_val2, 2) or just the float?
                        # The interface expects "integer/half-integer entries".
                        # Usually this means numbers like 1, 0.5, -1.5.
                        # Python floats are fine if they are exact representations of halves.
                        row_out.append(val2 / 2.0)
                W_out.append(row_out)
            
            return {"U": U_out, "V": V_out, "W": W_out}

    # If no solution found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
