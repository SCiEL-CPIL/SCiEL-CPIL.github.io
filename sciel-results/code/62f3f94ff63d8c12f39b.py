"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    # Target rank search from 34 downwards
    # Fallback to 36 (schoolbook) if not found
    
    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9

    def check_solution(U, V, W, R):
        # Verify sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W, rank

    # Try to find lower rank solutions using random search + linear solve for W
    # We fix U and V to random integer values, then solve for W.
    # Equation: sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1
    # This is a linear system A * W_flat = b
    # Where A is (n*m*p) x (p*n*R) ? No.
    # Variables are W[k,i,c]. Total p*n*R variables.
    # Equations are for each (i,j,k). Total n*m*p equations.
    
    for R in [34, 35]:
        for attempt in range(200):
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Build linear system for W
            # Variables: W_flat of size (p*n*R)
            # Index in W_flat: (k*n + i) * R + c
            # Equation for (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            A = np.zeros((n * m * p, p * n * R))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                var_idx = (k * n + i) * R + c
                                A[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            try:
                # Solve A W_flat = b
                # Use least squares if underdetermined
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape((p * n, R))
                
                # Check if W has half-integer entries
                if all(is_half_integer(w) for w in W):
                    # Snap to half-integers
                    W = np.round(W * 2) / 2.0
                    # Verify
                    if check_solution(U, V, W, R):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except Exception:
                continue

    # Fallback to Schoolbook
    U, V, W, rank = generate_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
