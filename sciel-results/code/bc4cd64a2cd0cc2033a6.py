import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p - n - m + 1  # 38
    
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
    deadline = start_time + 140

    max_restarts = 5000 # Can afford more since lstsq is fast
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        r = target_rank
        
        # Generate sparse U and V
        # Density 0.2 to 0.5
        density = 0.2 + 0.3 * (restart % 10) / 10.0
        
        U = np.zeros((n * m, r), dtype=int)
        V = np.zeros((m * p, r), dtype=int)
        
        # Random sparse
        for i in range(n * m):
            for c in range(r):
                if random.random() < density:
                    U[i, c] = random.choice([1, -1])
        
        for j in range(m * p):
            for c in range(r):
                if random.random() < density:
                    V[j, c] = random.choice([1, -1])
        
        # Build A and b
        # A: (n*p*m*m, p*n*r)
        # b: (n*p*m*m,)
        
        num_eqs = n * p * m * m
        num_vars = p * n * r
        
        A = np.zeros((num_eqs, num_vars), dtype=float)
        b = np.zeros(num_eqs, dtype=float)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        eq_idx = ((i * p + k) * m + j) * m + jp
                        b[eq_idx] = 1.0 if j == jp else 0.0
                        
                        base_var = (k * n + i) * r
                        u_row = U[i * m + j]
                        v_row = V[j * p + jp]
                        
                        # A[eq_idx, base_var:base_var+r] = u_row * v_row
                        A[eq_idx, base_var:base_var+r] = u_row * v_row
        
        # Solve
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        # Check residuals
        if len(residuals) > 0 and residuals[0] > 1e-9:
            continue
            
        # Check if W_flat is close to int or half-int
        # Scale by 2 to check for integers
        W_scaled = 2 * W_flat
        W_rounded = np.round(W_scaled)
        
        # Check distance
        if np.any(np.abs(W_scaled - W_rounded) > 1e-6):
            continue
            
        # Convert to integers (which are 2*W)
        W_int = W_rounded.astype(int)
        
        # Verify exact solution
        # W = W_int / 2
        # Check: sum_c (W_int[c] / 2) * U[i,j,c] * V[j,k,c] == delta
        # Equivalent to: sum_c W_int[c] * U[i,j,c] * V[j,k,c] == 2 * delta
        
        valid = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        target = 2 if j == jp else 0
                        val = 0
                        base_var = (k * n + i) * r
                        # Sum over c
                        # W_int[base_var+c] * U[i*m+j, c] * V[j*p+jp, c]
                        # Vectorized
                        u_row = U[i * m + j]
                        v_row = V[j * p + jp]
                        w_row = W_int[base_var:base_var+r]
                        
                        # Element-wise product and sum
                        val = np.sum(w_row * u_row * v_row)
                        
                        if val != target:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            # Construct output
            U_out = U.tolist()
            V_out = V.tolist()
            
            # W is W_int / 2
            W_mat = W_int.reshape((p * n, r))
            W_out = []
            for row in W_mat:
                row_out = []
                for val in row:
                    # val is integer representing 2*W
                    # W = val / 2
                    # If val is even, W is integer. If odd, W is half-integer.
                    # Interface accepts integer/half-integer.
                    # We can return val / 2.0
                    row_out.append(val / 2.0)
                W_out.append(row_out)
                
            return {"U": U_out, "V": V_out, "W": W_out}

    return schoolbook()
# EVOLVE-BLOCK-END
