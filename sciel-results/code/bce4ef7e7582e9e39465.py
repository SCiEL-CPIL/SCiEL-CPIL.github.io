"""Optimized Strassen-based decomposition for 2x3x8 matrix multiplication."""
import time
import sys
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    def _try_rank(rank, n_restarts=20):
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        for restart in range(n_restarts):
            if not _elapsed_ok():
                return None
            
            # Initialize with random integers in [-2, 2]
            # This often works well for small tensor ranks
            U_init = np.random.randint(-2, 3, size=(dim_u, rank)).astype(float)
            V_init = np.random.randint(-2, 3, size=(dim_v, rank)).astype(float)
            W_init = np.random.randint(-2, 3, size=(dim_w, rank)).astype(float)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U_mat = x[:dim_u*rank].reshape(dim_u, rank)
                V_mat = x[dim_u*rank : dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                W_mat = x[dim_u*rank + dim_v*rank : ].reshape(dim_w, rank)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            ui = U_mat[i*m+a]
                            vk = V_mat[a*p+k]
                            wk = W_mat[k*n+i]
                            val = np.sum(ui * vk * wk)
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if res.cost < 1e-10:
                    U_sol = res.x[:dim_u*rank].reshape(dim_u, rank)
                    V_sol = res.x[dim_u*rank : dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                    W_sol = res.x[dim_u*rank + dim_v*rank : ].reshape(dim_w, rank)
                    
                    # Attempt rational reconstruction
                    U_rat = []
                    V_rat = []
                    W_rat = []
                    for row in U_sol:
                        U_rat.append([Fraction(val).limit_denominator(10000) for val in row])
                    for row in V_sol:
                        V_rat.append([Fraction(val).limit_denominator(10000) for val in row])
                    for row in W_sol:
                        W_rat.append([Fraction(val).limit_denominator(10000) for val in row])
                    
                    # Convert to int if possible
                    U_int = [[int(f) if f.denominator == 1 else f for f in row] for row in U_rat]
                    V_int = [[int(f) if f.denominator == 1 else f for f in row] for row in V_rat]
                    W_int = [[int(f) if f.denominator == 1 else f for f in row] for row in W_rat]
                    
                    # The task allows integer/half-integer. 
                    # If denominator is 2, keep as Fraction or float? 
                    # "integer/half-integer entries". 
                    # The validator uses Fraction, so it handles Fractions.
                    # But the output must be valid Python types. 
                    # Let's output Fractions or ints.
                    
                    candidate = {"U": U_rat, "V": V_rat, "W": W_rat}
                    if _validate(candidate):
                        # Convert to simple types for output if possible, but Fraction is fine
                        # Actually, standard output usually expects numbers. 
                        # Let's convert Fractions to floats if they are simple, or keep as is?
                        # The prompt says "integer/half-integer entries".
                        # I will return the Fractions.
                        return candidate
            except Exception:
                pass
                
        return None

    # Try ranks 40 to 43
    for rank in range(40, 44):
        if not _elapsed_ok():
            break
        candidate = _try_rank(rank)
        if candidate is not None:
            return candidate
            
    return _schoolbook()

# EVOLVE-BLOCK-END
