"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    total_eqs = n * m * p  # 112
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(r):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
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
        return {"U": U, "V": W, "W": W}
    
    # Strategy: Fix U and W to structured bases, solve for V via linear system.
    # We try various structured patterns for U and W, and for each, solve
    # the resulting linear system for V. If the solution is exact, we're done.
    
    # Try different ranks from low to high
    for r in range(56, 112):
        # Try multiple structured patterns for U and W
        
        # Pattern 1: Cyclic - U[i*m+j, t] = 1 if t % (n*m) == i*m+j
        #             W[k*n+i, t] = 1 if t % (p*n) == k*n+i
        U = np.zeros((n * m, r))
        W = np.zeros((p * n, r))
        for t in range(r):
            u_idx = t % (n * m)
            w_idx = t % (p * n)
            U[u_idx, t] = 1.0
            W[w_idx, t] = 1.0
        
        # Build linear system for V
        # Unknowns: V[j*p+k, t] for j in [0,m), k in [0,p), t in [0,r)
        # Total unknowns: m*p*r
        # Equations: n*m*p = 112
        # For equation (i,j,k): sum_t U[i*m+j,t]*W[k*n+i,t]*V[j*p+k,t] = 1
        
        num_unknowns = m * p * r
        A = np.zeros((n * m * p, num_unknowns))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
            V = V_flat.reshape(m * p, r)
            
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
        
        # Pattern 2: Block-cyclic - group columns into blocks
        # U[i*m+j, t] = 1 if (t // block_size) % (n*m) == i*m+j
        # W[k*n+i, t] = 1 if (t // block_size) % (p*n) == k*n+i
        block_size = max(1, r // max(1, min(n*m, p*n)))
        U = np.zeros((n * m, r))
        W = np.zeros((p * n, r))
        for t in range(r):
            block = t // block_size
            u_idx = block % (n * m)
            w_idx = block % (p * n)
            U[u_idx, t] = 1.0
            W[w_idx, t] = 1.0
        
        num_unknowns = m * p * r
        A = np.zeros((n * m * p, num_unknowns))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
            V = V_flat.reshape(m * p, r)
            
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
        
        # Pattern 3: Repetition - each (i,j) appears in multiple columns
        # U[i*m+j, t] = 1 if t < n*m and t == i*m+j, or t >= n*m and (t-n*m) % (n*m) == i*m+j
        U = np.zeros((n * m, r))
        W = np.zeros((p * n, r))
        for t in range(r):
            if t < n * m:
                U[t, t] = 1.0
            else:
                U[(t - n * m) % (n * m), t] = 1.0
            
            if t < p * n:
                W[t, t] = 1.0
            else:
                W[(t - p * n) % (p * n), t] = 1.0
        
        num_unknowns = m * p * r
        A = np.zeros((n * m * p, num_unknowns))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
            V = V_flat.reshape(m * p, r)
            
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
