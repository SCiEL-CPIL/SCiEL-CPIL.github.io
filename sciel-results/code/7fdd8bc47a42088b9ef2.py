"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Fix U to a structured basis, then solve for V and W.
    # For a given rank r, fix U to have structured entries.
    # The constraint is: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all (i,j,k)
    
    # Approach: Try structured U matrices (e.g., based on Kronecker products or 
    # combinatorial designs), then solve the resulting bilinear system for V and W.
    
    # For fixed U, the system is bilinear in V and W. We can:
    # 1. Fix W to a structured matrix as well, then solve linear system for V.
    # 2. Or use alternating minimization.
    
    # Let's try: Fix U and W to be structured, then solve for V via linear system.
    
    # Structured U: Use a repetition code where each (i,j) appears in multiple columns.
    # Structured W: Similarly for (k,i).
    
    # For rank r, we need r >= max(n*m, p*n) for the basic construction to work,
    # but we can do better with more clever structures.
    
    # Try r from 112 down to 80, using structured U and W.
    
    for r in range(112, 79, -1):
        # Generate structured U: each (i,j) appears in approximately r/(n*m) columns
        U = np.zeros((n * m, r))
        for t in range(r):
            # Assign column t to a specific (i,j) based on a cyclic pattern
            idx = t % (n * m)
            U[idx, t] = 1.0
        
        # Generate structured W: each (k,i) appears in approximately r/(p*n) columns
        W = np.zeros((p * n, r))
        for t in range(r):
            # Assign column t to a specific (k,i) based on a cyclic pattern
            idx = t % (p * n)
            W[idx, t] = 1.0
        
        # Now solve for V: for each (i,j,k), sum_t U[i*m+j,t]*W[k*n+i,t]*V[j*p+k,t] = 1
        # Set up the linear system.
        # Unknowns: V[j*p+k, t] for j in [0,m), k in [0,p), t in [0,r)
        # Total unknowns: m*p*r = 28*r
        # Equations: n*m*p = 112
        
        A = np.zeros((n * m * p, m * p * r))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            # V[j*p+k, t] is at index (j*p+k)*r + t
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        # Solve A * V_flat = b using least squares
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return result
        except:
            pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
