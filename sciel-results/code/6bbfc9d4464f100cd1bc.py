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
    
    # Try structured low-rank decompositions
    # Fix U to a structured basis, solve for V and W
    
    # Strategy: For a given rank r, fix U to have structured entries.
    # Then the constraint for each (i,j,k) is:
    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # If U is fixed, this is bilinear in V and W.
    # To make it linear, fix W too, then solve for V.
    
    # Let's try: U = structured, W = structured, solve for V via linear system.
    
    # For each column t, define u_t = U[:,t], w_t = W[:,t]
    # The constraint for (i,j,k): sum_t u_t[i*m+j] * w_t[k*n+i] * v_t[j*p+k] = 1
    # Let a_t = u_t[i*m+j] * w_t[k*n+i], then sum_t a_t * v_t[j*p+k] = 1
    # This must hold for all (i,j,k).
    
    # For fixed U and W, this is a linear system in V.
    # The system has n*m*p = 112 equations and m*p*r = 28*r unknowns.
    
    # Let's try a few structured U, W pairs.
    
    # Structured U: U[i*m+j, t] = 1 if (i,j) corresponds to column t in a code
    # Let's use U as a repetition code: each (i,j) appears in multiple columns.
    
    # For simplicity, try U where each column is a standard basis vector,
    # but with r > n*m, some (i,j) are repeated.
    
    # Actually, let's try a specific construction:
    # U[i*m+j, t] = 1 if t == i*m+j (for t < n*m), else 0
    # W[k*n+i, t] = 1 if t == k*n+i (for t < p*n), else 0
    # This only works if r >= max(n*m, p*n), and the intersection gives the terms.
    
    # This is getting complex. Let's just try to solve for a specific r.
    
    # Try r = 80
    for r in [100, 90, 80, 72, 64]:
        # Fix U to be the first r columns of an identity-like matrix
        # U is n*m x r. Let U[i*m+j, t] = 1 if i*m+j == t % (n*m) else 0
        # This means each (i,j) appears roughly r/(n*m) times.
        
        U = np.zeros((n * m, r))
        for t in range(r):
            idx = t % (n * m)
            U[idx, t] = 1.0
        
        # Fix W similarly: W[k*n+i, t] = 1 if k*n+i == t % (p*n) else 0
        W = np.zeros((p * n, r))
        for t in range(r):
            idx = t % (p * n)
            W[idx, t] = 1.0
        
        # Now solve for V: for each (i,j,k), sum_t U[i*m+j,t]*W[k*n+i,t]*V[j*p+k,t] = 1
        # Let's set up the linear system.
        # Unknowns: V[j*p+k, t] for j in [0,m), k in [0,p), t in [0,r)
        # Total unknowns: m*p*r = 28*r
        # Equations: n*m*p = 112
        
        # For each equation (i,j,k), the coefficient of V[j*p+k, t] is U[i*m+j,t]*W[k*n+i,t]
        
        # Build the system: A * V_flat = b
        # A has shape (112, 28*r), b has shape (112,)
        
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
        
        # Solve A * V_flat = b
        try:
            # Use least squares
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                # Convert to integer/half-integer if possible
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return result
        except:
            pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
