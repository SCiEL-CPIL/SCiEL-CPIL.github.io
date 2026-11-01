import time
import numpy as np

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Checks both the 1s (diagonal) and 0s (off-diagonal).
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Precompute U * V for each c? No, just loop.
    # Dimensions: U (nm, r), V (mp, r), W (pn, r)
    
    for i in range(n):
        for k in range(p):
            # Check diagonal (j == jp)
            for j in range(m):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
            
            # Check off-diagonal (j != jp)
            for j in range(m):
                for jp in range(m):
                    if j == jp: continue
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 110  # 10s safety margin
    
    # Schoolbook fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # We want to find r < 60
    # Lower bound r >= 9
    # Try ranks from 9 to 59
    
    rng = np.random.RandomState(42)
    
    for r in range(9, best_rank):
        if time.time() > deadline:
            break
            
        # Max attempts per rank
        # If r is close to 60, solution is more likely?
        # Actually, lower r is harder.
        attempts = 500 if r < 30 else 1000
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            # Use integers -1, 0, 1 to keep numbers small
            # Density of non-zeros matters.
            # If too sparse, rank of M might be low.
            # If too dense, M is full rank.
            
            U = rng.randint(-1, 2, size=(n*m, r))
            V = rng.randint(-1, 2, size=(m*p, r))
            
            # Build Linear System M * w = b
            # w is length p*n*r
            # M is 180 x (p*n*r)
            
            num_eqs = n * p * (m + m*(m-1))
            M = np.zeros((num_eqs, p*n*r))
            b = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    # Diagonal
                    for j in range(m):
                        for c in range(r):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            if coeff != 0:
                                M[eq_idx, c*p*n + k*n + i] += coeff
                        b[eq_idx] = 1
                        eq_idx += 1
                    # Off-diagonal
                    for j in range(m):
                        for jp in range(m):
                            if j == jp: continue
                            for c in range(r):
                                coeff = U[i*m+j, c] * V[jp*p+k, c]
                                if coeff != 0:
                                    M[eq_idx, c*p*n + k*n + i] += coeff
                            b[eq_idx] = 0
                            eq_idx += 1
                            
            # Solve
            try:
                w, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                # Check if solution is exact
                # lstsq returns min residual.
                if residuals is not None and len(residuals) > 0:
                    res = residuals[0]
                else:
                    res = np.linalg.norm(M @ w - b)
                
                if res < 1e-9:
                    # Reconstruct W
                    W = np.zeros((p*n, r))
                    for c in range(r):
                        W[:, c] = w[c*p*n : (c+1)*p*n]
                    
                    # Verify
                    if _verify_full(U, V, W, n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except:
                pass
                
    return best
