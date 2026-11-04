import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    T = np.zeros((n*m, m*p, p*n))
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                g = k*n + i
                T[a, b, g] = 1.0
                valid_indices.append((a, b, g))
                
    start_time = time.time()
    timeout = 110.0
    
    # Precompute the system matrix M for the linear system in W
    # M has shape (p*n, n*m*m*p*r)
    # M[g, (a,b,c)] = U[a,c] * V[b,c]
    # We will build this incrementally or just solve per g.
    
    # Strategy:
    # 1. Try small ranks (25-35) with random U, V and solve for W.
    # 2. If found, snap to half-integers and verify.
    # 3. Fallback to schoolbook (rank 36).
    
    for r in range(25, 37):
        if time.time() - start_time > timeout:
            break
        
        # Adjust attempts based on rank and time
        # Lower ranks are harder but more valuable.
        max_attempts = 1000 if r < 30 else (500 if r < 33 else 200)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Random U, V with {-1, 0, 1}
            # Bias towards sparse matrices might help?
            # Let's try a few different distributions.
            # For small r, sparsity is good.
            if attempt % 3 == 0:
                U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.1, 0.8, 0.1])
                V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.1, 0.8, 0.1])
            elif attempt % 3 == 1:
                U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.2, 0.6, 0.2])
                V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.2, 0.6, 0.2])
            else:
                U = np.random.randint(-1, 2, size=(n*m, r))
                V = np.random.randint(-1, 2, size=(m*p, r))
            
            # Solve for W
            W = np.zeros((p*n, r))
            valid = True
            
            for g in range(p*n):
                # Build M_g: shape (n*m * m*p, r)
                # b_g: shape (n*m * m*p)
                
                # Efficient construction
                # M_g row (a, b) is U[a, :] * V[b, :]
                # We can precompute outer products?
                # U has shape (9, r), V has shape (12, r)
                # For each a, b: M_g[a*12+b, :] = U[a, :] * V[b, :]
                
                # This is a bit slow in a loop.
                # Let's vectorize if possible.
                # M_g is a matrix where row index is flattened (a,b) and col is c.
                # M_g[ (a,b), c ] = U[a,c] V[b,c]
                
                # We can compute this as:
                # Let U_rep be U repeated m*p times? No.
                # Let V_rep be V repeated n*m times? No.
                
                # Actually, we can just loop over a, b since 9*12=108 is small.
                M_g = np.zeros((n*m * m*p, r))
                b_g = np.zeros(n*m * m*p)
                
                idx = 0
                for a in range(n*m):
                    Ua = U[a, :]
                    for b in range(m*p):
                        Vb = V[b, :]
                        M_g[idx, :] = Ua * Vb
                        b_g[idx] = T[a, b, g]
                        idx += 1
                
                # Solve M_g @ W[g, :] = b_g
                # Use lstsq
                try:
                    w_g, residuals, rank_g, sv = np.linalg.lstsq(M_g, b_g, rcond=None)
                except Exception:
                    valid = False
                    break
                    
                # Check if solution is exact
                # residual should be ~0
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                else:
                    # If underdetermined, residual is empty?
                    # lstsq returns residuals only if M is full rank or overdetermined.
                    # Here M is 108 x r. r <= 36. So overdetermined.
                    # So residuals should exist.
                    pass
                    
                W[g, :] = w_g
                
            if not valid:
                continue
                
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            # Verify exactness
            # T_approx[a, b, g] = sum_c U_s[a,c] V_s[b,c] W_s[g,c]
            T_approx = np.einsum('ac,bc,gc->abg', U_s, V_s, W_s)
            
            if np.allclose(T_approx, T, atol=1e-9):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    # Fallback
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
