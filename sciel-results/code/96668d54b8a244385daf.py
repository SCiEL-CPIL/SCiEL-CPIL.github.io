"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Precompute tensor structure
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    # i.e., i=a//m, j=a%m, j2=b//p, k=b%p, k2=c//n, i2=c%n
    # T=1 iff i==i2 and j==j2 and k==k2
    
    def tensor_residual(U, V, W):
        """Compute residual of tensor decomposition."""
        r = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # For each (a,b,c), compute sum_t U[a,t]*V[b,t]*W[c,t]
        # We need this to be 1 for valid entries, 0 otherwise
        # Use einsum for efficiency
        # T_reconstructed[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Build target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        return np.sum((T_rec - T_target) ** 2)
    
    def verify(U, V, W, tol=1e-7):
        """Verify exact decomposition."""
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T_rec[a, b, c] - 1.0) > tol:
                        return False
        # Check some zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1 = a // m; j1 = a % m
                    j2 = b // p; k1 = b % p
                    k2 = c // n; i2 = c % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    if abs(T_rec[a, b, c]) > tol:
                        return False
        return True
    
    # Try ranks from lower to higher
    # Lower bound: max(n*m, m*p, p*n) = max(10,25,10) = 25? No, that's not a valid lower bound for tensor rank
    # Actually for matrix mult, known bounds are complex. Let's try from 25 upward.
    # But we know rank 50 works. Let's try to find if lower is possible.
    
    for r in range(25, 50):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization with structured hints
        best_residual = float('inf')
        found = False
        
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization: use schoolbook pattern but compressed
            # For each (i,j,k), we have a term. Try to merge terms.
            # Initialize U, V, W randomly but with some structure
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Scale to help convergence
            # Normalize columns
            for t in range(r):
                nu = np.linalg.norm(U[:, t])
                nv = np.linalg.norm(V[:, t])
                nw = np.linalg.norm(W[:, t])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    scale = (nu * nv * nw) ** (1/3)
                    U[:, t] /= scale
                    V[:, t] /= scale
                    W[:, t] /= scale
            
            # Use scipy minimize on the residual
            # Flatten parameters
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(x):
                Ux = x[:n*m*r].reshape(n*m, r)
                Vx = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                Wx = x[n*m*r+m*p*r:].reshape(p*n, r)
                return tensor_residual(Ux, Vx, Wx)
            
            try:
                res = minimize(obj, params, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_residual:
                    best_residual = res.fun
                    if best_residual < 1e-8:
                        U_opt = res.x[:n*m*r].reshape(n*m, r)
                        V_opt = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                        W_opt = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                        if verify(U_opt, V_opt, W_opt, tol=1e-7):
                            best = {
                                "U": U_opt.tolist(),
                                "V": V_opt.tolist(),
                                "W": W_opt.tolist()
                            }
                            best_rank = r
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
