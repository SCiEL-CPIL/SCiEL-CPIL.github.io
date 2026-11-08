"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to find low-rank decomposition
    # For M[2,3,9], try ranks from 27 to 54
    
    def solve_rank(r, seed=42):
        """Find rank-r decomposition using alternating least squares."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # Target: T[i*m+j, j*p+k, k*n+i] = 1
        for iteration in range(200):
            if time.time() > deadline - 10:
                return None
            
            # Update U: for each row a=i*m+j, solve linear system
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all b,c
            # Let M_U be (m*p*p*n) x r where M_U[(b,c),r] = V[b,r]*W[c,r]
            # Then U[a,:] = lstsq(M_U, T[a,:,:].flatten())
            
            M_U = np.zeros((m * p * p * n, r))
            for ri in range(r):
                M_U[:, ri] = np.outer(V[:, ri], W[:, ri]).flatten()
            
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    target = np.zeros(m * p * p * n)
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        target[b * p * n + c] = 1.0
                    U[a, :], _, _, _ = np.linalg.lstsq(M_U, target, rcond=None)
            
            # Update V
            M_V = np.zeros((n * m * p * n, r))
            for ri in range(r):
                M_V[:, ri] = np.outer(U[:, ri], W[:, ri]).flatten()
            
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    target = np.zeros(n * m * p * n)
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        target[a * p * n + c] = 1.0
                    V[b, :], _, _, _ = np.linalg.lstsq(M_V, target, rcond=None)
            
            # Update W
            M_W = np.zeros((n * m * m * p, r))
            for ri in range(r):
                M_W[:, ri] = np.outer(U[:, ri], V[:, ri]).flatten()
            
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    target = np.zeros(n * m * m * p)
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        target[a * m * p + b] = 1.0
                    W[c, :], _, _, _ = np.linalg.lstsq(M_W, target, rcond=None)
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            
            if err < 1e-8:
                return U, V, W
        
        return None
    
    # Try ranks
    for r in range(27, 55):
        if time.time() > deadline - 15:
            break
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            result = solve_rank(r, seed)
            if result is not None:
                U, V, W = result
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                            if abs(val - 1.0) > 1e-6:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    # Return unrounded
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
