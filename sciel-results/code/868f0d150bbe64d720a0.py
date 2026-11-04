"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    fallback = schoolbook()
    
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (j == jp and k == kp) else 0.0
                            actual = np.dot(W[k*n+i, :], U[i*m+j, :] * V[jp*p+kp, :])
                            if abs(actual - expected) > 1e-8:
                                return False
        return True
    
    lower = max(n*p, n*m, m*p)
    
    for rank in range(lower, n*m*p + 1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 1000 if rank <= 40 else 300
        for att in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(att * 7919 + rank * 104729 + 42)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            
            W = np.zeros((p*n, rank))
            ok = True
            
            for i in range(n):
                # Build the full linear system for this i
                # Unknowns: W[k*n+i, c] for k=0..p-1, c=0..rank-1
                # Equations: for (j, jp, k, kp): sum_c W[k*n+i,c] * U[i*m+j,c] * V[jp*p+kp,c] = δ_{j,jp}δ_{k,kp}
                
                A = np.zeros((p*m*m, p*rank))
                rhs = np.zeros(p*m*m)
                
                eq_idx = 0
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                # Coefficient of W[kp*n+i, c] is U[i*m+j,c]*V[jp*p+kp,c] if kp==k else 0
                                # Wait, the equation is indexed by (j, jp, k, kp) and the unknown is W[kp*n+i, c]
                                # The coefficient is U[i*m+j,c] * V[jp*p+kp,c] if the output k matches kp
                                # Actually, the equation for output (i,k) and input pair (j, (jp,kp)) is:
                                # sum_c W[k*n+i,c] * U[i*m+j,c] * V[jp*p+kp,c] = δ_{j,jp}δ_{k,kp}
                                # So the unknown is W[k*n+i,c] (fixed k for this equation)
                                # And the coefficient of W[kp*n+i,c] is 0 unless kp==k
                                # So for equation (j,jp,k,kp), only W[k*n+i,c] appears
                                # Coefficient: U[i*m+j,c] * V[jp*p+kp,c]
                                
                                for c in range(rank):
                                    A[eq_idx, kp*rank + c] = U[i*m+j, c] * V[jp*p+kp, c]
                                rhs[eq_idx] = 1.0 if (j == jp and k == kp) else 0.0
                                eq_idx += 1
                
                # Solve A @ W_flat = rhs
                try:
                    sol, res, rk, sv = np.linalg.lstsq(A, rhs, rcond=1e-12)
                    if np.allclose(A @ sol, rhs, atol=1e-9):
                        W_flat = sol.reshape(p, rank)
                        for k in range(p):
                            W[k*n+i, :] = W_flat[k, :]
                    else:
                        ok = False
                        break
                except:
                    ok = False
                    break
            
            if ok and verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return fallback
# EVOLVE-BLOCK-END
