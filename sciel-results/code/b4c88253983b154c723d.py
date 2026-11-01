import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    # T[a, b, c] = 1 if a%m == b//p and c == (b%p)*n + a//m, else 0
    # Indices:
    # a in [0, nm) -> i = a//m, j = a%m
    # b in [0, mp) -> j = b//p, k = b%p
    # c in [0, pn) -> k = c//n, i = c%n
    # Condition: a%m == b//p (j==j) and c%n == a//m (i==i) and b%p == c//n (k==k)
    # So T[a,b,c] = 1 if (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Try ranks from 17 down to 9
    # Schoolbook is 18. We want to find r < 18.
    # Theoretical lower bound is max(nm, mp, pn) = 9.
    
    for rank in range(17, 8, -1):
        if time.time() > deadline - 15:
            break
            
        best_sol = None
        best_cost = np.inf
        
        # Multiple random restarts
        # Number of restarts depends on time available
        # Let's say max 10 restarts per rank if time allows
        max_restarts = 10
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + rank * 100 + trial)
            # Scale initialization to avoid numerical issues
            U0 = np.random.randn(nm, rank) * 0.5
            V0 = np.random.randn(mp, rank) * 0.5
            W0 = np.random.randn(pn, rank) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def residual(x):
                U = x[:nm*rank].reshape(nm, rank)
                V = x[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
                W = x[nm*rank + mp*rank:].reshape(pn, rank)
                
                # Compute sum_s U[a,s]*V[b,s]*W[c,s]
                # Efficiently: 
                # R = np.einsum('as,bs,cs->abc', U, V, W)
                # But for speed in optimization, we can pre-compute something?
                # No, just use einsum. It's fast for small arrays.
                R = np.einsum('as,bs,cs->abc', U, V, W)
                return (R - T).flatten()
            
            try:
                # Use 'trf' method which is robust for bounded/linear constraints, but we don't have bounds.
                # 'lm' is for linear least squares, not suitable here.
                # 'trf' is good.
                sol = least_squares(residual, x0, max_nfev=200, method='trf', ftol=1e-9, xtol=1e-9, gtol=1e-9)
                cost = np.sum(sol.fun**2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol.x
                    if cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:nm*rank].reshape(nm, rank)
            V = best_sol[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
            W = best_sol[nm*rank + mp*rank:].reshape(pn, rank)
            
            # Convert to list of lists with integer/half-integer entries
            # The problem states "integer/half-integer entries".
            # This implies the solution might have rational coefficients.
            # We round to the nearest 0.5.
            U_out = [[round(x*2)/2 for x in row] for row in U]
            V_out = [[round(x*2)/2 for x in row] for row in V]
            W_out = [[round(x*2)/2 for x in row] for row in W]
            
            # Verify exactness
            U_np = np.array(U_out, dtype=float)
            V_np = np.array(V_out, dtype=float)
            W_np = np.array(W_out, dtype=float)
            R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
            
            if np.allclose(R, T, atol=1e-9):
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: Schoolbook decomposition of rank n*m*p
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
