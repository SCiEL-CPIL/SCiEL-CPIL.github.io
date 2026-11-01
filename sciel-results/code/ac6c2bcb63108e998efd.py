import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt rank-6 decomposition via homotopy continuation,
    fall back to verified Strassen rank-7 solution.
    """
    _start = time.time()
    _timeout = 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # First: try rank-6 search with homotopy continuation
    rank6_solution = _try_rank6_homotopy(_timeout - (time.time() - _start))
    if rank6_solution is not None:
        return rank6_solution
    
    # Fallback: verified Strassen rank-7
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen products:
    # r=0: M1 = (a+d)(e+h)
    # r=1: M2 = (c+d)e
    # r=2: M3 = a(f-h)
    # r=3: M4 = d(g-e)
    # r=4: M5 = (a+b)h
    # r=5: M6 = (c-a)(e+f)
    # r=6: M7 = (b-d)(g+h)
    
    # U[i*m+j, r]: a=U[0], b=U[1], c=U[2], d=U[3]
    U[0,0]=1; U[3,0]=1    # M1: a+d
    U[2,1]=1; U[3,1]=1    # M2: c+d
    U[0,2]=1              # M3: a
    U[3,3]=1              # M4: d
    U[0,4]=1; U[1,4]=1    # M5: a+b
    U[2,5]=1; U[0,5]=-1   # M6: c-a
    U[1,6]=1; U[3,6]=-1   # M7: b-d
    
    # V[j*p+k, r]: e=V[0], f=V[1], g=V[2], h=V[3]
    V[0,0]=1; V[3,0]=1    # M1: e+h
    V[0,1]=1              # M2: e
    V[1,2]=1; V[3,2]=-1   # M3: f-h
    V[2,3]=1; V[0,3]=-1   # M4: g-e
    V[3,4]=1              # M5: h
    V[0,5]=1; V[1,5]=1    # M6: e+f
    V[2,6]=1; V[3,6]=1    # M7: g+h
    
    # W[k*n+i, r]: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    # C[0,0] = M1 + M4 - M5 + M7
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    # C[1,0] = M2 + M4
    W[1,1]=1; W[1,3]=1
    # C[0,1] = M3 + M5
    W[2,2]=1; W[2,4]=1
    # C[1,1] = M1 - M2 + M3 + M6
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank6_homotopy(budget):
    """
    Attempt to find a rank-6 decomposition using homotopy continuation.
    Returns None if no solution found or time runs out.
    """
    if budget < 2.0:
        return None
    
    start = time.time()
    n, m, p = 2, 2, 2
    
    # The tensor T[i,j,k] = delta for matrix multiplication
    # We want: sum_{r=1}^{6} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # for all (i,j,k) in {0,1}^3
    
    # T values: T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1, others=0
    # In flattened index (i,j,k):
    # (0,0,0) -> T=1
    # (0,1,1) -> T=1  (i=0,j=1,k=1)
    # (1,0,0) -> T=1  (i=1,j=0,k=0)
    # (1,1,1) -> T=1  (i=1,j=1,k=1)
    
    # For rank 6, we have 6*4*3 = 72 unknowns and 8 equations.
    # This is underdetermined, so solutions likely exist in real space.
    # The constraint is that entries should be "nice" (integer/half-integer).
    
    # Strategy: Use a smooth optimization approach to minimize:
    #   sum_{(i,j,k)} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - T[i,j,k])^2
    # subject to normalization constraints.
    
    # We'll use scipy's least_squares if available, otherwise a custom gradient descent.
    try:
        from scipy.optimize import least_squares
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    rank = 6
    
    # Parameterize: flatten U, V, W into one vector
    # Total params: 4*6 + 4*6 + 4*6 = 72
    
    def residuals(x):
        U = x[:24].reshape(4, rank)
        V = x[24:48].reshape(4, rank)
        W = x[48:72].reshape(4, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    target = 1.0 if (i==0 and j==0 and k==0) or \
                               (i==0 and j==1 and k==1) or \
                               (i==1 and j==0 and k==0) or \
                               (i==1 and j==1 and k==1) else 0.0
                    res.append(val - target)
        return np.array(res)
    
    # Try multiple random initializations
    best_sol = None
    best_res = np.inf
    
    np.random.seed(42)
    max_iters = min(50, int(budget / 0.5))  # adaptive based on budget
    
    for attempt in range(max_iters):
        if time.time() - start > budget - 2.0:
            break
        
        # Random initialization with some structure
        x0 = np.random.randn(72) * 0.5
        # Add some structured components
        for r in range(rank):
            x0[r] = 1.0  # U[0,r]
            x0[18+r] = 1.0  # V[0,r]
        
        if use_scipy:
            try:
                result = least_squares(residuals, x0, method='lm', 
                                       max_nfev=200, ftol=1e-10, xtol=1e-10)
                if result.cost < best_res:
                    best_res = result.cost
                    best_sol = result.x
            except Exception:
                pass
        else:
            # Simple gradient descent
            x = x0.copy()
            lr = 0.01
            for step in range(100):
                if time.time() - start > budget - 2.0:
                    break
                r = residuals(x)
                cost = np.sum(r**2)
                if cost < best_res:
                    best_res = cost
                    best_sol = x.copy()
                # Numerical gradient
                grad = np.zeros(72)
                eps = 1e-7
                for idx in range(0, 72, 4):  # subsample for speed
                    xp = x.copy()
                    xp[idx] += eps
                    grad[idx] = (np.sum(residuals(xp)**2) - cost) / eps
                x -= lr * grad
                lr *= 0.99
    
    # Check if we found a valid solution
    if best_sol is not None and best_res < 1e-8:
        U = best_sol[:24].reshape(4, rank)
        V = best_sol[24:48].reshape(4, rank)
        W = best_sol[48:72].reshape(4, rank)
        
        # Verify exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    target = 1.0 if (i==0 and j==0 and k==0) or \
                               (i==0 and j==1 and k==1) or \
                               (i==1 and j==0 and k==0) or \
                               (i==1 and j==1 and k==1) else 0.0
                    if abs(val - target) > 1e-6:
                        return None
        
        # Round to nearest half-integer if close
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Final verification after rounding
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    target = 1.0 if (i==0 and j==0 and k==0) or \
                               (i==0 and j==1 and k==1) or \
                               (i==1 and j==0 and k==0) or \
                               (i==1 and j==1 and k==1) else 0.0
                    if abs(val - target) > 1e-6:
                        return None
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
# EVOLVE-BLOCK-END
