import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start = time.time()
    deadline = start + 148.0
    
    # Schoolbook fallback
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = schoolbook()
    best_rank = n*m*p
    
    # Precompute index lists
    idx_U = [i*m + j for i in range(n) for j in range(m)]
    idx_V = [j*p + k for j in range(m) for k in range(p)]
    idx_W = [k*n + i for k in range(p) for i in range(n)]
    idx_T = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U, V, W, tol=1e-8):
        for (i, j, k) in idx_T:
            s = 0.0
            for c in range(U.shape[1]):
                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            if abs(s - 1.0) > tol:
                return False
        return True
    
    def snap_to_half_int(x, tol=1e-6):
        x2 = 2.0 * x
        xr = np.round(x2)
        if np.max(np.abs(x2 - xr)) < tol:
            return xr / 2.0, True
        return x, False
    
    def try_rank(r, max_iters=50, seed_base=0):
        nonlocal best, best_rank
        if r >= best_rank:
            return False
        if time.time() > deadline - 5:
            return False
        
        np.random.seed(seed_base)
        
        # Try different initialization strategies
        for strat in range(5):
            if time.time() > deadline - 3:
                return False
            
            # Strategy 0: random binary
            if strat == 0:
                U = np.random.randint(-1, 2, size=(n*m, r)).astype(float)
                V = np.random.randint(-1, 2, size=(m*p, r)).astype(float)
            # Strategy 1: sparse random
            elif strat == 1:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for _ in range(int(n*m*r*0.3)):
                    U[np.random.randint(n*m), np.random.randint(r)] = np.random.choice([-1, 1])
                for _ in range(int(m*p*r*0.3)):
                    V[np.random.randint(m*p), np.random.randint(r)] = np.random.choice([-1, 1])
            # Strategy 2: structured blocks
            elif strat == 2:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for c in range(r):
                    U[np.random.permutation(n*m)[:3], c] = np.random.choice([-1, 1], size=3)
                    V[np.random.permutation(m*p)[:3], c] = np.random.choice([-1, 1], size=3)
            # Strategy 3: based on known patterns
            elif strat == 3:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for c in range(r):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n*m)) % p
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    if c < n*m*p:
                        pass
                    else:
                        U[(c % n)*m + (c // n) % m, c] = 1
                        V[(c // (n*m)) % m * p + c % p, c] = 1
            # Strategy 4: random with small values
            else:
                U = np.random.choice([-1, 0, 1], size=(n*m, r)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(m*p, r)).astype(float)
            
            # Alternating least squares
            W = np.random.randn(p*n, r) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    return False
                
                # Update U: for each row i*m+j, solve for U[i*m+j,:]
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        b = np.zeros(p)
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # We want sum_c U[i*m+j,c] * A[k,c] = 1 for all k
                        try:
                            sol, res, rank_a, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            sol, res, rank_a, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            sol, res, rank_a, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check convergence and try to snap
                if it % 10 == 9 or it == max_iters - 1:
                    W_snap, ok1 = snap_to_half_int(W)
                    U_snap, ok2 = snap_to_half_int(U)
                    V_snap, ok3 = snap_to_half_int(V)
                    if ok1 and ok2 and ok3:
                        if verify(U_snap, V_snap, W_snap):
                            if r < best_rank:
                                best_rank = r
                                best = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                            return True
                    # Also try snapping only W
                    W_snap, ok1 = snap_to_half_int(W)
                    if ok1:
                        if verify(U, V, W_snap):
                            if r < best_rank:
                                best_rank = r
                                best = {"U": U.tolist(), "V": V.tolist(), "W": W_snap.tolist()}
                            return True
                    U_snap, ok2 = snap_to_half_int(U)
                    V_snap, ok3 = snap_to_half_int(V)
                    if ok2 and ok3:
                        if verify(U_snap, V_snap, W):
                            if r < best_rank:
                                best_rank = r
                                best = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W.tolist()}
                            return True
                    if ok2:
                        if verify(U_snap, V, W):
                            if r < best_rank:
                                best_rank = r
                                best = {"U": U_snap.tolist(), "V": V.tolist(), "W": W.tolist()}
                            return True
                    if ok3:
                        if verify(U, V_snap, W):
                            if r < best_rank:
                                best_rank = r
                                best = {"U": U.tolist(), "V": V_snap.tolist(), "W": W.tolist()}
                            return True
        
        return False
    
    # Search ranks from low to high
    # Known lower bound for 3x3x3 is 19, so for 3x3x7 maybe around 25-35
    for r in range(18, 63):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        # Try multiple seeds for each rank
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            if try_rank(r, max_iters=80, seed_base=seed * 1000 + 42):
                # Found solution, continue to lower ranks? No, we're going up.
                # But we should check if we can find lower ones
                break
        # If we found a solution at this rank, we can stop since we're going in increasing order
        # and lower ranks were already tried
        if best_rank <= r:
            break
    
    return best

