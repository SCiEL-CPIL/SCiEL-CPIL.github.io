import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    # Precompute the target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    # We only need to check the 27 "1" positions and a subset of "0" positions for speed?
    # No, for exactness, we must check all or be very confident.
    # Actually, if the 27 "1"s are satisfied and the rank is low, the "0"s are likely satisfied?
    # No, we must check the "0"s to ensure it's not a "bogus" solution.
    
    # Let's store the indices for the "1" entries
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_indices.append((a, b, c))
                
    # We also need to check some "0" entries to ensure the decomposition is correct.
    # If we check all 729 entries, it's fast enough (729 * rank ops).
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check all 1s
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
                
        # Check a random subset of 0s to be safe, or all 0s if time permits
        # For rank < 27, the space is constrained.
        # Let's check all 0s to be absolutely sure.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in ones_indices:
                        continue
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s) > tol:
                        return False
        return True

    # Schoolbook rank 27
    rank_sb = 27
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    # Try to find lower rank solutions
    # We will try ranks 23, 24, 25, 26
    # Using scipy.optimize.least_squares
    
    from scipy.optimize import least_squares
    
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > deadline - 10:
            break
            
        # Try a few random starts
        for attempt in range(5):
            if time.time() - start_time > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + r)
            
            # Initialize with small random values
            U_init = np.random.randn(nm, r) * 0.5
            V_init = np.random.randn(mp, r) * 0.5
            W_init = np.random.randn(pn, r) * 0.5
            
            def residual(x):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:2*nm*r].reshape(mp, r)
                W_f = x[2*nm*r:3*nm*r].reshape(pn, r)
                
                res = []
                # 1s
                for a, b, c in ones_indices:
                    s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                    res.append(s - 1.0)
                # 0s (sampled to speed up optimization, verified exactly later)
                # We need to enforce 0s during optimization too, otherwise it might converge to a solution that satisfies 1s but not 0s.
                # However, with rank < 27, satisfying 1s might not be enough.
                # Let's add a subset of 0s.
                # There are 702 zeros. Let's pick 100 random ones for the optimization step.
                # Actually, just using the 1s might lead to a "spurious" solution.
                # But if we verify exactly later, it's safe.
                # To help the optimizer, let's include some 0s.
                # We'll pre-generate indices for 0s.
                # For simplicity in this snippet, let's just use the 1s for the residual in the optimization step, 
                # and rely on the exact verification to filter out bad solutions.
                # This is risky. Better to include 0s.
                
                # Let's add 50 random 0 constraints to the residual
                # We need a list of 0 indices
                # Generating 702 indices every call is slow.
                # Pre-generate them.
                
                return np.array(res)
            
            # Pre-generate 0 indices for optimization
            all_indices = [(a,b,c) for a in range(nm) for b in range(mp) for c in range(pn)]
            zero_indices = [idx for idx in all_indices if idx not in ones_indices]
            # Pick 100 random zero indices for this attempt
            np.random.shuffle(zero_indices)
            opt_zero_indices = zero_indices[:100]
            
            def residual_full(x):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:2*nm*r].reshape(mp, r)
                W_f = x[2*nm*r:3*nm*r].reshape(pn, r)
                
                res = []
                for a, b, c in ones_indices:
                    s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                    res.append(s - 1.0)
                for a, b, c in opt_zero_indices:
                    s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                    res.append(s)
                return np.array(res)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual_full, x0, method='trf', max_nfev=1000)
                
                if sol.success or sol.cost < 1e-6:
                    U_sol = sol.x[:nm*r].reshape(nm, r)
                    V_sol = sol.x[nm*r:2*nm*r].reshape(mp, r)
                    W_sol = sol.x[2*nm*r:3*nm*r].reshape(pn, r)
                    
                    # Round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if verify(U_r, V_r, W_r, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        break # Found a valid solution for this rank, go to next rank
            except Exception:
                pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
