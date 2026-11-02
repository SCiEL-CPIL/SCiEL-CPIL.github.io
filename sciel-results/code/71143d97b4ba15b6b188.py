import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = 115.0 # Safety margin

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    def verify(U, V, W):
        M = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(M, T, atol=1e-8)

    def try_rank(R, max_restarts=200):
        if time.time() > deadline - 5:
            return None
        
        # Strategy: Local search with random restarts
        # We use L-BFGS-B on the flattened parameters.
        # To handle the "half-integer" constraint, we search in continuous space
        # and then round. If rounding fails, we try to fix it by local adjustments
        # or just accept if the continuous solution is extremely close (which implies
        # a half-integer solution exists nearby if the landscape is nice).
        # However, the problem statement says entries must be integer/half-integer.
        # So we MUST round and verify.
        
        # Better strategy: Initialize with random values from {-1, -0.5, 0, 0.5, 1}
        # and do local search on the continuous relaxation, then round.
        # Or, do a discrete search? Discrete search on 3x3x5 is hard.
        
        # Let's use a hybrid:
        # 1. Random continuous init.
        # 2. Optimize continuous.
        # 3. Round to half-integers.
        # 4. Verify.
        # 5. If fail, try "snap" optimization: fix U, V to rounded values, optimize W?
        #    Or optimize all but keep them close to rounded values?
        
        # Actually, a more robust approach for small ranks:
        # Use differential evolution or basin hopping on the discrete space?
        # The space is huge.
        
        # Let's stick to continuous optimization + rounding, but with more restarts
        # and better initialization.
        
        # Known lower bound for 3x3x5 is 14?
        # Strassen for 3x3 is 7. For 3x3x5, we can extend Strassen.
        # Let's try to find a solution.
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                return None
                
            # Initialize with random continuous values
            # Scale: entries are typically small (0, 1, -1).
            U = np.random.uniform(-1.5, 1.5, (nm, R))
            V = np.random.uniform(-1.5, 1.5, (mp, R))
            W = np.random.uniform(-1.5, 1.5, (pn, R))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat = params[:nm*R]
                V_flat = params[nm*R:nm*R + mp*R]
                W_flat = params[nm*R + mp*R:]
                
                U_ = U_flat.reshape(nm, R)
                V_ = V_flat.reshape(mp, R)
                W_ = W_flat.reshape(pn, R)
                
                M = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
                
                # Sparse objective: only compute non-zero entries of T
                # T has 45 non-zero entries.
                # M has nm*mp*pn = 9*15*15 = 2025 entries.
                # Computing full M is fast enough.
                data_err = np.sum((M - T)**2)
                return data_err
            
            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                U_opt = res.x[:nm*R].reshape(nm, R)
                V_opt = res.x[nm*R:nm*R + mp*R].reshape(mp, R)
                W_opt = res.x[nm*R + mp*R:].reshape(pn, R)
                
                # Round to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                
                # If rounding failed, try a "polish" step:
                # Optimize W given U_rounded, V_rounded?
                # Or optimize all but penalize deviation from rounded values?
                
                # Let's try a simple coordinate ascent on the discrete lattice?
                # Too slow.
                
                # Try one more optimization step starting from the rounded values
                # to see if we can get closer to 0 error, which might suggest
                # a better rounding strategy or that the continuous solution
                # was slightly off.
                
                # Actually, if the continuous solution has error < 1e-8,
                # the rounded solution should be exact IF the true solution
                # is nearby. If not, we just continue.
                
        return None

    # Try ranks from 14 upwards
    # 14 is a known lower bound for 3x3x5?
    # 3x3x3 is 7. 3x3x5 should be at least 14?
    # Let's try 14, 15, 16...
    
    ranks_to_try = list(range(14, 46))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Increase restarts for lower ranks as they are harder
        max_restarts = 500 if R < 20 else 200
        
        res = try_rank(R, max_restarts)
        
        if res is not None:
            best_rank = R
            best_solution = res
            break

    return best_solution
