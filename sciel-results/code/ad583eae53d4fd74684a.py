import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n

    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    deadline = time.time() + 115  # Safety margin before 150s kill

    def verify(U, V, W, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)

    def snap_half_integer(mat):
        return np.round(2.0 * mat) / 2.0

    def try_solve_U(V, W, T, r):
        """Given V and W, solve for U exactly."""
        # M shape: (mp*pn, r) where M[bp, r] = sum over c of V[b,r]*W[c,r]
        # Actually: for each a (row of U), T[a,:,:].flatten() = V^T ⊗ W^T times U[a,:]
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # So T[a,:,:].flatten() = (V ⊙ W)^T * U[a,:] where (V⊙W)[b*c, r] = V[b,r]*W[c,r]
        
        M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
        U = np.zeros((nm, r))
        for a in range(nm):
            target = T[a, :, :].flatten()
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                if residuals[0] > 1e-8:
                    return None
                U[a, :] = sol
            except:
                return None
        return U

    def try_solve_V(U, W, T, r):
        """Given U and W, solve for V exactly."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each b: T[:,:,b].flatten() = (U ⊙ W)^T * V[b,:]
        # (U⊙W)[a*c, r] = U[a,r]*W[c,r]
        M = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, r)
        V = np.zeros((mp, r))
        for b in range(mp):
            target = T[:, :, b].flatten()
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                if residuals[0] > 1e-8:
                    return None
                V[b, :] = sol
            except:
                return None
        return V

    def try_solve_W(U, V, T, r):
        """Given U and V, solve for W exactly."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each c: T[:,:,c].flatten() = (U ⊙ V)^T * W[c,:]
        # (U⊙V)[a*b, r] = U[a,r]*V[b,r]
        M = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, r)
        W = np.zeros((pn, r))
        for c in range(pn):
            target = T[:, :, c].flatten()
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                if residuals[0] > 1e-8:
                    return None
                W[c, :] = sol
            except:
                return None
        return W

    def als_decompose(T, r, max_iter=50, tol=1e-10):
        """ALS decomposition with random initialization."""
        rng = np.random.RandomState(hash(r) % (2**31))
        U = rng.randint(-1, 2, size=(nm, r)).astype(float)
        V = rng.randint(-1, 2, size=(mp, r)).astype(float)
        W = rng.randint(-1, 2, size=(pn, r)).astype(float)
        
        for it in range(max_iter):
            U_new = try_solve_U(V, W, T, r)
            if U_new is None:
                return None
            V_new = try_solve_V(U_new, W, T, r)
            if V_new is None:
                return None
            W_new = try_solve_W(U_new, V_new, T, r)
            if W_new is None:
                return None
            
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                if verify(U, V, W):
                    return U, V, W
                break
        
        if verify(U, V, W):
            return U, V, W
        return None

    def random_integer_decompose(T, r, max_trials=200):
        """Try random integer initializations for V and W, solve for U."""
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            np.random.seed(42 + r * 1000 + trial)
            
            # Try different sparsity patterns
            density = 0.3 + 0.5 * np.random.random()
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for i in range(mp):
                mask = np.random.random(r) < density
                V[i, mask] = np.random.randint(-2, 3, size=mask.sum())
            for i in range(pn):
                mask = np.random.random(r) < density
                W[i, mask] = np.random.randint(-2, 3, size=mask.sum())
            
            U = try_solve_U(V, W, T, r)
            if U is not None:
                if verify(U, V, W):
                    # Try to snap to half-integers
                    U_s = snap_half_integer(U)
                    V_s = snap_half_integer(V)
                    W_s = snap_half_integer(W)
                    if verify(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                    return U, V, W
        
        return None

    # Schoolbook decomposition (rank 54)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
    best_rank = rank_sb

    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But the structure may require more.
    
    for r in range(27, best_rank):
        if time.time() > deadline:
            break
        
        # Try ALS with several random initializations
        found = None
        for attempt in range(5):
            if time.time() > deadline:
                break
            result = als_decompose(T, r, max_iter=30)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    # Try to snap to half-integers
                    U_s = snap_half_integer(U)
                    V_s = snap_half_integer(V)
                    W_s = snap_half_integer(W)
                    if verify(U_s, V_s, W_s):
                        found = (U_s, V_s, W_s)
                    else:
                        found = (U, V, W)
                    break
        
        if found is None:
            # Try random integer approach
            result = random_integer_decompose(T, r, max_trials=50)
            if result is not None:
                found = result
        
        if found is not None:
            U, V, W = found
            if verify(U, V, W):
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                best_rank = r
                break  # Found the minimum, stop searching
    
    return best_result
