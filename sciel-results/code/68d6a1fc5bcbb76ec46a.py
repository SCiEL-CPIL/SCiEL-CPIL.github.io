import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s

    # Deadline guard: 140s to leave 10s for overhead/serialization
    deadline = time.time() + 140

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        # Vectorized computation of the tensor S
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # S: (nm, mp, pn)
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    def round_and_verify(U, V, W):
        """
        Try to round U, V, W to integers or half-integers and verify.
        Returns (U_int, V_int, W_int) if successful, else None.
        """
        # Try integers first
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half):
            return U_half, V_half, W_half
            
        # Try quarters (0.25) as a broader net
        U_q = np.round(U * 4) / 4.0
        V_q = np.round(V * 4) / 4.0
        W_q = np.round(W * 4) / 4.0
        if verify(U_q, V_q, W_q):
            return U_q, V_q, W_q

        return None

    def try_rank(R, max_attempts=2):
        """
        Attempt to find a decomposition of rank R.
        Uses random linear combinations of schoolbook basis vectors as initialization.
        """
        if R >= best_rank:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize U, V, W with random linear combinations of schoolbook columns
            U_init = np.zeros((nm, R))
            V_init = np.zeros((mp, R))
            W_init = np.zeros((pn, R))
            
            # Correct initialization for coupled basis:
            # u_r = sum_{i,j,k} c_{r,i,j,k} e_{i,j}
            # v_r = sum_{i,j,k} c_{r,i,j,k} e_{j,k}
            # w_r = sum_{i,j,k} c_{r,i,j,k} e_{k,i}
            
            C = np.random.randn(R, n, m, p)
            
            U_init = np.zeros((nm, R))
            V_init = np.zeros((mp, R))
            W_init = np.zeros((pn, R))
            
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = C[r, i, j, k]
                            U_init[i*m+j, r] += val
                            V_init[j*p+k, r] += val
                            W_init[k*n+i, r] += val
                            
            # Add small noise to avoid symmetry breaking issues
            U_init += np.random.randn(nm, R) * 0.1
            V_init += np.random.randn(mp, R) * 0.1
            W_init += np.random.randn(pn, R) * 0.1

            # Optimize
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    Ur = params[:nm*R].reshape(nm, R)
                    Vr = params[nm*R:nm*R+mp*R].reshape(mp, R)
                    Wr = params[nm*R+mp*R:].reshape(pn, R)
                    
                    # Compute S = sum_r Ur[:,r] outer Vr[:,r] outer Wr[:,r]
                    # Vectorized for speed
                    S = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
                    diff = S - T
                    return np.sum(diff * diff)
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                # Use L-BFGS-B
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000})
                
                if res.fun < 1e-10:
                    Ur = res.x[:nm*R].reshape(nm, R)
                    Vr = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    Wr = res.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try to round
                    result = round_and_verify(Ur, Vr, Wr)
                    if result:
                        return result
                        
            except Exception:
                pass
                
        return None

    # Try to reduce rank
    # Start from best_rank - 1 and go down
    # Limit the search to a reasonable range to fit in time
    # Schoolbook is 84. Try down to 70.
    for R in range(best_rank - 1, 69, -1):
        if time.time() > deadline:
            break
            
        # Try a few attempts for each rank
        # More attempts for lower ranks? No, lower ranks are harder.
        # But we break as soon as we find one.
        res = try_rank(R, max_attempts=3)
        
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = R
            best_U, best_V, best_W = Ur, Vr, Wr
            # We found a lower rank, can we go lower?
            # Yes, continue the loop.
            
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
