import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    # Build the target tensor T as a 3D array
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    # a in [0, nm), b in [0, mp), c in [0, np_)
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def try_rank(R, time_limit=10):
        start_time = time.time()
        # Try several random initializations
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            # Initialize U, V randomly
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.zeros((np_, R))
            
            # ALS iterations
            for _iter in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update W
                # For each c, solve min sum_ab (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
                # This is a linear system: A_c W_c = b_c
                # A_c is (nm*mp) x R
                # b_c is (nm*mp) x 1
                A = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, R) # (nm*mp, R)
                # Wait, U[a,r] * V[b,r]. Sum over r.
                # The term is sum_r (U[a,r] V[b,r]) W[c,r].
                # So for fixed c, the coefficient of W[c,r] is U[a,r] V[b,r].
                # So A is indeed (U * V) reshaped.
                b = T[:, :, c].flatten()
                # Solve least squares
                W[c, :], _res, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Update U
                # For each a, solve min sum_bc (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
                # Coefficient of U[a,r] is V[b,r] W[c,r].
                # A is (mp*np_) x R
                A = (V[:, :, None] * W[None, :, :]).reshape(mp * np_, R)
                b = T[a, :, :].flatten() # Wait, T[a, b, c]
                # We need to do this for all a
                for a in range(nm):
                    A = (V[:, :, None] * W[None, :, :]).reshape(mp * np_, R)
                    # The term is sum_r U[a,r] (V[b,r] W[c,r]).
                    # So A is the same for all a? No, A depends on V and W, which are fixed.
                    # So A is the same matrix for all a.
                    # b depends on a.
                    U[a, :], _res, _rank, _sv = np.linalg.lstsq(A, T[a, :, :].flatten(), rcond=None)
                    
                # Update V
                # For each b, solve min sum_ac (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
                # Coefficient of V[b,r] is U[a,r] W[c,r].
                A = (U[:, :, None] * W[None, :, :]).reshape(nm * np_, R)
                for b in range(mp):
                    V[b, :], _res, _rank, _sv = np.linalg.lstsq(A, T[:, b, :].flatten(), rcond=None)
            
            # Check residual
            # T_approx[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
            # Compute residual
            # This is expensive, so only do it if time permits
            if time.time() - start_time > time_limit - 2:
                break
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.sum((T_approx - T)**2)
            if residual < 1e-10:
                # Check if entries are close to integers or half-integers
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                residual_rounded = np.sum((T_approx_rounded - T)**2)
                if residual_rounded < 1e-10:
                    return U_rounded, V_rounded, W_rounded
        return None

    # Try ranks from 29 down to 30
    # Start with 29
    for R in range(29, 30): # Just 29 for now
        result = try_rank(R)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
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
