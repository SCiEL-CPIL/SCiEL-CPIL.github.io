import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 130  # safety margin

    # Build target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0

    def verify(U, V, W, tol=1e-8):
        # Check active entries equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1.0) > tol:
                        return False
        # Check inactive entries are 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a, j_a = divmod(a, m)
                    j_b, k_b = divmod(b, p)
                    k_c, i_c = divmod(c, n)
                    if not (i_a == i_c and j_a == j_b and k_b == k_c):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True

    def tensor_error(U, V, W):
        """Compute max absolute error of the decomposition vs T."""
        # Efficient: compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is sum_r outer(U[:,r], V[:,r]) * W[:,r] reshaped
        r = U.shape[1]
        total = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for rr in range(r):
            total += U[:,rr][:,None,None] * V[:,rr][None,:,None] * W[:,rr][None,None,:]
        return np.max(np.abs(total - T))

    # Schoolbook initialization
    def schoolbook_init(rank):
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
        return U[:, :rank], V[:, :rank], W[:, :rank]

    def als_decompose(U, V, W, T, max_iter=100, tol=1e-10, deadline=None):
        """Alternating least squares for tensor decomposition."""
        r = U.shape[1]
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            prev_err = tensor_error(U, V, W)
            
            # Update U: for each a, solve min_u ||sum_r u[r]*outer(V[:,r],W[:,r]) - T[a]||
            # Vectorize: A_U[r] = vec(outer(V[:,r], W[:,r]))
            A_U = np.zeros((m*p*p*n, r))
            for rr in range(r):
                A_U[:, rr] = np.outer(V[:,rr], W[:,rr]).ravel()
            
            for a in range(n*m):
                rhs = T[a].ravel()
                try:
                    u_a, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                    U[a, :] = u_a
                except:
                    pass
            
            # Update V
            A_V = np.zeros((n*m*p*n, r))
            for rr in range(r):
                A_V[:, rr] = np.outer(U[:,rr], W[:,rr]).ravel()
            for b in range(m*p):
                rhs = T[:, b, :].ravel()
                try:
                    v_b, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                    V[b, :] = v_b
                except:
                    pass
            
            # Update W
            A_W = np.zeros((n*m*m*p, r))
            for rr in range(r):
                A_W[:, rr] = np.outer(U[:,rr], V[:,rr]).ravel()
            for c in range(p*n):
                rhs = T[:, :, c].ravel()
                try:
                    w_c, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                    W[c, :] = w_c
                except:
                    pass
            
            new_err = tensor_error(U, V, W)
            if abs(prev_err - new_err) < tol:
                break
        
        return U, V, W

    def polish(U, V, W, T, max_iter=50, tol=1e-12, deadline=None):
        """High-precision polish with more iterations."""
        r = U.shape[1]
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            prev_err = tensor_error(U, V, W)
            
            A_U = np.zeros((m*p*p*n, r))
            for rr in range(r):
                A_U[:, rr] = np.outer(V[:,rr], W[:,rr]).ravel()
            for a in range(n*m):
                rhs = T[a].ravel()
                try:
                    u_a, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=1e-14)
                    U[a, :] = u_a
                except:
                    pass
            
            A_V = np.zeros((n*m*p*n, r))
            for rr in range(r):
                A_V[:, rr] = np.outer(U[:,rr], W[:,rr]).ravel()
            for b in range(m*p):
                rhs = T[:, b, :].ravel()
                try:
                    v_b, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=1e-14)
                    V[b, :] = v_b
                except:
                    pass
            
            A_W = np.zeros((n*m*m*p, r))
            for rr in range(r):
                A_W[:, rr] = np.outer(U[:,rr], V[:,rr]).ravel()
            for c in range(p*n):
                rhs = T[:, :, c].ravel()
                try:
                    w_c, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=1e-14)
                    W[c, :] = w_c
                except:
                    pass
            
            new_err = tensor_error(U, V, W)
            if abs(prev_err - new_err) < tol:
                break
        
        return U, V, W

    def round_to_half(mat):
        """Round entries to nearest half-integer if close."""
        mat_r = mat.copy()
        for i in range(mat_r.shape[0]):
            for j in range(mat_r.shape[1]):
                val = mat_r[i,j]
                r1 = round(val * 2) / 2.0
                r2 = round(val)
                if abs(val - r1) < 1e-5:
                    mat_r[i,j] = r1
                elif abs(val - r2) < 1e-5:
                    mat_r[i,j] = float(r2)
                else:
                    mat_r[i,j] = val
        return mat_r

    best = None
    best_rank = 40

    # Try ranks from low to high
    for target_rank in range(25, 40):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Multiple initializations
        inits = 0
        max_inits = 8
        
        while inits < max_inits and not found:
            if time.time() > deadline - 30:
                break
            
            if inits == 0:
                # Schoolbook truncation
                U0, V0, W0 = schoolbook_init(target_rank)
            elif inits == 1:
                # Random small
                rng = np.random.RandomState(inits * 1000 + target_rank)
                U0 = rng.randn(n*m, target_rank) * 0.1
                V0 = rng.randn(m*p, target_rank) * 0.1
                W0 = rng.randn(p*n, target_rank) * 0.1
            else:
                # Perturbed schoolbook
                rng = np.random.RandomState(inits * 7919 + target_rank)
                U0, V0, W0 = schoolbook_init(target_rank)
                U0 += rng.randn(n*m, target_rank) * 0.05
                V0 += rng.randn(m*p, target_rank) * 0.05
                W0 += rng.randn(p*n, target_rank) * 0.05
            
            U1, V1, W1 = als_decompose(U0.copy(), V0.copy(), W0.copy(), T, 
                                        max_iter=80, tol=1e-10, deadline=deadline - 30)
            
            err = tensor_error(U1, V1, W1)
            if err < 1e-6:
                # Polish
                U1, V1, W1 = polish(U1, V1, W1, T, max_iter=40, tol=1e-13, deadline=deadline - 10)
                err = tensor_error(U1, V1, W1)
                if err < 1e-8:
                    found = True
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best = (U1.copy(), V1.copy(), W1.copy())
            
            inits += 1
        
        if found:
            # Found a valid decomposition at this rank; check if we can go lower
            # But since we're going low to high, break and try even lower
            # Actually we go from 25 up, so if we found at rank r, we should 
            # continue checking if lower ranks also work (but we already tried them)
            # So break here since we want the minimum
            break
    
    if best is not None:
        U, V, W = best
        # Try to round
        U_r, V_r, W_r = round_to_half(U), round_to_half(V), round_to_half(W)
        if verify(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook
        U, V, W = schoolbook_init(40)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
