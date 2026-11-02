import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
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
    
    # Use a safe deadline, e.g., 100s to leave buffer for overhead
    deadline = time.time() + 100

    def verify(U, V, W, tol=1e-10):
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)

    def try_rank(R, max_iters=20, n_attempts=3):
        """
        Optimized ALS with rounding.
        """
        for attempt in range(n_attempts):
            if time.time() > deadline:
                return None
            
            # Sparse initialization
            density = 0.3
            Ur = np.random.choice([-1, 0, 1], (nm, R))
            Ur[np.random.rand(nm, R) > density] = 0
            Vr = np.random.choice([-1, 0, 1], (mp, R))
            Vr[np.random.rand(mp, R) > density] = 0
            Wr = np.random.choice([-1, 0, 1], (pn, R))
            Wr[np.random.rand(pn, R) > density] = 0
            
            if not np.any(Ur): Ur[0,0] = 1
            if not np.any(Vr): Vr[0,0] = 1
            if not np.any(Wr): Wr[0,0] = 1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Precompute S_total to speed up updates
                S_total = np.zeros((nm, mp, pn))
                for s in range(R):
                    if np.any(Ur[:, s]) and np.any(Vr[:, s]) and np.any(Wr[:, s]):
                        S_total += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                
                # Update U
                for r_idx in range(R):
                    S_r = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                    S_other = S_total - S_r
                    Res = T - S_other
                    X_r = np.outer(Vr[:, r_idx], Wr[:, r_idx])
                    X_flat = X_r.flatten()
                    X_norm_sq = np.dot(X_flat, X_flat)
                    if X_norm_sq < 1e-15:
                        continue
                    Res_mat = Res.reshape(nm, mp * pn)
                    Ur[:, r_idx] = (Res_mat @ X_flat) / X_norm_sq
                    # Update S_total incrementally? No, S_total is used for all U updates.
                    # But Ur changed, so S_total is stale for the next r_idx?
                    # Actually, in ALS, we usually update one variable at a time and keep others fixed.
                    # If we update Ur[:, r_idx], S_total should reflect the new Ur[:, r_idx] for subsequent updates in the same pass?
                    # Standard ALS updates U_r based on fixed V, W. Then moves to U_{r+1}.
                    # The "S_other" for U_{r+1} should include the *new* U_r.
                    # So we must update S_total after each U update.
                    S_r_new = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                    S_total = S_total - S_r + S_r_new

                # Update V
                for r_idx in range(R):
                    S_r = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                    S_other = S_total - S_r
                    Res = T - S_other
                    X_r = np.outer(Ur[:, r_idx], Wr[:, r_idx])
                    X_flat = X_r.flatten()
                    X_norm_sq = np.dot(X_flat, X_flat)
                    if X_norm_sq < 1e-15:
                        continue
                    Res_mat = Res.reshape(mp, nm * pn)
                    Vr[:, r_idx] = (Res_mat @ X_flat) / X_norm_sq
                    S_r_new = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                    S_total = S_total - S_r + S_r_new

                # Update W
                for r_idx in range(R):
                    S_r = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                    S_other = S_total - S_r
                    Res = T - S_other
                    X_r = np.outer(Ur[:, r_idx], Vr[:, r_idx])
                    X_flat = X_r.flatten()
                    X_norm_sq = np.dot(X_flat, X_flat)
                    if X_norm_sq < 1e-15:
                        continue
                    Res_mat = Res.reshape(pn, nm * mp)
                    Wr[:, r_idx] = (Res_mat @ X_flat) / X_norm_sq
                    S_r_new = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                    S_total = S_total - S_r + S_r_new
                
                # Check for exact solution after rounding
                Ur_int = np.round(Ur)
                Vr_int = np.round(Vr)
                Wr_int = np.round(Wr)
                
                if verify(Ur_int, Vr_int, Wr_int):
                    return Ur_int, Vr_int, Wr_int
                    
        return None

    found = False
    # Try ranks slightly below schoolbook (84) first, as they are easier to find
    # If we find one, we stop. If not, we return schoolbook.
    # Trying very low ranks (e.g. 30) with this method is unlikely to succeed in 100s.
    target_ranks = [80, 75, 70, 65]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        res = try_rank(R, max_iters=10, n_attempts=5)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = R
            best_U, best_V, best_W = Ur, Vr, Wr
            found = True
            break

    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
