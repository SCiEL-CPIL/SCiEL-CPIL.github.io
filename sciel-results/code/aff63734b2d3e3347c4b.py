"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    deadline = time.time() + 110

    # Build target tensor T as (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        """Check decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c"""
        err = 0.0
        for a in range(nm):
            # Implied: sum_r U[a,r] * outer(V[:,r], W[:,r])
            # = (V * U[a, :]) @ W.T
            implied = (V * U[a, :]) @ W.T
            err += np.sum((implied - T[a]) ** 2)
        return np.sqrt(err)

    def solve_U(U, V, W, T):
        """Solve for U given V, W, T using least squares"""
        U_new = np.zeros_like(U)
        r = U.shape[1]
        # For each a: T[a] (mp x pn) = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # Vectorize: T[a].flatten() = U[a, :] @ Phi_VW
        # where Phi_VW[(b,c), r] = V[b,r] * W[c,r]
        Phi_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
        for a in range(nm):
            target = T[a].flatten()
            # Solve U[a, :] @ Phi_VW = target
            # Least squares: U[a, :] = pinv(Phi_VW) @ target
            U_new[a, :] = np.linalg.lstsq(Phi_VW, target, rcond=None)[0]
        return U_new

    def solve_V(U, V, W, T):
        """Solve for V given U, W, T"""
        V_new = np.zeros_like(V)
        r = V.shape[1]
        # For each b: T[:, b, :] (nm x pn) = sum_r V[b,r] * outer(U[:,r], W[:,r])
        # Vectorize: for each b, (T[:, b, :]).flatten() = V[b, :] @ Phi_UW
        # where Phi_UW[(a,c), r] = U[a,r] * W[c,r]
        Phi_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
        for b in range(mp):
            target = T[:, b, :].flatten()
            V_new[b, :] = np.linalg.lstsq(Phi_UW, target, rcond=None)[0]
        return V_new

    def solve_W(U, V, W, T):
        """Solve for W given U, V, T"""
        W_new = np.zeros_like(W)
        r = W.shape[1]
        # For each c: T[:, :, c] (nm x mp) = sum_r W[c,r] * outer(U[:,r], V[:,r])
        # Vectorize: (T[:, :, c]).flatten() = W[c, :] @ Phi_UV
        # where Phi_UV[(a,b), r] = U[a,r] * V[b,r]
        Phi_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
        for c in range(pn):
            target = T[:, :, c].flatten()
            W_new[c, :] = np.linalg.lstsq(Phi_UV, target, rcond=None)[0]
        return W_new

    def solve_all(U, V, W, T, n_iters=100, tol=1e-8):
        """Run ALS iterations to convergence"""
        best_err = verify(U, V, W)
        for iteration in range(n_iters):
            U = solve_U(U, V, W, T)
            V = solve_V(U, V, W, T)
            W = solve_W(U, V, W, T)
            
            err = verify(U, V, W)
            if err < best_err:
                best_err = err
            if err < tol:
                break
        return U, V, W, best_err

    def try_round(U, V, W, T):
        """Try to round to half-integers or integers and verify"""
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        err_r = verify(U_r, V_r, W_r)
        if err_r < 1e-4:
            return U_r, V_r, W_r, err_r
        
        # Try rounding to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        err_r = verify(U_r, V_r, W_r)
        if err_r < 1e-4:
            return U_r, V_r, W_r, err_r
            
        return None, None, None, err_r

    best_result = None
    best_rank = n * m * p  # schoolbook = 96

    # Generate schoolbook as safe fallback
    U_fb = np.zeros((nm, best_rank))
    V_fb = np.zeros((mp, best_rank))
    W_fb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, col] = 1
                V_fb[j*p + k, col] = 1
                W_fb[k*n + i, col] = 1
                col += 1

    # Helper to create a structured Kronecker-like initialization
    # Strategy: Decompose 3x4x8 into (3x2x4) x (1x2x2) or similar.
    # Here we try to exploit the factorization of dimensions.
    # m=4=2*2, p=8=2*4.
    # We can view the middle index j as (j1, j2) where j1 in 0..1, j2 in 0..1.
    # And k as (k1, k2) where k1 in 0..1, k2 in 0..3.
    # This allows for block structure.
    
    def init_kronecker(r):
        """
        Initialize U, V, W with a structure that respects the Kronecker product of smaller MM problems.
        We assume a decomposition like (n1 x m1 x p1) x (n2 x m2 x p2) where n1*n2=n, etc.
        For 3x4x8:
        n=3, m=4, p=8.
        Let's try n=3, m=2*2, p=2*4.
        This is tricky because n=3 is prime.
        Instead, let's just use a random init but with a specific sparsity pattern or scale.
        """
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        return U, V, W

    def init_structured_j(r):
        """
        Group columns by the middle index j.
        Since the tensor is zero if the j indices don't match, 
        we can try to make U and V sparse in a way that aligns with j.
        """
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.random.randn(pn, r) * 0.5
        
        # Assign columns to j blocks
        cols_per_j = r // m
        for j in range(m):
            c_start = j * cols_per_j
            c_end = c_start + cols_per_j
            if j == m - 1:
                c_end = r
            n_cols = c_end - c_start
            if n_cols > 0:
                # For this j, only rows corresponding to this j in U and V should be non-zero
                # U rows: i*m + j' for j' in 0..m-1. We want only j'=j to be non-zero.
                # V rows: j'*p + k. We want only j'=j to be non-zero.
                # So we can initialize these blocks with random values.
                U[j::m, c_start:c_end] = np.random.randn(n, n_cols) * 0.5
                V[j*p:(j+1)*p, c_start:c_end] = np.random.randn(p, n_cols) * 0.5
                # W can be fully dense or also structured, but let's keep it random for now
                # Actually, W connects k and i. It's independent of j in the index structure 
                # except through the value. So random is fine.
        return U, V, W

    # Ranks to try: 32 (lower bound), 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90
    # Start from 32 to 95
    for r in range(32, min(96, best_rank + 1)):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            break
        
        found = False
        n_tries = 10
        for trial in range(n_tries):
            if time.time() > deadline - 15:
                break
            
            # Different initializations
            if trial == 0:
                U, V, W = init_kronecker(r)
            elif trial == 1:
                U, V, W = init_structured_j(r)
            elif trial == 2:
                # Partial schoolbook
                U = U_fb[:, :r].copy()
                V = V_fb[:, :r].copy()
                W = W_fb[:, :r].copy()
            elif trial == 3:
                # Kronecker-like structure with diagonal boosts
                U, V, W = init_kronecker(r)
                for idx in range(min(r, nm)):
                    U[idx, idx % r] += 1.0
                for idx in range(min(r, mp)):
                    V[idx, idx % r] += 1.0
                for idx in range(min(r, pn)):
                    W[idx, idx % r] += 1.0
            elif trial == 4:
                # Sparse random init
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for rr in range(r):
                    a_idx = np.random.randint(0, nm)
                    b_idx = np.random.randint(0, mp)
                    c_idx = np.random.randint(0, pn)
                    U[a_idx, rr] = 1
                    V[b_idx, rr] = 1
                    W[c_idx, rr] = 1
            else:
                U, V, W = init_kronecker(r)
                # Scale down for larger trials to avoid divergence
                scale = 0.1 + 0.05 * (trial - 5)
                U *= scale
                V *= scale
                W *= scale
            
            # ALS iterations
            U, V, W, best_err = solve_all(U, V, W, T, n_iters=50, tol=1e-6)
            
            if best_err < 1e-5:
                U_r, V_r, W_r, err_r = try_round(U, V, W, T)
                if U_r is not None:
                    best_rank = r
                    best_result = (U_r, V_r, W_r)
                    found = True
                    break
                else:
                    # If rounding failed but error is very small, keep unrounded for now?
                    # No, requirement is integer/half-integer.
                    # But maybe we can polish?
                    # Let's try one more round of ALS with tighter tol
                    U, V, W, best_err2 = solve_all(U, V, W, T, n_iters=50, tol=1e-10)
                    if best_err2 < 1e-8:
                        U_r, V_r, W_r, err_r = try_round(U, V, W, T)
                        if U_r is not None:
                            best_rank = r
                            best_result = (U_r, V_r, W_r)
                            found = True
                            break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(U[i, j]) for j in range(len(U[0]))] for i in range(len(U))]
        V_list = [[float(V[i, j]) for j in range(len(V[0]))] for i in range(len(V))]
        W_list = [[float(W[i, j]) for j in range(len(W[0]))] for i in range(len(W))]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(U_fb[i, j]) for j in range(len(U_fb[0]))] for i in range(len(U_fb))]
        V_list = [[float(V_fb[i, j]) for j in range(len(V_fb[0]))] for i in range(len(V_fb))]
        W_list = [[float(W_fb[i, j]) for j in range(len(W_fb[0]))] for i in range(len(W_fb))]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
