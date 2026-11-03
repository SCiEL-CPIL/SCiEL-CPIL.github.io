import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T for 2x4x5
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                a = i
                d = k
                c = b
                T[a*m+b, c*p+d, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 140

    def verify_solution_copy(U, V, W, tol=1e-9):
        T_copy = T.copy()
        R = U.shape[1]
        for mu in range(R):
            T_copy -= np.einsum('a,b,c->abc', U[:,mu], V[:,mu], W[:,mu])
        return np.max(np.abs(T_copy)) < tol

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def check_and_snap(U, V, W, tol=1e-9):
        U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
        T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.max(np.abs(T_rec_snap - T)) < tol:
            return U_snap, V_snap, W_snap
        return None

    # --- Phase 1: Construct known Strassen-based rank-36 solution ---
    def get_strassen_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        res = check_and_snap(U, V, W)
        if res is not None:
            return res
        return (U, V, W)

    U36, V36, W36 = get_strassen_rank36()
    best_rank = 36
    best_result = (U36, V36, W36)

    # --- Phase 2: Search for lower rank using 2x2x5 block decomposition ---
    # The 2x4x5 problem can be split into two 2x2x5 problems.
    # If we find a rank R decomposition for 2x2x5, we get rank 2R for 2x4x5.
    
    # Construct 2x2x5 tensor
    n2, m2, p2 = 2, 2, 5
    nm2, mp2, pn2 = n2*m2, m2*p2, p2*n2
    T225 = np.zeros((nm2, mp2, pn2))
    for i in range(n2):
        for k in range(p2):
            for b in range(m2):
                T225[i*m2+b, b*p2+k, k*n2+i] = 1.0

    def residual_225(x, R):
        size_U = nm2 * R
        size_V = mp2 * R
        U = x[0:size_U].reshape(nm2, R)
        V = x[size_U:size_U+size_V].reshape(mp2, R)
        W = x[size_U+size_V:].reshape(pn2, R)
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T225).flatten()

    def gradient_225(x, R):
        size_U = nm2 * R
        size_V = mp2 * R
        U = x[0:size_U].reshape(nm2, R)
        V = x[size_U:size_U+size_V].reshape(mp2, R)
        W = x[size_U+size_V:].reshape(pn2, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        residual = T_rec - T225
        
        grad_U = np.einsum('abc,bm,cm->am', residual, V, W, optimize=True)
        grad_V = np.einsum('abc,am,cm->bm', residual, U, W, optimize=True)
        grad_W = np.einsum('abc,am,bm->cm', residual, U, V, optimize=True)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def optimize_225(U_init, V_init, W_init, R, maxiter=2000, maxfun=4000):
        size_U = nm2 * R
        size_V = mp2 * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: 0.5 * np.sum(residual_225(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient_225(x, R),
                maxiter=maxiter,
                maxfun=maxfun,
                options={'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 80}
            )
            
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(nm2, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(mp2, R)
            W_opt = x_opt[size_U+size_V:].reshape(pn2, R)
            
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T225))
            
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf

    def polish_225(U_init, V_init, W_init, R):
        size_U = nm2 * R
        size_V = mp2 * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.minimize(
                lambda x: 0.5 * np.sum(residual_225(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient_225(x, R),
                maxiter=10000,
                maxfun=20000,
                options={'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(nm2, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(mp2, R)
            W_opt = x_opt[size_U+size_V:].reshape(pn2, R)
            
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T225))
            
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf

    def check_and_snap_225(U, V, W, tol=1e-9):
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.max(np.abs(T_rec_snap - T225)) < tol:
            return U_snap, V_snap, W_snap
        return None

    def map_225_to_245(U225, V225, W225):
        R225 = U225.shape[1]
        R = 2 * R225
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # First block: i in {0,1}, j in {0,1}, k in {0..4}
        # U indices: i*m + j. m=4.
        # U225 indices: i*2 + j.
        for i in range(2):
            for j in range(2):
                U[i*4 + j, :R225] = U225[i*2 + j, :]
                U[i*4 + (j+2), R225:] = U225[i*2 + j, :]
        
        # V indices: j*p + k. p=5.
        # V225 indices: j*5 + k.
        for j in range(2):
            for k in range(5):
                V[j*5 + k, :R225] = V225[j*5 + k, :]
                V[(j+2)*5 + k, R225:] = V225[j*5 + k, :]
        
        # W indices: k*n + i. n=2.
        # W225 indices: k*2 + i.
        for k in range(5):
            for i in range(2):
                W[k*2 + i, :R225] = W225[k*2 + i, :]
                W[k*2 + i, R225:] = W225[k*2 + i, :]
                
        return U, V, W

    # Search for lower rank 225
    # Start from 18 (current best for 225 is 18, giving 36 for 245)
    # Try 17, 16, ...
    
    for R225 in range(17, 9, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        found = False
        # Try a few random inits
        for attempt in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            U_init = np.random.normal(0, 1.0, (nm2, R225))
            V_init = np.random.normal(0, 1.0, (mp2, R225))
            W_init = np.random.normal(0, 1.0, (pn2, R225))
            
            U_opt, V_opt, W_opt, max_res = optimize_225(U_init, V_init, W_init, R225)
            if U_opt is not None and max_res < 1e-6:
                U_pol, V_pol, W_pol, max_res_pol = polish_225(U_opt, V_opt, W_opt, R225)
                if U_pol is not None and max_res_pol < 1e-8:
                    res_225 = check_and_snap_225(U_pol, V_pol, W_pol)
                    if res_225 is not None:
                        U225, V225, W225 = res_225
                        U_full, V_full, W_full = map_225_to_245(U225, V225, W225)
                        res_full = check_and_snap(U_full, V_full, W_full)
                        if res_full is not None:
                            best_rank = 2 * R225
                            best_result = res_full
                            found = True
                            break
        
        if found:
            break

    # Final Fallback
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
