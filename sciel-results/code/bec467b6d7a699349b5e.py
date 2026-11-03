import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build the target tensor
    # T[a*m+b, c*p+d, k*n+i] = 1 if b==c and a==i and d==k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                T[i*m+b, b*p+k, k*n+i] = 1.0
    
    def get_rank36_solution():
        """Strassen 2x2x2 blocks + schoolbook for remaining column."""
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1 = (A11+A22)(B11+B22)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2 = (A21+A22)B11
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3 = A11(B21-B22)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4 = A22(B11-B12)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5 = (A11+A12)B22
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6 = (A21-A11)(B11+B12)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7 = (A12-A22)(B21+B22)
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

        # Apply Strassen to four 2x2x2 blocks
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        # Schoolbook for the 5th column (k=4)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    def residual(x, R):
        size_U = n * m * R
        size_V = m * p * R
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()

    def gradient(x, R):
        size_U = n * m * R
        size_V = m * p * R
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        res = T_rec - T
        gU = np.einsum('abc,bm,cm->am', res, V, W, optimize=True)
        gV = np.einsum('abc,am,cm->bm', res, U, W, optimize=True)
        gW = np.einsum('abc,am,bm->cm', res, U, V, optimize=True)
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

    start_time = time.time()
    time_limit = 145
    
    best_rank = 36
    U36, V36, W36 = get_rank36_solution()
    best_result = (U36, V36, W36)
    
    # Try to find lower rank solutions
    for R in range(35, 24, -1):
        if time.time() - start_time > time_limit:
            break
            
        size_U = n * m * R
        size_V = m * p * R
        
        # Multiple random restarts
        n_attempts = max(3, int(120 * (time_limit - (time.time() - start_time)) / (time_limit - start_time)))
        n_attempts = min(n_attempts, 15)
        
        found = False
        for attempt in range(n_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Perturb the rank-36 solution truncated
                U_init = U36[:, :R] + np.random.normal(0, 0.1, (n*m, R))
                V_init = V36[:, :R] + np.random.normal(0, 0.1, (m*p, R))
                W_init = W36[:, :R] + np.random.normal(0, 0.1, (p*n, R))
            elif attempt == 1:
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 2:
                U_init = np.random.choice([-1, 0, 0, 1], (n*m, R))
                V_init = np.random.choice([-1, 0, 0, 1], (m*p, R))
                W_init = np.random.choice([-1, 0, 0, 1], (p*n, R))
            elif attempt == 3:
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            else:
                scale = 0.3 + 0.15 * (attempt % 4)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # First pass: L-BFGS-B
                result = opt.minimize(
                    lambda x: 0.5 * np.sum(residual(x, R)**2),
                    x0,
                    method='L-BFGS-B',
                    jac=lambda x: gradient(x, R),
                    maxiter=3000,
                    maxfun=8000,
                    options={'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                x_opt = result.x
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
                
                T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
                max_res = np.max(np.abs(T_rec - T))
                
                if max_res < 1e-5:
                    # Polish with tighter tolerance
                    result2 = opt.minimize(
                        lambda x: 0.5 * np.sum(residual(x, R)**2),
                        x_opt,
                        method='L-BFGS-B',
                        jac=lambda x: gradient(x, R),
                        maxiter=10000,
                        maxfun=20000,
                        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
                    )
                    x_pol = result2.x
                    U_pol = x_pol[0:size_U].reshape(n*m, R)
                    V_pol = x_pol[size_U:size_U+size_V].reshape(m*p, R)
                    W_pol = x_pol[size_U+size_V:].reshape(p*n, R)
                    
                    T_rec_pol = np.einsum('am,bm,cm->abc', U_pol, V_pol, W_pol, optimize=True)
                    max_res_pol = np.max(np.abs(T_rec_pol - T))
                    
                    if max_res_pol < 1e-8:
                        U_snap, V_snap, W_snap = snap_to_half_integers(U_pol, V_pol, W_pol)
                        T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
                        if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                            best_rank = R
                            best_result = (U_snap, V_snap, W_snap)
                            found = True
                            break
                
                elif max_res < 1e-3:
                    # Try more aggressive polishing
                    for polish_iter in range(3):
                        if time.time() - start_time > time_limit:
                            break
                        result_p = opt.minimize(
                            lambda x: 0.5 * np.sum(residual(x, R)**2),
                            x_opt,
                            method='L-BFGS-B',
                            jac=lambda x: gradient(x, R),
                            maxiter=5000,
                            maxfun=10000,
                            options={'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 80}
                        )
                        x_opt = result_p.x
                        U_opt = x_opt[0:size_U].reshape(n*m, R)
                        V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                        W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
                        T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
                        max_res = np.max(np.abs(T_rec - T))
                        if max_res < 1e-8:
                            U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                            T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
                            if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                                best_rank = R
                                best_result = (U_snap, V_snap, W_snap)
                                found = True
                                break
                    if found:
                        break
                    
            except Exception:
                pass
        
        if found:
            break
    
    U_out, V_out, W_out = best_result
    U_out, V_out, W_out = snap_to_half_integers(U_out, V_out, W_out)
    
    # Final verification
    T_rec = np.einsum('am,bm,cm->abc', U_out, V_out, W_out, optimize=True)
    if np.max(np.abs(T_rec - T)) < 1e-9:
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    # Fallback to known good solution
    U36, V36, W36 = get_rank36_solution()
    U36, V36, W36 = snap_to_half_integers(U36, V36, W36)
    T_rec = np.einsum('am,bm,cm->abc', U36, V36, W36, optimize=True)
    if np.max(np.abs(T_rec - T)) < 1e-9:
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
