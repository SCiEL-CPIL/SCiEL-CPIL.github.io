import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Precompute indices for the 1600 constraints
    # Constraint j corresponds to (i, k, a, b, c, d)
    # Target is 1 if (b==c and a==i and d==k) else 0
    # Indices:
    # U index: a*m + b
    # V index: c*p + d
    # W index: k*n + i
    
    cons_i = []
    cons_k = []
    cons_a = []
    cons_b = []
    cons_c = []
    cons_d = []
    cons_target = []
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            cons_i.append(i)
                            cons_k.append(k)
                            cons_a.append(a)
                            cons_b.append(b)
                            cons_c.append(c)
                            cons_d.append(d)
                            cons_target.append(1.0 if (b == c and a == i and d == k) else 0.0)
                            
    cons_i = np.array(cons_i)
    cons_k = np.array(cons_k)
    cons_a = np.array(cons_a)
    cons_b = np.array(cons_b)
    cons_c = np.array(cons_c)
    cons_d = np.array(cons_d)
    cons_target = np.array(cons_target)
    
    idx_U = cons_a * m + cons_b
    idx_V = cons_c * p + cons_d
    idx_W = cons_k * n + cons_i
    
    n_cons = len(cons_target)
    
    def get_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][col] = 1
                    V[(j0 + dj)*p + k0][col] = 1
                    W[k0*n + (i0 + di)][col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    U36, V36, W36 = get_rank36()
    
    def residual(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        U_sel = U[idx_U]
        V_sel = V[idx_V]
        W_sel = W[idx_W]
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return vals - cons_target

    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_sel = U[idx_U]
        V_sel = V[idx_V]
        W_sel = W[idx_W]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - cons_target)) < tol

    def snap(U, V, W):
        return np.round(U*2)/2.0, np.round(V*2)/2.0, np.round(W*2)/2.0

    def polish(U, V, W, R, max_iter=100, tol=1e-10):
        """High-precision polish using Gauss-Newton on the nonlinear system"""
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        for it in range(max_iter):
            res = residual(x, R)
            if np.max(np.abs(res)) < tol:
                break
            
            # Compute Jacobian numerically (or analytically)
            # Use finite differences for simplicity
            J = np.zeros((n_cons, len(x)))
            eps = 1e-7
            for j in range(0, len(x), 50):  # Batch to save time
                for jj in range(j, min(j+50, len(x))):
                    xp = x.copy()
                    xp[jj] += eps
                    xm = x.copy()
                    xm[jj] -= eps
                    J[:, jj] = (residual(xp, R) - residual(xm, R)) / (2*eps)
            
            # Solve least squares
            try:
                dx, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            except:
                break
            
            x = x + dx
        
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_opt = x[0:size_U].reshape(n*m, R)
        V_opt = x[size_U:size_U+size_V].reshape(m*p, R)
        W_opt = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
        
        return U_opt, V_opt, W_opt

    start_time = time.time()
    deadline = 140.0
    
    # Try to find lower rank
    # We start from 35 down to 20
    for R in range(35, 19, -1):
        if time.time() > deadline:
            break
            
        # Try multiple initializations
        # 1. Truncate 36
        # 2. Random
        # 3. Random with small noise on truncated
        # 4. SVD-based from rank 36
        
        attempts = 0
        max_attempts = 20  # More attempts with faster residual
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            if attempts == 1:
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempts == 2:
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempts == 3:
                # Perturb truncated
                U_init = U36[:, :R] + np.random.normal(0, 0.1, (n*m, R))
                V_init = V36[:, :R] + np.random.normal(0, 0.1, (m*p, R))
                W_init = W36[:, :R] + np.random.normal(0, 0.1, (p*n, R))
            elif attempts == 4:
                # SVD-based: project rank 36 to lower rank
                U36_svd = U36.copy()
                V36_svd = V36.copy()
                W36_svd = W36.copy()
                
                # Do SVD on each and keep top R components
                try:
                    _, _, Vt_U = np.linalg.svd(U36_svd, full_matrices=False)
                    U_init = U36_svd @ Vt_U[:R, :].T
                    _, _, Vt_V = np.linalg.svd(V36_svd, full_matrices=False)
                    V_init = V36_svd @ Vt_V[:R, :].T
                    _, _, Vt_W = np.linalg.svd(W36_svd, full_matrices=False)
                    W_init = W36_svd @ Vt_W[:R, :].T
                except:
                    U_init = U36[:, :R]
                    V_init = V36[:, :R]
                    W_init = W36[:, :R]
            elif attempts == 5:
                # Try different blocking: 2x2x2 Strassen + 2x2x3
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                col = 0
                # Add 4 Strassen 2x2x2 blocks (28 rank)
                for j0 in [0, 2]:
                    for k0 in [0, 2]:
                        # M1
                        U[0*m + j0][col] += 1; U[1*m + (j0+1)][col] += 1
                        V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
                        W[k0*n + 0][col] += 1; W[(k0+1)*n + 1][col] += 1
                        col += 1
                        # M2
                        U[1*m + j0][col] += 1; U[1*m + (j0+1)][col] += 1
                        V[j0*p + k0][col] += 1
                        W[k0*n + 1][col] += 1; W[(k0+1)*n + 1][col] -= 1
                        col += 1
                        # M3
                        U[0*m + j0][col] += 1
                        V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
                        W[(k0+1)*n + 0][col] += 1; W[(k0+1)*n + 1][col] += 1
                        col += 1
                        # M4
                        U[1*m + (j0+1)][col] += 1
                        V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
                        W[k0*n + 0][col] += 1; W[k0*n + 1][col] += 1
                        col += 1
                        # M5
                        U[0*m + j0][col] += 1; U[0*m + (j0+1)][col] += 1
                        V[(j0+1)*p + (k0+1)][col] += 1
                        W[k0*n + 0][col] -= 1; W[(k0+1)*n + 0][col] += 1
                        col += 1
                        # M6
                        U[1*m + j0][col] += 1; U[0*m + j0][col] -= 1
                        V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
                        W[(k0+1)*n + 1][col] += 1
                        col += 1
                        # M7
                        U[0*m + (j0+1)][col] += 1; U[1*m + (j0+1)][col] -= 1
                        V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
                        W[k0*n + 0][col] += 1
                        col += 1
                # Add schoolbook for last column (k=4), 2x4x1 = 8 rank
                for di in range(2):
                    for dj in range(4):
                        U[di*m + dj][col] = 1
                        V[dj*p + 4][col] = 1
                        W[4*n + di][col] = 1
                        col += 1
                # Now we have 36 rank, truncate to R
                U_init = U_init[:, :R]
                V_init = V_init[:, :R]
                W_init = W_init[:, :R]
            else:
                # Random with different scales
                scale = 0.5 + 0.5 * np.random.random()
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'lm' for speed if possible, otherwise 'trf'
                result = opt.least_squares(residual, x0, args=(R,), method='lm', max_nfev=5000)
                
                if result.success or result.cost < 1e-6:
                    x_opt = result.x
                    size_U = n * m * R
                    size_V = m * p * R
                    size_W = p * n * R
                    U_opt = x_opt[0:size_U].reshape(n*m, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                    W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                    
                    # Check residual
                    if verify(U_opt, V_opt, W_opt, tol=1e-5):
                        # Try polishing for higher precision
                        U_pol, V_pol, W_pol = polish(U_opt, V_opt, W_opt, R)
                        
                        # Try snapping
                        U_snap, V_snap, W_snap = snap(U_pol, V_pol, W_pol)
                        if verify(U_snap, V_snap, W_snap, tol=1e-9):
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        else:
                            # Try snapping the unpolished version
                            U_snap2, V_snap2, W_snap2 = snap(U_opt, V_opt, W_opt)
                            if verify(U_snap2, V_snap2, W_snap2, tol=1e-9):
                                return {"U": U_snap2.tolist(), "V": V_snap2.tolist(), "W": W_snap2.tolist()}
            except Exception:
                pass
        
    # Fallback
    U_snap, V_snap, W_snap = snap(U36, V36, W36)
    if verify(U_snap, V_snap, W_snap, tol=1e-9):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
