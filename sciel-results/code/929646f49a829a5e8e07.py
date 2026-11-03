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
        
        attempts = 0
        max_attempts = 10 # Adjust based on time
        
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
            else:
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'lm' for speed if possible, otherwise 'trf'
                # 'lm' is faster for small problems but doesn't support bounds.
                # We don't need bounds.
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
                        # Try snapping
                        U_snap, V_snap, W_snap = snap(U_opt, V_opt, W_opt)
                        if verify(U_snap, V_snap, W_snap, tol=1e-9):
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        else:
                            # Maybe it's exact but not snapped?
                            # The problem requires integer/half-integer entries.
                            # If verify passes with float, we must return float?
                            # "integer/half-integer entries" is a requirement.
                            # So if snap fails, we cannot return this solution unless we find a way to make it snap.
                            # But if the residual is small, snapping should work if the true solution is on the grid.
                            pass
            except Exception:
                pass
        
    # Fallback
    U_snap, V_snap, W_snap = snap(U36, V36, W36)
    if verify(U_snap, V_snap, W_snap, tol=1e-9):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
