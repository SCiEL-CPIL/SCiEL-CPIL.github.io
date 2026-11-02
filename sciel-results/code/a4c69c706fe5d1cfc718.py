import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    time_limit = 115
    
    def elapsed():
        return time.time() - start_time
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                val = np.dot(U[a*m+b], V[c*p+d] * W[k*n+i])
                                target = 1.0 if (b == c and a == i and d == k) else 0.0
                                if abs(val - target) > tol:
                                    return False
        return True
    
    def snap_to_half_integers(U, V, W):
        return (np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0)
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2: (A10+A11)B00
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3: A00(B10-B11)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5: (A00+A01)B11
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1
        
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return U, V, W
    
    U36, V36, W36 = get_initial_solution_rank36()
    
    if not verify_solution(U36, V36, W36):
        # Fallback: schoolbook
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap):
        best_rank = 36
        best_solution = (U_snap, V_snap, W_snap)
    else:
        best_rank = 36
        best_solution = (U36, V36, W36)
    
    # Precompute constraint indices for fast residual
    cons_idx = []
    cons_target = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_idx.append((a*m+b, c*p+d, k*n+i))
                            cons_target.append(target)
    
    cons_idx = np.array(cons_idx)
    cons_target = np.array(cons_target)
    
    def residual_vector_fast(x, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U+size_V].reshape(mp, R)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(pn, R)
        
        # Vectorized computation
        U_rows = U[cons_idx[:, 0]]
        V_rows = V[cons_idx[:, 1]]
        W_rows = W[cons_idx[:, 2]]
        
        vals = np.einsum('ij,ij,ij->i', U_rows, V_rows, W_rows)
        return vals - cons_target
    
    # Try to reduce rank by merging columns
    # Strategy: for each pair of columns, try to replace them with a single column
    # that best approximates their combined contribution
    
    for R in range(35, 25, -1):
        if elapsed() > time_limit:
            break
        
        # Try multiple strategies
        strategies = 0
        max_strategies = 5
        
        # Strategy 1: Truncate and optimize
        if elapsed() < time_limit:
            x0 = np.concatenate([U36[:, :R].flatten(), V36[:, :R].flatten(), W36[:, :R].flatten()])
            try:
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=3000)
                if np.max(np.abs(result.fun)) < 1e-8:
                    x_opt = result.x
                    U_opt = x_opt[:nm*R].reshape(nm, R)
                    V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = x_opt[nm*R+mp*R:].reshape(pn, R)
                    U_s, V_s, W_s = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_s, V_s, W_s):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            except:
                pass
        
        # Strategy 2: Random initialization
        for attempt in range(max_strategies):
            if elapsed() > time_limit:
                break
            
            U_init = np.random.normal(0, 1.0, (nm, R))
            V_init = np.random.normal(0, 1.0, (mp, R))
            W_init = np.random.normal(0, 1.0, (pn, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=5000)
                if np.max(np.abs(result.fun)) < 1e-8:
                    x_opt = result.x
                    U_opt = x_opt[:nm*R].reshape(nm, R)
                    V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = x_opt[nm*R+mp*R:].reshape(pn, R)
                    U_s, V_s, W_s = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_s, V_s, W_s):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            except:
                pass
        
        strategies += 1
    
    return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
