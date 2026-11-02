import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Start with the known rank-36 solution.
    2. Use greedy column elimination: try removing each column and solve for the remaining.
    3. Use structured optimization with better initializations.
    4. Verify exactness with integer/half-integer snapping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Precompute constraint structure
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
    cons_i = np.array([c[0] for c in cons_list])
    cons_k = np.array([c[1] for c in cons_list])
    cons_a = np.array([c[2] for c in cons_list])
    cons_b = np.array([c[3] for c in cons_list])
    cons_c = np.array([c[4] for c in cons_list])
    cons_d = np.array([c[5] for c in cons_list])
    cons_target = np.array([c[6] for c in cons_list])
    
    num_cons = len(cons_list)
    
    # Precompute index arrays for vectorized operations
    U_idx = cons_a * m + cons_b
    V_idx = cons_c * p + cons_d
    W_idx = cons_k * n + cons_i
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
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

    U36, V36, W36 = get_initial_solution_rank36()
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        vals = np.sum(U[U_idx] * V[V_idx] * W[W_idx], axis=1)
        return np.all(np.abs(vals - cons_target) < tol)

    def snap_to_half_integers(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def residual_vector_fast(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        vals = np.sum(U[U_idx] * V[V_idx] * W[W_idx], axis=1)
        return vals - cons_target

    def try_column_elimination(U, V, W, cols_to_try, max_attempts=50):
        """Try to eliminate columns from a valid solution."""
        R = U.shape[1]
        if R <= 1:
            return None
        
        # Try removing each column individually
        for col_idx in cols_to_try:
            if len(cols_to_try) < 2:
                break
            
            # Remove column col_idx
            mask = np.ones(R, dtype=bool)
            mask[col_idx] = False
            U_reduced = U[:, mask]
            V_reduced = V[:, mask]
            W_reduced = W[:, mask]
            
            # Check if the reduced solution still satisfies constraints
            if verify_solution(U_reduced, V_reduced, W_reduced, tol=1e-8):
                # Try to further reduce by optimization
                R_new = U_reduced.shape[1]
                x0 = np.concatenate([U_reduced.flatten(), V_reduced.flatten(), W_reduced.flatten()])
                
                try:
                    result = opt.least_squares(residual_vector_fast, x0, args=(R_new,), method='lm', max_nfev=10000)
                    x_opt = result.x
                    size_U = n * m * R_new
                    size_V = m * p * R_new
                    U_opt = x_opt[0:size_U].reshape(n*m, R_new)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R_new)
                    W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R_new)
                    
                    res = residual_vector_fast(x_opt, R_new)
                    if np.max(np.abs(res)) < 1e-8:
                        U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                        if verify_solution(U_snap, V_snap, W_snap):
                            return U_snap, V_snap, W_snap
                except:
                    pass
                
                # If optimization didn't help, try the reduced solution directly
                U_snap, V_snap, W_snap = snap_to_half_integers(U_reduced, V_reduced, W_reduced)
                if verify_solution(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
        
        return None

    start_time = time.time()
    time_limit = 115  # Leave 35s margin for safety
    
    # Start with rank 36 and try to reduce
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    current_rank = 36
    
    # Greedy column elimination from rank 36
    for iteration in range(10):
        if time.time() - start_time > time_limit:
            break
        
        cols_to_try = list(range(current_rank))
        result = try_column_elimination(best_U, best_V, best_W, cols_to_try)
        
        if result is not None:
            best_U, best_V, best_W = result
            current_rank = best_U.shape[1]
        else:
            break
    
    # If we reduced the rank, verify and return
    if current_rank < 36:
        U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
        if verify_solution(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try optimization-based approach for lower ranks
    for R in range(min(current_rank - 1, 35), 25, -1):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                # Truncate from current best
                U_init = best_U[:, :R]
                V_init = best_V[:, :R]
                W_init = best_W[:, :R]
            else:
                # Structured random initialization
                scale = 0.5 + 0.1 * attempt
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=20000)
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                res = residual_vector_fast(x_opt, R)
                max_res = np.max(np.abs(res))
                
                if max_res < 1e-6:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            except:
                pass

    # Return the best solution found
    U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return original rank 36
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
