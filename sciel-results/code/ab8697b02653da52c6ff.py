import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Start with the known rank-36 solution (Strassen blocks + schoolbook).
    2. Attempt to reduce the rank to 35, 34, etc. using vectorized least-squares optimization.
    3. Verify exactness with integer/half-integer snapping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Precompute constraint structure for speed
    # Constraint for (i, k, a, b, c, d):
    # sum_mu U[a, b, mu] V[c, d, mu] W[k, i, mu] == delta[b, c] * delta[a, i] * delta[d, k]
    
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
    # Precompute constraint arrays for vectorization
    cons_i = np.array([c[0] for c in cons_list])
    cons_k = np.array([c[1] for c in cons_list])
    cons_a = np.array([c[2] for c in cons_list])
    cons_b = np.array([c[3] for c in cons_list])
    cons_c = np.array([c[4] for c in cons_list])
    cons_d = np.array([c[5] for c in cons_list])
    cons_target = np.array([c[6] for c in cons_list])
    
    num_cons = len(cons_list)
    
    def get_initial_solution_rank36():
        # Reconstruct the known rank 36 solution
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0, col_start):
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
            return col_start + 7

        def add_schoolbook_2x4x1(i0, j0, k0, col_start):
            c = col_start
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][c] = 1
                    V[(j0 + dj)*p + k0][c] = 1
                    W[k0*n + (i0 + di)][c] = 1
                    c += 1
            return c

        col = add_strassen_222(0, 0, 0, 0)
        col = add_strassen_222(0, 2, 0, col)
        col = add_strassen_222(0, 0, 2, col)
        col = add_strassen_222(0, 2, 2, col)
        col = add_schoolbook_2x4x1(0, 0, 4, col)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    # Helper to verify a solution using vectorized computation
    def verify_solution(U, V, W, tol=1e-9):
        # Check if U, V, W satisfy the tensor equations exactly (with tolerance)
        R = U.shape[1]
        # For each constraint, compute sum_mu U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
        # Vectorize over constraints
        U_idx = cons_a * m + cons_b  # shape (num_cons,)
        V_idx = cons_c * p + cons_d  # shape (num_cons,)
        W_idx = cons_k * n + cons_i  # shape (num_cons,)
        
        # U[U_idx] has shape (num_cons, R)
        # V[V_idx] has shape (num_cons, R)
        # W[W_idx] has shape (num_cons, R)
        # For each constraint i: sum_mu U[i, mu] * V[i, mu] * W[i, mu]
        vals = np.sum(U[U_idx] * V[V_idx] * W[W_idx], axis=1)
        
        return np.all(np.abs(vals - cons_target) < tol)

    # Helper to check if entries are close to half-integers
    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    # Vectorized residual function for optimization
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
        
        # Vectorized computation of all residuals
        U_idx = cons_a * m + cons_b  # shape (num_cons,)
        V_idx = cons_c * p + cons_d  # shape (num_cons,)
        W_idx = cons_k * n + cons_i  # shape (num_cons,)
        
        # U[U_idx] has shape (num_cons, R)
        # V[V_idx] has shape (num_cons, R)
        # W[W_idx] has shape (num_cons, R)
        # For each constraint i: sum_mu U[i, mu] * V[i, mu] * W[i, mu]
        vals = np.sum(U[U_idx] * V[V_idx] * W[W_idx], axis=1)
        
        return vals - cons_target

    start_time = time.time()
    time_limit = 110 # Leave margin
    
    # Try to find lower ranks
    best_solution = None
    best_rank = 36
    
    for R in range(35, 25, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations with more attempts
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize x
            # Scale: values in rank 36 are -1, 0, 1.
            # Random initialization with small values might work better for finding new structures
            # Or truncate the rank 36 solution
            if attempt == 0:
                # Truncate rank 36 solution
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempt == 1:
                # Scaled random initialization
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 2:
                # Another random initialization with different scale
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            else:
                # Random with varying scales
                scale = 0.3 * (1 + attempt/10)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use 'lm' for small problems, fall back to 'trf' if needed
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=30000)
                if not result.success:
                    # Try with 'trf' method as fallback
                    result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='trf', max_nfev=30000)
                
                x_opt = result.x
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                
                # Check residual
                res = residual_vector_fast(x_opt, R)
                max_res = np.max(np.abs(res))
                
                if max_res < 1e-6:
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify_solution(U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                elif max_res < 1e-3:
                    # Good but not perfect - try to polish
                    # Use the optimized solution as starting point for a more careful optimization
                    try:
                        result2 = opt.least_squares(residual_vector_fast, x_opt, args=(R,), method='lm', max_nfev=50000)
                        x_opt2 = result2.x
                        U_opt2 = x_opt2[0:size_U].reshape(n*m, R)
                        V_opt2 = x_opt2[size_U:size_U+size_V].reshape(m*p, R)
                        W_opt2 = x_opt2[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
                        
                        res2 = residual_vector_fast(x_opt2, R)
                        if np.max(np.abs(res2)) < 1e-8:
                            U_snap2, V_snap2, W_snap2 = snap_to_half_integers(U_opt2, V_opt2, W_opt2)
                            if verify_solution(U_snap2, V_snap2, W_snap2):
                                return {"U": U_snap2.tolist(), "V": V_snap2.tolist(), "W": W_snap2.tolist()}
                    except:
                        pass
            except Exception as e:
                pass

    # If no better rank found, return the rank 36 solution (snapped)
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return original
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
