import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _snap_to_half_integer(mat, tol=1e-8):
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped
    return arr

def _build_strassen_44(n, m, p):
    """
    Construct the rank-44 decomposition.
    Split A into row 0 (1x4) and rows 1-2 (2x4).
    Part 1: 1x4 * 4x4 -> 1x4. Schoolbook rank 16.
    Part 2: 2x4 * 4x4 -> 2x4. Split B into four 2x2 blocks. Use Strassen (rank 7) for each block. 4*7=28.
    Total rank 44.
    """
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen 2x2 terms: (x_coeff, y_coeff, z_coeff)
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    
    # Part 1: Row 0 schoolbook
    # A[0, j] * B[j, k] -> C[0, k]
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: Rows 1-2 Strassen
    # Blocks: (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
    # A rows are 1, 2. B rows 0-3. C rows 1, 2.
    blocks = [
        (1, 0, 0, 0, 1, 0), # A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
        (1, 0, 0, 2, 1, 2), # A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
        (1, 2, 2, 0, 1, 0), # A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
        (1, 2, 2, 2, 1, 2), # A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    ]
    
    for (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off) in blocks:
        for (x_coeff, y_coeff, z_coeff) in strassen_steps:
            for (xi, xj), val in x_coeff.items():
                U[(a_row_off + xi) * m + (a_col_off + xj), col] += val
            for (yi, yj), val in y_coeff.items():
                V[(b_row_off + yi) * p + (b_col_off + yj), col] += val
            for (zi, zj), val in z_coeff.items():
                # W index: k*n + i. k is col of C (c_col_off + zj), i is row of C (c_row_off + zi)
                W[(c_col_off + zj) * n + (c_row_off + zi), col] += val
            col += 1
    
    return U, V, W

def _try_column_elimination(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among columns to reduce rank."""
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    eliminated = set()
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        current_R = best_U.shape[1]
        
        for c in range(current_R):
            if time.time() > deadline - 10:
                break
            if c in eliminated:
                continue
            
            others = [r for r in range(current_R) if r != c and r not in eliminated]
            if len(others) < 1:
                continue
            
            U_others = best_U[:, others]
            V_others = best_V[:, others]
            W_others = best_W[:, others]
            
            U_c = best_U[:, c]
            V_c = best_V[:, c]
            W_c = best_W[:, c]
            
            A_sys = np.vstack([U_others, V_others, W_others])
            b_sys = np.concatenate([U_c, V_c, W_c])
            
            try:
                alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-10)
            except:
                continue
            
            residual = np.linalg.norm(A_sys @ alpha - b_sys)
            if residual < 1e-9:
                keep = [r for r in range(current_R) if r != c and r not in eliminated]
                best_U = best_U[:, keep]
                best_V = best_V[:, keep]
                best_W = best_W[:, keep]
                eliminated.add(c)
                changed = True
                best_rank = best_U.shape[1]
                break
    
    if best_rank < R:
        return best_U, best_V, best_W
    return None

def _try_structured_reduction(U, V, W, n, m, p, deadline):
    """Try structured reductions by checking if certain column patterns allow merging."""
    R = U.shape[1]
    best_result = None
    best_rank = R
    
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 15:
            break
        
        c1 = np.random.randint(0, R)
        c2 = np.random.randint(0, R)
        if c1 == c2:
            continue
        
        for sign in [1.0, -1.0]:
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            U_try[:, c1] = U[:, c1] + sign * U[:, c2]
            V_try[:, c1] = V[:, c1] + sign * V[:, c2]
            W_try[:, c1] = W[:, c1] + sign * W[:, c2]
            
            keep = [r for r in range(R) if r != c2]
            U_try = U_try[:, keep]
            V_try = V_try[:, keep]
            W_try = W_try[:, keep]
            
            ok, err = _verify(U_try, V_try, W_try, n, m, p)
            if ok and U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_result = (U_try, V_try, W_try)
    
    if best_result is not None:
        return best_result
    return None

def _try_rank_research(U, V, W, n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by solving the tensor decomposition problem.
    """
    R_current = U.shape[1]
    best_rank = R_current
    best_result = None
    
    # Try ranks from R_current-1 down to 30
    for target_rank in range(R_current - 1, 29, -1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values
            np.random.seed(restart * 1000 + target_rank)
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            W_rand = np.random.randn(p * n, target_rank) * 0.5
            
            # Use scipy optimization if available
            try:
                from scipy.optimize import minimize
            except ImportError:
                continue
            
            def objective(params):
                U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_p = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U3 = U_p.reshape(n, m, target_rank)
                V3 = V_p.reshape(m, p, target_rank)
                W3 = W_p.reshape(p, n, target_rank)
                
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
                err = np.sum((T - 1.0)**2)
                return err
            
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < 1e-8:
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                ok, err = _verify(U_sol, V_sol, W_sol, n, m, p)
                if ok:
                    best_rank = target_rank
                    best_result = (U_sol, V_sol, W_sol)
                    break  # Found valid decomposition at this rank
    
    if best_result is not None:
        return best_result
    return None

def _try_pauli_based(n, m, p, deadline):
    """
    Try a decomposition based on Pauli matrix structure.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    n, m, p = DIMENSIONS
    
    # Start with the rank-44 Strassen decomposition
    U_arr, V_arr, W_arr = _build_strassen_44(n, m, p)
    
    # Verify the initial decomposition
    ok, err = _verify(U_arr, V_arr, W_arr, n, m, p)
    if not ok:
        # Fallback to schoolbook
        rank = n * m * p
        U_arr = np.zeros((n * m, rank))
        V_arr = np.zeros((m * p, rank))
        W_arr = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_arr[i * m + j, col] = 1.0
                    V_arr[j * p + k, col] = 1.0
                    W_arr[k * n + i, col] = 1.0
                    col += 1
    
    best_rank = U_arr.shape[1]
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Attempt 1: Column elimination via linear dependency detection
    if time.time() < deadline - 60:
        elim_result = _try_column_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if elim_result is not None:
            eU, eV, eW = elim_result
            if eU.shape[1] < best_rank:
                best_rank = eU.shape[1]
                best_U, best_V, best_W = eU, eV, eW
    
    # Attempt 2: Structured pair merging
    if time.time() < deadline - 40:
        struct_result = _try_structured_reduction(best_U, best_V, best_W, n, m, p, deadline)
        if struct_result is not None:
            sU, sV, sW = struct_result
            if sU.shape[1] < best_rank:
                best_rank = sU.shape[1]
                best_U, best_V, best_W = sU, sV, sW
    
    # Attempt 3: Rank research via optimization
    if time.time() < deadline - 20 and best_rank > 30:
        research_result = _try_rank_research(best_U, best_V, best_W, n, m, p, deadline)
        if research_result is not None:
            rU, rV, rW = research_result
            if rU.shape[1] < best_rank:
                best_rank = rU.shape[1]
                best_U, best_V, best_W = rU, rV, rW
    
    # Final: snap to half-integers and verify
    U_final = _snap_to_half_integer(best_U)
    V_final = _snap_to_half_integer(best_V)
    W_final = _snap_to_half_integer(best_W)
    
    ok, err = _verify(U_final, V_final, W_final, n, m, p)
    if not ok:
        # Fallback to the unsnapped version if it's better
        ok2, err2 = _verify(best_U, best_V, best_W, n, m, p)
        if ok2:
            U_final = best_U
            V_final = best_V
            W_final = best_W
        else:
            # Ultimate fallback to schoolbook
            rank = n * m * p
            U_final = np.zeros((n * m, rank))
            V_final = np.zeros((m * p, rank))
            W_final = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_final[i * m + j, col] = 1.0
                        V_final[j * p + k, col] = 1.0
                        W_final[k * n + i, col] = 1.0
                        col += 1
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
