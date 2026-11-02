import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]
DEADLINE = 140.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2) -> C[0:2, 0:2]
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2) -> C[0:2, 0:2]
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2) -> C[0:2, 2:4]
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2) -> C[0:2, 2:4]
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W


def _verify_fast(U, V, W, tol=1e-9):
    """Fast vectorized verification of tensor decomposition."""
    n, m, p = 2, 4, 5
    u_rows = []
    v_rows = []
    w_rows = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_rows.append(i * m + j)
                v_rows.append(j * p + k)
                w_rows.append(k * n + i)
    
    u_rows = np.array(u_rows)
    v_rows = np.array(v_rows)
    w_rows = np.array(w_rows)
    
    U_sel = U[u_rows, :]
    V_sel = V[v_rows, :]
    W_sel = W[w_rows, :]
    
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    return np.all(np.abs(vals - 1.0) < tol)


def _tensor_contribution(U, V, W, cols):
    """Compute the tensor contribution of specific columns."""
    n, m, p = 2, 4, 5
    u_rows = []
    v_rows = []
    w_rows = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_rows.append(i * m + j)
                v_rows.append(j * p + k)
                w_rows.append(k * n + i)
    
    u_rows = np.array(u_rows)
    v_rows = np.array(v_rows)
    w_rows = np.array(w_rows)
    
    U_sel = U[u_rows, cols]
    V_sel = V[v_rows, cols]
    W_sel = W[w_rows, cols]
    
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    return vals


def _greedy_reduce(U, V, W, n, m, p, max_iterations=80, time_limit=25.0):
    """Greedy column removal: try single and pair removals."""
    start_time = time.time()
    rank = U.shape[1]
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        progress = False
        
        # Try single column removal
        for col in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            mask = [i for i in range(rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify_fast(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank = U.shape[1]
                progress = True
                break
        
        if progress:
            continue
        
        # Try pair column removal (structured merge search)
        found_pair = False
        for c1 in range(rank):
            if time.time() - start_time > time_limit:
                break
            for c2 in range(c1 + 1, rank):
                if time.time() - start_time > time_limit:
                    break
                
                mask = [i for i in range(rank) if i != c1 and i != c2]
                if len(mask) == 0:
                    continue
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                
                if _verify_fast(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    progress = True
                    found_pair = True
                    break
            if found_pair:
                break
        
        if not progress:
            break
    
    return U, V, W


def _structured_merge_search(U, V, W, n, m, p, time_limit=30.0):
    """
    Structured merge: try to replace pairs of columns with a single column.
    For each pair (c1, c2), check if the tensor contribution of both columns
    can be represented by a single new column.
    """
    start_time = time.time()
    rank = U.shape[1]
    
    # Precompute the full tensor target
    target = np.ones(n * m * p)
    
    # For each pair of columns, try to find a single column that replaces them
    for c1 in range(rank):
        if time.time() - start_time > time_limit:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() - start_time > time_limit:
                break
            
            # Compute what the remaining columns give
            mask = [i for i in range(rank) if i != c1 and i != c2]
            if len(mask) == 0:
                continue
            
            U_rem = U[:, mask]
            V_rem = V[:, mask]
            W_rem = W[:, mask]
            
            # Contribution of remaining columns
            u_rows = []
            v_rows = []
            w_rows = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_rows.append(i * m + j)
                        v_rows.append(j * p + k)
                        w_rows.append(k * n + i)
            
            u_rows = np.array(u_rows)
            v_rows = np.array(v_rows)
            w_rows = np.array(w_rows)
            
            U_sel = U_rem[u_rows, :]
            V_sel = V_rem[v_rows, :]
            W_sel = W_rem[w_rows, :]
            rem_vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            
            # What we need from the new single column
            needed = target - rem_vals
            
            # Check if 'needed' is rank-1 in the tensor sense
            # i.e., if there exist vectors u, v, w such that
            # u[i*m+j] * v[j*p+k] * w[k*n+i] = needed[i,j,k] for all i,j,k
            # This is a hard problem, but we can check a necessary condition:
            # The tensor 'needed' must be rank 1.
            # A quick check: if any entry is 0, all entries sharing an index must be 0.
            
            needed_tensor = needed.reshape(n, m, p)
            
            # Quick rank-1 check
            is_rank1 = True
            nonzero_indices = np.argwhere(needed_tensor != 0)
            
            if len(nonzero_indices) == 0:
                # All zero - this pair is redundant
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_fast(U_new, V_new, W_new):
                    return U_new, V_new, W_new
                continue
            
            if len(nonzero_indices) == 1:
                # Single nonzero - could be rank 1
                pass
            else:
                # Check if all nonzeros share the same i, j, or k pattern
                # For rank 1: needed[i,j,k] = a[i]*b[j]*c[k]
                # If needed[i1,j1,k1] != 0 and needed[i1,j1,k2] != 0, then c[k1] and c[k2] both nonzero
                # This means for any i',j', needed[i',j',k1] and needed[i',j',k2] must have same sign pattern
                # Simple check: the tensor should be expressible as outer product
                # Try: fix first nonzero, derive a, b, c, then verify
                i0, j0, k0 = nonzero_indices[0]
                val0 = needed_tensor[i0, j0, k0]
                
                # Try to derive
                a = np.zeros(n)
                b = np.zeros(m)
                c = np.zeros(p)
                
                # Set a[i0]=1, then b[j0]=val0/c[k0], but we don't know c[k0]
                # Instead: a[i0]*b[j0]*c[k0] = val0
                # For another nonzero at (i0, j1, k0): a[i0]*b[j1]*c[k0] = val1
                #   => b[j1]/b[j0] = val1/val0
                # For (i1, j0, k0): a[i1]*b[j0]*c[k0] = val2
                #   => a[i1]/a[i0] = val2/val0
                # For (i0, j0, k1): a[i0]*b[j0]*c[k1] = val3
                #   => c[k1]/c[k0] = val3/val0
                
                # This is getting complex. Let's just do a numerical check.
                # Reshape needed into a matrix and check rank
                # Flatten (i,j) vs k
                mat1 = needed_tensor.reshape(n*m, p)
                r1 = np.linalg.matrix_rank(mat1, tol=1e-10)
                if r1 > 1:
                    is_rank1 = False
                else:
                    mat2 = needed_tensor.reshape(n, m*p)
                    r2 = np.linalg.matrix_rank(mat2, tol=1e-10)
                    if r2 > 1:
                        is_rank1 = False
            
            if is_rank1:
                # Try to construct the rank-1 decomposition
                # needed[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                # This is not a standard outer product due to the index coupling
                # We need to solve for u, v, w
                # This is a nonlinear system. Let's use a simple approach:
                # If the tensor is rank 1 in the "matricized" sense, we can try
                # to find u, v, w via alternating least squares.
                
                # Initialize
                u = np.ones(n*m)
                v = np.ones(m*p)
                w = np.ones(p*n)
                
                # Alternating least squares
                for _iter in range(50):
                    # Update u: for each (i,j), u[i*m+j] = sum over k of needed[i,j,k] * v[j*p+k] * w[k*n+i] / sum over k of (v[j*p+k]*w[k*n+i])**2
                    for i in range(n):
                        for j in range(m):
                            num = 0.0
                            den = 0.0
                            for k in range(p):
                                num += needed_tensor[i,j,k] * v[j*p+k] * w[k*n+i]
                                den += (v[j*p+k] * w[k*n+i]) ** 2
                            if den > 1e-15:
                                u[i*m+j] = num / den
                    
                    # Update v: for each (j,k), v[j*p+k] = sum over i of needed[i,j,k] * u[i*m+j] * w[k*n+i] / sum over i of (u[i*m+j]*w[k*n+i])**2
                    for j in range(m):
                        for k in range(p):
                            num = 0.0
                            den = 0.0
                            for i in range(n):
                                num += needed_tensor[i,j,k] * u[i*m+j] * w[k*n+i]
                                den += (u[i*m+j] * w[k*n+i]) ** 2
                            if den > 1e-15:
                                v[j*p+k] = num / den
                    
                    # Update w: for each (k,i), w[k*n+i] = sum over j of needed[i,j,k] * u[i*m+j] * v[j*p+k] / sum over j of (u[i*m+j]*v[j*p+k])**2
                    for k in range(p):
                        for i in range(n):
                            num = 0.0
                            den = 0.0
                            for j in range(m):
                                num += needed_tensor[i,j,k] * u[i*m+j] * v[j*p+k]
                                den += (u[i*m+j] * v[j*p+k]) ** 2
                            if den > 1e-15:
                                w[k*n+i] = num / den
                
                # Verify
                test_vals = np.zeros(n*m*p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            test_vals[i*m*p + j*p + k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                
                if np.allclose(test_vals, needed, atol=1e-8):
                    # Create new matrix with the merged column
                    U_new = np.zeros((n*m, rank - 1))
                    V_new = np.zeros((m*p, rank - 1))
                    W_new = np.zeros((p*n, rank - 1))
                    
                    new_col = 0
                    for orig_col in range(rank):
                        if orig_col == c1 or orig_col == c2:
                            continue
                        U_new[:, new_col] = U[:, orig_col]
                        V_new[:, new_col] = V[:, orig_col]
                        W_new[:, new_col] = W[:, orig_col]
                        new_col += 1
                    
                    U_new[:, new_col] = u
                    V_new[:, new_col] = v
                    W_new[:, new_col] = w
                    
                    if _verify_fast(U_new, V_new, W_new, tol=1e-8):
                        return U_new, V_new, W_new
    
    return U, V, W


def _optimize_rank(R, U_init, V_init, W_init, n, m, p, time_limit=8.0, max_restarts=10):
    """Try to find a valid decomposition of rank R using optimization."""
    import scipy.optimize as opt
    
    start_time = time.time()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    best_x = None
    best_err = np.inf
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            x = x0.copy()
        else:
            x = x0 + np.random.normal(0, 0.05, x0.shape)
        
        try:
            def _err(x_flat):
                Uf = x_flat[:n*m*R].reshape(n*m, R)
                Vf = x_flat[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                Wf = x_flat[n*m*R + m*p*R : ].reshape(p*n, R)
                
                u_rows = []
                v_rows = []
                w_rows = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_rows.append(i * m + j)
                            v_rows.append(j * p + k)
                            w_rows.append(k * n + i)
                
                u_rows = np.array(u_rows)
                v_rows = np.array(v_rows)
                w_rows = np.array(w_rows)
                
                U_sel = Uf[u_rows, :]
                V_sel = Vf[v_rows, :]
                W_sel = Wf[w_rows, :]
                
                vals = np.sum(U_sel * V_sel * W_sel, axis=1)
                return np.sum((vals - 1.0)**2)
            
            res = opt.minimize(
                _err, x,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
            
            if res.fun < 1e-10:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                if _verify_fast(U, V, W, tol=1e-9):
                    return U, V, W
        except Exception:
            pass
    
    if best_x is not None and best_err < 1e-8:
        U = best_x[:n*m*R].reshape(n*m, R)
        V = best_x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = best_x[n*m*R + m*p*R : ].reshape(p*n, R)
        if _verify_fast(U, V, W, tol=1e-8):
            return U, V, W
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = 2, 4, 5
    
    # Start with rank-36 Strassen decomposition
    U, V, W = _build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 36
    
    # Phase 1: Greedy column removal (singles + pairs)
    if time.time() < deadline - 40:
        U, V, W = _greedy_reduce(U, V, W, n, m, p, max_iterations=60, time_limit=15.0)
        if U.shape[1] < best_rank and _verify_fast(U, V, W):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Phase 2: Structured merge search (try to fold pairs into singles)
    if time.time() < deadline - 30:
        U, V, W = _structured_merge_search(U, V, W, n, m, p, time_limit=20.0)
        if U.shape[1] < best_rank and _verify_fast(U, V, W):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Phase 3: Optimization-based rank reduction
    for target_rank in range(best_rank - 1, max(19, best_rank - 8), -1):
        if time.time() > deadline - 15:
            break
        
        time_limit = min(8.0, max(2.0, (deadline - time.time() - 10) / 3))
        
        result = _optimize_rank(target_rank, best_U[:, :target_rank], best_V[:, :target_rank], 
                                best_W[:, :target_rank], n, m, p, time_limit=time_limit, 
                                max_restarts=6)
        
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
    
    # Final verification
    if not _verify_fast(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
