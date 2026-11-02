"""Combined program: block Strassen + structured optimization + exact polish."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4

# EVOLVE-BLOCK-START

def _verify_einsum(U, V, W, n=N, m=M, p=P, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def _verify_strict(U, V, W, n=N, m=M, p=P, tol=1e-12):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def schoolbook(na, mb, pb):
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _add_sub(U, V, W, U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start, n, m, p):
    r = len(U_sub[0])
    for c in range(r):
        col = col_start + c
        for i_prime in range(na):
            for j_prime in range(mb):
                u_val = U_sub[i_prime*mb + j_prime][c]
                if u_val != 0:
                    U[(i_off + i_prime)*m + (j_off + j_prime)][col] += u_val
        for j_prime in range(mb):
            for k_prime in range(pb):
                v_val = V_sub[j_prime*pb + k_prime][c]
                if v_val != 0:
                    V[(j_off + j_prime)*p + (k_off + k_prime)][col] += v_val
        for k_prime in range(pb):
            for i_prime in range(na):
                w_val = W_sub[k_prime*na + i_prime][c]
                if w_val != 0:
                    W[(k_off + k_prime)*n + (i_off + i_prime)][col] += w_val


def _build_2plus1_split():
    """Split rows as 2+1, columns as 2+2."""
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for k_off in [0, 2]:
        for j_off in [0, 2]:
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U, V, W, Us, Vs, Ws, 2, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4
    return U, V, W


def _build_1plus2_split():
    """Split rows as 1+2, columns as 2+2."""
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for k_off in [0, 2]:
        for j_off in [0, 2]:
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 1, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
    return U, V, W


def _build_3x4x4_strassen_like():
    """
    Alternative block decomposition: 
    Split A (3x4) into [A_top (2x4); A_bot (1x4)]
    Split B (4x4) into [B_left (4x2) B_right (4x2)]
    
    C_top = A_top * B: 2x4 * 4x4 = 2x4
    C_bot = A_bot * B: 1x4 * 4x4 = 1x4
    
    For C_top: use Strassen on 2x2 blocks
    For C_bot: schoolbook (rank 4)
    """
    n, m, p = N, M, P
    # C_top is 2x4, computed as 2x4 * 4x4
    # Split into four 2x2 * 2x2 blocks: rank 7 each = 28
    # C_bot is 1x4, computed as 1x4 * 4x4: rank 4
    # Total: 28 + 4 = 32
    
    rank_total = 28 + 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    # Top 2 rows: 2x4 * 4x4
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
    
    # Bottom row: 1x4 * 4x4
    Us, Vs, Ws = schoolbook(1, 4, 4)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 0, 0, 1, 4, 4, col, n, m, p)
    col += 4
    
    return U, V, W


def _try_rank_reduction_via_merging(U, V, W, n, m, p, max_time=15):
    """Try to reduce rank by removing columns that don't affect the result."""
    start = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    if r > 48:
        return None
    best_rank = r
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column one at a time
    changed = True
    while changed and (time.time() - start) < max_time:
        changed = False
        for col_idx in range(U.shape[1] - 1, -1, -1):
            if time.time() - start > max_time:
                break
            U_test = np.delete(U, col_idx, axis=1)
            V_test = np.delete(V, col_idx, axis=1)
            W_test = np.delete(W, col_idx, axis=1)
            if _verify_einsum(U_test, V_test, W_test, n, m, p, tol=1e-10):
                U, V, W = U_test, V_test, W_test
                changed = True
                break
    
    if W.shape[1] < r:
        return U.tolist(), V.tolist(), W.tolist()
    return None


def _optimize_rank(U_init, V_init, W_init, n, m, p, r, max_iter=500, ftol=1e-14, gtol=1e-12):
    """Optimize a given rank-r decomposition starting from initial guess."""
    from scipy.optimize import minimize
    
    nm, mp, pn = n*m, m*p, p*n
    target = np.ones((n, m, p))
    
    U_flat = np.array(U_init, dtype=np.float64).flatten()
    V_flat = np.array(V_init, dtype=np.float64).flatten()
    W_flat = np.array(W_init, dtype=np.float64).flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def loss_func(x):
        U_f = x[:nm*r]
        V_f = x[nm*r:nm*r+mp*r]
        W_f = x[nm*r+mp*r:]
        
        U = U_f.reshape(nm, r)
        V = V_f.reshape(mp, r)
        W = W_f.reshape(pn, r)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        loss = np.sum((recon - target) ** 2)
        return loss
    
    try:
        res = minimize(loss_func, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                'maxls': 50, 'maxcor': 20})
        return res
    except Exception:
        return None


def _coordinate_descent_polish(U, V, W, n, m, p, deadline, max_iters=100):
    """
    Coordinate descent polish: alternate between optimizing U, V, W 
    with the others fixed. This is faster than full L-BFGS-B for polishing.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Optimize U with V, W fixed
        # For each (i,j), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in U[i*m+j,:] for fixed V, W
        # Build system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        # So: U[i*m+j,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
        # Let M[i,j,k] = V[j*p+k,:] * W[k*n+i,:] (shape r)
        # Then U[i*m+j,:] @ M[:,k] = 1 for all k
        # This is: M^T @ U[i*m+j,:]^T = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build M matrix: p x r
                M_mat = np.zeros((p, r))
                for k in range(p):
                    M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                
                # Solve M_mat^T @ u = ones(p) for u (shape r)
                # Least squares if overdetermined
                try:
                    u_sol, residuals, rank_m, sv = np.linalg.lstsq(M_mat.T, np.ones(p), rcond=1e-15)
                    U[i*m+j, :] = u_sol
                except:
                    pass
        
        # Optimize V with U, W fixed
        for j in range(m):
            for k in range(p):
                M_mat = np.zeros((n, r))
                for i in range(n):
                    M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                
                try:
                    v_sol, residuals, rank_m, sv = np.linalg.lstsq(M_mat.T, np.ones(n), rcond=1e-15)
                    V[j*p+k, :] = v_sol
                except:
                    pass
        
        # Optimize W with U, V fixed
        for k in range(p):
            for i in range(n):
                M_mat = np.zeros((m, r))
                for j in range(m):
                    M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                try:
                    w_sol, residuals, rank_m, sv = np.linalg.lstsq(M_mat.T, np.ones(m), rcond=1e-15)
                    W[k*n+i, :] = w_sol
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            if _verify_strict(U, V, W, n, m, p, tol=1e-10):
                break
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_optimization_for_rank(r, n, m, p, deadline, n_restarts=15):
    """Try to find a valid rank-r decomposition via optimization with structured inits."""
    nm, mp, pn = n*m, m*p, p*n
    target = np.ones((n, m, p))
    
    best_loss = np.inf
    best_x = None
    
    # Generate structured initializations
    inits = []
    
    # 1. From block decompositions (truncated/padded to rank r)
    for build_fn in [_build_2plus1_split, _build_1plus2_split, _build_3x4x4_strassen_like]:
        try:
            U_b, V_b, W_b = build_fn()
            r_b = len(U_b[0])
            if r_b >= r:
                # Truncate to rank r
                inits.append((np.array(U_b[:, :r], dtype=np.float64),
                              np.array(V_b[:, :r], dtype=np.float64),
                              np.array(W_b[:, :r], dtype=np.float64)))
            else:
                # Pad with small random values
                U_pad = np.zeros((nm, r), dtype=np.float64)
                V_pad = np.zeros((mp, r), dtype=np.float64)
                W_pad = np.zeros((pn, r), dtype=np.float64)
                U_pad[:, :r_b] = np.array(U_b, dtype=np.float64)
                V_pad[:, :r_b] = np.array(V_b, dtype=np.float64)
                W_pad[:, :r_b] = np.array(W_b, dtype=np.float64)
                U_pad[:, r_b:] = np.random.randn(nm, r - r_b) * 0.01
                V_pad[:, r_b:] = np.random.randn(mp, r - r_b) * 0.01
                W_pad[:, r_b:] = np.random.randn(pn, r - r_b) * 0.01
                inits.append((U_pad, V_pad, W_pad))
        except:
            pass
    
    # 2. Random initializations with various scales
    for scale in [0.1, 0.5, 1.0]:
        U_init = np.random.randn(nm, r) * scale
        V_init = np.random.randn(mp, r) * scale
        W_init = np.random.randn(pn, r) * scale
        inits.append((U_init, V_init, W_init))
    
    # 3. Sparse random initializations
    for _ in range(3):
        U_init = np.random.randn(nm, r) * 0.5
        V_init = np.random.randn(mp, r) * 0.5
        W_init = np.random.randn(pn, r) * 0.5
        # Zero out ~50% of entries
        mask_u = np.random.rand(nm, r) < 0.5
        mask_v = np.random.rand(mp, r) < 0.5
        mask_w = np.random.rand(pn, r) < 0.5
        U_init[mask_u] = 0
        V_init[mask_v] = 0
        W_init[mask_w] = 0
        inits.append((U_init, V_init, W_init))
    
    for U_init, V_init, W_init in inits:
        if time.time() > deadline - 3:
            break
        
        res = _optimize_rank(U_init, V_init, W_init, n, m, p, r, max_iter=300)
        
        if res is not None and res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x.copy()
            
            if best_loss < 1e-10:
                U = best_x[:nm*r].reshape(nm, r)
                V = best_x[nm*r:nm*r+mp*r].reshape(mp, r)
                W = best_x[nm*r+mp*r:].reshape(pn, r)
                if _verify_strict(U, V, W, n, m, p, tol=1e-9):
                    # Polish with coordinate descent
                    U_p, V_p, W_p = _coordinate_descent_polish(U, V, W, n, m, p, deadline)
                    if _verify_strict(U_p, V_p, W_p, n, m, p, tol=1e-9):
                        return U_p, V_p, W_p, r
                    return U.tolist(), V.tolist(), W.tolist(), r
    
    return None


def _polish_solution(U, V, W, n, m, p, deadline, max_iter=500):
    """High-precision polish pass on a near-solution."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # First, try coordinate descent polish
    U_cd, V_cd, W_cd = _coordinate_descent_polish(U_arr, V_arr, W_arr, n, m, p, deadline, max_iters=50)
    
    if _verify_strict(U_cd, V_cd, W_cd, n, m, p, tol=1e-10):
        return U_cd, V_cd, W_cd
    
    # Then try L-BFGS-B polish
    for tol in [1e-8, 1e-10, 1e-12]:
        if time.time() > deadline - 5:
            break
        
        res = _optimize_rank(U_cd, V_cd, W_cd, n, m, p, r, 
                           max_iter=max_iter, ftol=tol*0.1, gtol=tol*0.01)
        
        if res is not None and res.fun < tol:
            U_arr = res.x[:nm*r].reshape(nm, r)
            V_arr = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W_arr = res.x[nm*r+mp*r:].reshape(pn, r)
            
            if _verify_strict(U_arr, V_arr, W_arr, n, m, p, tol=tol):
                return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
    
    return None


def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    # Start with the best block decomposition
    candidates = []
    
    for build_fn in [_build_2plus1_split, _build_1plus2_split, _build_3x4x4_strassen_like]:
        try:
            U, V, W = build_fn()
            if _verify_einsum(U, V, W, n, m, p):
                candidates.append((len(U[0]), U, V, W))
        except:
            pass
    
    if not candidates:
        # Fallback to schoolbook
        Us, Vs, Ws = schoolbook(n, m, p)
        return {"U": Us, "V": Vs, "W": Ws}
    
    # Sort by rank
    candidates.sort(key=lambda x: x[0])
    best_rank = candidates[0][0]
    best_result = (candidates[0][1], candidates[0][2], candidates[0][3])

    # Try to reduce rank via column merging on each candidate
    for rank_c, U_c, V_c, W_c in candidates:
        if time.time() > deadline - 20:
            break
        if rank_c >= best_rank:
            continue
            
        merged = _try_rank_reduction_via_merging(U_c, V_c, W_c, n, m, p, max_time=8)
        if merged is not None:
            U_m, V_m, W_m = merged
            if _verify_strict(U_m, V_m, W_m, n, m, p, tol=1e-10):
                new_rank = len(U_m[0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = (U_m, V_m, W_m)

    # Try optimization for lower ranks
    # Lower bound: for n x m * m x p, rank >= max(n*m, m*p, p*n) / max(n, m, p)
    # For 3x4*4x4: max(12, 16, 12) / 4 = 4 (very weak)
    # Better lower bound: rank >= n*m*p / max(n*m, m*p, p*n) = 48/16 = 3 (still weak)
    # Practical lower bound from known results: ~24-30
    
    for r in range(max(20, best_rank - 10), best_rank):
        if time.time() > deadline - 10:
            break
            
        result = _try_optimization_for_rank(r, n, m, p, deadline, n_restarts=10)
        if result is not None:
            U_opt, V_opt, W_opt, rank_opt = result
            if _verify_strict(U_opt, V_opt, W_opt, n, m, p, tol=1e-9):
                best_rank = rank_opt
                best_result = (U_opt, V_opt, W_opt)
                # Try to polish
                if time.time() < deadline - 5:
                    polished = _polish_solution(U_opt, V_opt, W_opt, n, m, p, deadline)
                    if polished is not None:
                        best_result = polished
                        if _verify_strict(best_result[0], best_result[1], best_result[2], n, m, p, tol=1e-10):
                            break  # Found a good solution, stop

    # Final polish on the best result
    if time.time() < deadline - 3:
        polished = _polish_solution(best_result[0], best_result[1], best_result[2], n, m, p, deadline)
        if polished is not None:
            best_result = polished

    # Final verification
    if _verify_strict(best_result[0], best_result[1], best_result[2], n, m, p, tol=1e-10):
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

    # Fallback: return the best we have
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

# EVOLVE-BLOCK-END
