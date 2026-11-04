import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T
    # T[u, v, w] = 1 if u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    schoolbook_rank = n * m * p
    min_rank = max(n * p, m * p, n * m)
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    deadline = 145.0  # 5s margin

    def check_solution(U, V, W):
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T)
        return err < 1e-9

    def snap_to_half_int(U, V, W):
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        return U_int, V_int, W_int

    def polish_solution(U, V, W, max_iters=200):
        """Refine solution with alternating least squares until convergence."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update W
            A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
            T_W = T.reshape(n * m * m * p, p * n)
            W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
            W = W_T.T
            
            # Update U
            A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
            T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
            U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
            U = U_T.T
            
            # Update V
            A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
            T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
            V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
            V = V_T.T
            
            # Check convergence
            if it % 10 == 0:
                T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                err = np.linalg.norm(T_calc - T)
                if err < 1e-10:
                    break
        
        return U, V, W

    def solve_als(r, max_iters=100, restarts=10, use_structured_init=False):
        best_result = None
        best_err = float('inf')
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            if use_structured_init:
                # Structured initialization exploiting tensor symmetries
                # For matrix multiplication, we can initialize based on the factor structure
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                # Initialize with small random perturbations around a structured pattern
                for t in range(r):
                    # Random assignment of factors
                    i = np.random.randint(n)
                    j = np.random.randint(m) 
                    k = np.random.randint(p)
                    
                    # Base structure: one-hot-like with noise
                    for ii in range(n):
                        for jj in range(m):
                            U[ii * m + jj, t] = (1.0 if ii == i and jj == j else 0.0) + 0.1 * np.random.randn()
                    
                    for jj in range(m):
                        for kk in range(p):
                            V[jj * p + kk, t] = (1.0 if jj == j and kk == k else 0.0) + 0.1 * np.random.randn()
                    
                    for kk in range(p):
                        for ii in range(n):
                            W[kk * n + ii, t] = (1.0 if kk == k and ii == i else 0.0) + 0.1 * np.random.randn()
            else:
                U = np.random.uniform(-1, 1, (n * m, r))
                V = np.random.uniform(-1, 1, (m * p, r))
                W = np.random.uniform(-1, 1, (p * n, r))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break

                # Update W
                A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
                T_W = T.reshape(n * m * m * p, p * n)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                    W = W_T.T
                except:
                    continue

                # Update U
                A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
                T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                    U = U_T.T
                except:
                    continue

                # Update V
                A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
                T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                    V = V_T.T
                except:
                    continue

                # Check error periodically
                if it % 5 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-7:
                        # Try polishing
                        U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=50)
                        T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
                        err_pol = np.linalg.norm(T_calc_pol - T)
                        
                        if err_pol < 1e-9:
                            # Try snapping to half-integers
                            U_int, V_int, W_int = snap_to_half_int(U_pol, V_pol, W_pol)
                            if check_solution(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                            U_int2 = np.round(U_pol)
                            V_int2 = np.round(V_pol)
                            W_int2 = np.round(W_pol)
                            if check_solution(U_int2, V_int2, W_int2):
                                return U_int2, V_int2, W_int2
                        elif err_pol < err:
                            best_err = err_pol
                            best_result = (U_pol.copy(), V_pol.copy(), W_pol.copy())

        if best_result is not None:
            U, V, W = best_result
            # Final polish
            U, V, W = polish_solution(U, V, W, max_iters=100)
            T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
            err = np.linalg.norm(T_calc - T)
            if err < 1e-9:
                U_int, V_int, W_int = snap_to_half_int(U, V, W)
                if check_solution(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                U_int2 = np.round(U)
                V_int2 = np.round(V)
                W_int2 = np.round(W)
                if check_solution(U_int2, V_int2, W_int2):
                    return U_int2, V_int2, W_int2
        
        return None

    # Try ranks from lower bound up to schoolbook
    # For 4x5x5, known lower bound is around 40-50
    # Start from a reasonable lower bound
    search_start = max(min_rank, 35)
    
    for r in range(search_start, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # More restarts for smaller ranks (harder)
        restarts = 15 if r < 50 else 8
        max_iters = 60 if r < 50 else 40
        
        # Try both structured and random initialization
        for use_structured in [True, False]:
            if time.time() > deadline - 3:
                break
            result = solve_als(r, max_iters=max_iters, restarts=restarts, use_structured_init=use_structured)
            if result:
                U_int, V_int, W_int = result
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
