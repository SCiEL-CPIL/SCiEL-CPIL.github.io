import numpy as np
import time
import random

def construct():
    n, m, p = 2, 5, 5
    
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    def build_current_decomposition():
        """Build the current block-based Strassen decomposition."""
        rank = 53  # Actual rank from the blocks
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        
        def add_strassen_block(i_off, j_off, k_off):
            nonlocal c
            x00 = i_off * m + j_off
            x01 = i_off * m + j_off + 1
            x10 = (i_off+1) * m + j_off
            x11 = (i_off+1) * m + j_off + 1
            y00 = j_off * p + k_off
            y01 = j_off * p + k_off + 1
            y10 = (j_off+1) * p + k_off
            y11 = (j_off+1) * p + k_off + 1
            z00 = k_off * n + i_off
            z01 = (k_off+1) * n + i_off
            z10 = k_off * n + (i_off+1)
            z11 = (k_off+1) * n + (i_off+1)
            
            # M1
            U[x00][c] += 1; U[x11][c] += 1
            V[y00][c] += 1; V[y11][c] += 1
            W[z00][c] += 1; W[z11][c] += 1
            c += 1
            # M2
            U[x10][c] += 1; U[x11][c] += 1
            V[y00][c] += 1
            W[z10][c] += 1; W[z11][c] -= 1
            c += 1
            # M3
            U[x00][c] += 1
            V[y01][c] += 1; V[y11][c] -= 1
            W[z01][c] += 1; W[z11][c] += 1
            c += 1
            # M4
            U[x11][c] += 1
            V[y10][c] += 1; V[y00][c] -= 1
            W[z00][c] += 1; W[z10][c] += 1
            c += 1
            # M5
            U[x00][c] += 1; U[x01][c] += 1
            V[y11][c] += 1
            W[z00][c] -= 1; W[z01][c] += 1
            c += 1
            # M6
            U[x10][c] += 1; U[x00][c] -= 1
            V[y00][c] += 1; V[y01][c] += 1
            W[z11][c] += 1
            c += 1
            # M7
            U[x01][c] += 1; U[x11][c] -= 1
            V[y10][c] += 1; V[y11][c] += 1
            W[z00][c] += 1
            c += 1
        
        def add_schoolbook_2x2_x_2x1(i_off, j_off, k_off):
            nonlocal c
            for ii in range(2):
                for jj in range(2):
                    U[(i_off+ii)*m + (j_off+jj)][c] += 1
                    V[(j_off+jj)*p + k_off][c] += 1
                    W[k_off*n + (i_off+ii)][c] += 1
                    c += 1
        
        def add_schoolbook_2x1_x_1x2(i_off, j_off, k_off):
            nonlocal c
            for ii in range(2):
                for kk in range(2):
                    U[(i_off+ii)*m + j_off][c] += 1
                    V[j_off*p + (k_off+kk)][c] += 1
                    W[(k_off+kk)*n + (i_off+ii)][c] += 1
                    c += 1
        
        def add_schoolbook_2x1_x_1x1(i_off, j_off, k_off):
            nonlocal c
            for ii in range(2):
                U[(i_off+ii)*m + j_off][c] += 1
                V[j_off*p + k_off][c] += 1
                W[k_off*n + (i_off+ii)][c] += 1
                c += 1
        
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_schoolbook_2x2_x_2x1(0, 0, 4)
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_schoolbook_2x2_x_2x1(0, 2, 4)
        add_schoolbook_2x1_x_1x2(0, 4, 0)
        add_schoolbook_2x1_x_1x2(0, 4, 2)
        add_schoolbook_2x1_x_1x1(0, 4, 4)
        
        # Trim to actual used rank
        actual_rank = c
        U = [row[:actual_rank] for row in U]
        V = [row[:actual_rank] for row in V]
        W = [row[:actual_rank] for row in W]
        return U, V, W, actual_rank
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half_integer(arr):
        """Snap values to nearest half-integer."""
        return [[round(2*x)/2 for x in row] for row in arr]
    
    def als_optimize(U, V, W, max_iters=200, tol=1e-10):
        """ALS optimization to minimize ||T - sum_r u_r v_r w_r||."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Precompute target tensor entries
        # T[i*m+j, j*p+k, k*n+i] should sum to 1
        # We'll use the constraint: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # We want U[i*m+j, :] such that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                        # This is underdetermined, so use least squares
                        # Actually, ALS for tensor: fix V and W, solve for U
                        pass
            
            # Standard ALS: fix two factors, solve for third
            # For factor U: for each row a = i*m+j, we have constraints from all (i,j,k)
            # sum_r U[a,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is: sum_r U[a,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # Let M_a[r] = V[j*p+k,r] * W[k*n+i,r] for the specific j,k
            # But different k give different constraints for the same a
            
            # Better approach: iterate over all constraints
            # For U: fix V, W. For each a (row of U), we have multiple equations.
            # Use least squares: minimize sum over all (i,j,k) of (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
            # where a=i*m+j, b=j*p+k, c=k*n+i
            
            # This is complex. Let's use a simpler gradient-based approach.
            break
        
        return U.tolist(), V.tolist(), W.tolist()
    
    def try_lower_rank(start_rank, max_attempts=5):
        """Try to find a decomposition with rank < start_rank."""
        deadline = time.time() + 100  # Safety margin
        
        for target_rank in range(start_rank - 1, 30, -1):
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                # Initialize with random or structured initialization
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Initialize: copy from current decomposition if possible
                if target_rank <= start_rank:
                    U_init, V_init, W_init, _ = build_current_decomposition()
                    U = np.array(U_init[:n*m, :target_rank], dtype=float)
                    V = np.array(V_init[:m*p, :target_rank], dtype=float)
                    W = np.array(W_init[:p*n, :target_rank], dtype=float)
                else:
                    U = np.random.randn(n*m, target_rank) * 0.5
                    V = np.random.randn(m*p, target_rank) * 0.5
                    W = np.random.randn(p*n, target_rank) * 0.5
                
                # ALS iterations
                for it in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Update U: for each row a, solve least squares
                    for a in range(n*m):
                        i = a // m
                        j = a % m
                        # Constraints: for all k, sum_r U[a,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                        # Build matrix: rows are k, cols are r
                        # A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve A_mat @ U[a,:] = ones(p)
                        try:
                            U[a, :], residuals, rank_lsv, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        except:
                            pass
                    
                    # Update V
                    for b in range(m*p):
                        j = b // p
                        k = b % p
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            V[b, :], residuals, rank_lsv, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                        except:
                            pass
                    
                    # Update W
                    for c_idx in range(p*n):
                        k = c_idx // n
                        i = c_idx % n
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            W[c_idx, :], residuals, rank_lsv, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Check if solution is close to exact
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-6:
                    # Snap to half-integers and verify
                    U_snap = snap_to_half_integer(U.tolist())
                    V_snap = snap_to_half_integer(V.tolist())
                    W_snap = snap_to_half_integer(W.tolist())
                    if verify(U_snap, V_snap, W_snap, tol=1e-9):
                        return U_snap, V_snap, W_snap, target_rank
                
                if time.time() > deadline:
                    break
            
            if found:
                break
        
        return None
    
    # Build the current decomposition
    U_cur, V_cur, W_cur, cur_rank = build_current_decomposition()
    
    # Verify current decomposition
    if not verify(U_cur, V_cur, W_cur):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    best_U, best_V, best_W, best_rank = U_cur, V_cur, W_cur, cur_rank
    
    deadline = time.time() + 100
    if time.time() < deadline:
        result = try_lower_rank(cur_rank, max_attempts=3)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if rank_new < best_rank:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new
    
    return {"U": best_U, "V": best_V, "W": best_W}
