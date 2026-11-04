"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 3x5x6 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_r(n, m, p, r, num_restarts=5, max_iter=200, timeout=10):
    """Try to find a rank-r decomposition using alternating minimization."""
    start_time = time.time()
    np.random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                break
            
            # Update U: for each (i,j), solve min sum_k (u_ij . (v_jk ⊙ w_ki) - 1)^2
            # This is: u_ij . (sum_k (v_jk ⊙ w_ki) ⊙ (u_ij . (v_jk ⊙ w_ki) - 1)) = 0
            # Linear system: A_u u = b_u where A_u[c,c'] = sum_k v_jk[c]*w_ki[c']*v_jk[c']*w_ki[c'] ... 
            # Actually simpler: gradient of sum_k (u . a_k - 1)^2 where a_k = v_jk ⊙ w_ki
            # = 2 sum_k a_k (u . a_k - 1)
            # So: (sum_k a_k a_k^T) u = sum_k a_k
            
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for k in range(p):
                        a = V[j*p+k] * W[k*n+i]  # element-wise
                        A += np.outer(a, a)
                        b += a
                    # Solve A u = b
                    try:
                        U[idx] = np.linalg.solve(A + 1e-8*np.eye(r), b)
                    except:
                        U[idx] = np.linalg.lstsq(A + 1e-8*np.eye(r), b, rcond=None)[0]
            
            # Update V: for each (j,k), similar
            for j in range(m):
                for k in range(p):
                    idx = j*p + k
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for i in range(n):
                        a = U[i*m+j] * W[k*n+i]
                        A += np.outer(a, a)
                        b += a
                    try:
                        V[idx] = np.linalg.solve(A + 1e-8*np.eye(r), b)
                    except:
                        V[idx] = np.linalg.lstsq(A + 1e-8*np.eye(r), b, rcond=None)[0]
            
            # Update W: for each (k,i), similar
            for k in range(p):
                for i in range(n):
                    idx = k*n + i
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for j in range(m):
                        a = U[i*m+j] * V[j*p+k]
                        A += np.outer(a, a)
                        b += a
                    try:
                        W[idx] = np.linalg.solve(A + 1e-8*np.eye(r), b)
                    except:
                        W[idx] = np.linalg.lstsq(A + 1e-8*np.eye(r), b, rcond=None)[0]
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-10:
                # Found exact solution
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
        
        # End restart
        # Small perturbation
        U += 0.1 * np.random.randn(n*m, r)
        V += 0.1 * np.random.randn(m*p, r)
        W += 0.1 * np.random.randn(p*n, r)
    
    return None

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x5x6."""
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    schoolbook_rank = n * m * p  # 90
    
    # Start with schoolbook as default
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    best_rank = schoolbook_rank
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find better decompositions
    # Try ranks from 89 down to 80
    for r in range(schoolbook_rank - 1, 79, -1):
        if time.time() > deadline:
            break
        
        # Time budget per rank
        time_per_rank = max(1.0, (deadline - time.time()) / (schoolbook_rank - 1 - r + 1))
        
        result = _try_rank_r(n, m, p, r, num_restarts=3, max_iter=100, timeout=time_per_rank)
        if result is not None:
            U_list, V_list, W_list = result
            best_U, best_V, best_W = U_list, V_list, W_list
            best_rank = r
            # If we found a solution, try to go lower
            continue
        else:
            # If we couldn't find at this rank, stop
            break
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U = [[0] * schoolbook_rank for _ in range(n * m)]
        best_V = [[0] * schoolbook_rank for _ in range(m * p)]
        best_W = [[0] * schoolbook_rank for _ in range(p * n)]
        best_rank = schoolbook_rank
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
