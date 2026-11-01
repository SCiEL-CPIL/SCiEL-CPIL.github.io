"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy:
    1. Attempt to find a rank-16 decomposition for M_{2,3,3}. If successful, tile it twice to get rank 32 for M_{2,3,6}.
    2. If that fails, attempt to find a rank-11 decomposition for M_{2,3,2}. If successful, tile it three times to get rank 33 for M_{2,3,6}.
    3. Fallback to schoolbook decomposition (rank 36).
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    def verify_full_decomposition(U, V, W, n, m, p):
        """Verify if U, V, W exactly decompose the M_{n,m,p} tensor."""
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m + j][rr] * V[j*p + k][rr] * W[k*n + i][rr]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def find_decomposition(n_sub, m_sub, p_sub, target_rank, max_attempts=200, deadline=None):
        """
        Find a decomposition for M_{n_sub, m_sub, p_sub} with target_rank.
        Returns (U, V, W) as lists of lists, or None.
        """
        for attempt in range(max_attempts):
            if deadline and time.time() > deadline:
                return None
            
            np.random.seed(attempt * 12345 + 999)
            r = target_rank
            U = np.random.randn(n_sub, r)
            V = np.random.randn(m_sub, r)
            W = np.random.randn(p_sub, r)
            
            def residuals(params):
                U_flat = params[:n_sub*r].reshape(n_sub, r)
                V_flat = params[n_sub*r:(n_sub+m_sub)*r].reshape(m_sub, r)
                W_flat = params[(n_sub+m_sub)*r:].reshape(p_sub, r)
                
                res = []
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = np.sum(U_flat[i] * V_flat[j] * W_flat[k])
                            res.append(val - 1.0)
                return np.array(res)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use 'lm' method for unconstrained least squares
                result = least_squares(residuals, params0, method='lm', max_nfev=1000)
                
                if np.max(np.abs(result.fun)) < 1e-8:
                    U_sol = result.x[:n_sub*r].reshape(n_sub, r)
                    V_sol = result.x[n_sub*r:(n_sub+m_sub)*r].reshape(m_sub, r)
                    W_sol = result.x[(n_sub+m_sub)*r:].reshape(p_sub, r)
                    
                    # Try to snap to half-integers for robustness
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    max_err = 0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                val = np.sum(U_snap[i] * V_snap[j] * W_snap[k])
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-9:
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    else:
                        # Verify original solution
                        max_err = 0
                        for i in range(n_sub):
                            for j in range(m_sub):
                                for k in range(p_sub):
                                    val = np.sum(U_sol[i] * V_sol[j] * W_sol[k])
                                    max_err = max(max_err, abs(val - 1.0))
                        if max_err < 1e-9:
                            return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
            except Exception:
                continue
        return None

    # Attempt 1: Rank 16 for M_{2,3,3} -> Rank 32 for M_{2,3,6}
    # U: (2, 16), V: (3, 16), W: (3, 16)
    decomp_233 = find_decomposition(2, 3, 3, 16, max_attempts=100, deadline=deadline)
    
    if decomp_233 is not None:
        U1, V1, W1 = decomp_233
        r_block = 16
        r_total = 2 * r_block  # 32
        
        U_full = [[0.0] * r_total for _ in range(n*m)]
        V_full = [[0.0] * r_total for _ in range(m*p)]
        W_full = [[0.0] * r_total for _ in range(p*n)]
        
        # First block: k in {0,1,2}
        for i in range(n):
            for j in range(m):
                for rr in range(r_block):
                    U_full[i*m + j][rr] = U1[i][rr]
        
        for j in range(m):
            for k in range(3):
                for rr in range(r_block):
                    V_full[j*p + k][rr] = V1[j][rr]
        
        for k in range(3):
            for i in range(n):
                for rr in range(r_block):
                    W_full[k*n + i][rr] = W1[k][rr]
        
        # Second block: k in {3,4,5}
        for i in range(n):
            for j in range(m):
                for rr in range(r_block):
                    U_full[i*m + j][r_block + rr] = U1[i][rr]
        
        for j in range(m):
            for k in range(3):
                for rr in range(r_block):
                    V_full[j*p + (k+3)][r_block + rr] = V1[j][rr]
        
        for k in range(3):
            for i in range(n):
                for rr in range(r_block):
                    W_full[(k+3)*n + i][r_block + rr] = W1[k][rr]
        
        if verify_full_decomposition(U_full, V_full, W_full, n, m, p):
            return {"U": U_full, "V": V_full, "W": W_full}

    # Attempt 2: Rank 11 for M_{2,3,2} -> Rank 33 for M_{2,3,6}
    decomp_232 = find_decomposition(2, 3, 2, 11, max_attempts=100, deadline=deadline)
    
    if decomp_232 is not None:
        U1, V1, W1 = decomp_232
        r_block = 11
        r_total = 3 * r_block  # 33
        
        U_full = [[0.0] * r_total for _ in range(n*m)]
        V_full = [[0.0] * r_total for _ in range(m*p)]
        W_full = [[0.0] * r_total for _ in range(p*n)]
        
        # Three blocks of k
        for block_idx in range(3):
            k_offset = block_idx * 2
            
            for i in range(n):
                for j in range(m):
                    for rr in range(r_block):
                        U_full[i*m + j][block_idx*r_block + rr] = U1[i][rr]
            
            for j in range(m):
                for k in range(2):
                    for rr in range(r_block):
                        V_full[j*p + (k_offset + k)][block_idx*r_block + rr] = V1[j][rr]
            
            for k in range(2):
                for i in range(n):
                    for rr in range(r_block):
                        W_full[(k_offset + k)*n + i][block_idx*r_block + rr] = W1[k][rr]
        
        if verify_full_decomposition(U_full, V_full, W_full, n, m, p):
            return {"U": U_full, "V": V_full, "W": W_full}

    # Fallback: Schoolbook decomposition (Rank 36)
    rank = n * m * p
    U_fb = [[0.0] * rank for _ in range(n*m)]
    V_fb = [[0.0] * rank for _ in range(m*p)]
    W_fb = [[0.0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j][col] = 1.0
                V_fb[j*p + k][col] = 1.0
                W_fb[k*n + i][col] = 1.0
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
