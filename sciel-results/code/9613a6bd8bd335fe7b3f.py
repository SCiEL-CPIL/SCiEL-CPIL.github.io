import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Leave 5s margin for safety
    
    # Construct rank-71 solution using Strassen + schoolbook
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Try to find rank-70 solution using structured search
    if time.time() - start_time < time_limit - 20:
        res_70 = try_find_rank_70_structured(n, m, p, start_time, time_limit)
        if res_70 is not None:
            return res_70
    
    return {"U": U, "V": V, "W": W}

def try_find_rank_70_structured(n, m, p, start_time, time_limit):
    """
    Try to find a rank-70 solution by modifying the rank-71 solution.
    Strategy: The rank-71 solution has 4 Strassen blocks (28 rank) + 43 schoolbook terms.
    We need to eliminate one rank unit. 
    
    Approach: Use alternating minimization with better initialization and polishing.
    """
    rank = 70
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build the rank-71 solution as numpy arrays for initialization
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_strassen_rank = len(blocks) * 7
    total_terms = n * m * p
    strassen_terms_count = len(blocks) * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank_71 = total_strassen_rank + schoolbook_terms_count
    
    U71 = np.zeros((n_m, total_rank_71))
    V71 = np.zeros((m_p, total_rank_71))
    W71 = np.zeros((p_n, total_rank_71))
    
    col = 0
    
    def add_strassen_block_np(i0, j0, k0, col_offset):
        c = col_offset + 0
        U71[i0*m + j0, c] += 1
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[j0*p + k0, c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 1
        U71[(i0+1)*m + j0, c] += 1
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[j0*p + k0, c] += 1
        W71[k0*n + (i0+1), c] += 1
        W71[(k0+1)*n + (i0+1), c] += -1
        
        c = col_offset + 2
        U71[i0*m + j0, c] += 1
        V71[j0*p + (k0+1), c] += 1
        V71[(j0+1)*p + (k0+1), c] += -1
        W71[(k0+1)*n + i0, c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 3
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[(j0+1)*p + k0, c] += 1
        V71[j0*p + k0, c] += -1
        W71[k0*n + i0, c] += 1
        W71[k0*n + (i0+1), c] += 1
        
        c = col_offset + 4
        U71[i0*m + j0, c] += 1
        U71[i0*m + (j0+1), c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += -1
        W71[(k0+1)*n + i0, c] += 1
        
        c = col_offset + 5
        U71[(i0+1)*m + j0, c] += 1
        U71[i0*m + j0, c] += -1
        V71[j0*p + k0, c] += 1
        V71[j0*p + (k0+1), c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 6
        U71[i0*m + (j0+1), c] += 1
        U71[(i0+1)*m + (j0+1), c] += -1
        V71[(j0+1)*p + k0, c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block_np(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U71[i*m+j, col] += 1
                    V71[j*p+k, col] += 1
                    W71[k*n+i, col] += 1
                    col += 1
    
    def compute_error(U, V, W):
        """Compute squared error for the decomposition."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    dot = np.dot(u_val, v_val * w_val)
                    error += (dot - 1.0) ** 2
        return error
    
    def round_and_verify(U, V, W, tol=1e-6):
        """Round to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        error_r = compute_error(U_r, V_r, W_r)
        if error_r < tol:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        return None
    
    def polish(U, V, W, max_iters=50):
        """Polish with tighter tolerance."""
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    w_indices = np.array([k*n + i for k in range(p)])
                    W_sub = W[w_indices, :]
                    V_sub = V[j*p:(j+1)*p, :]
                    A = V_sub * W_sub
                    b = np.ones(p)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    u_indices = np.array([i*m + j for i in range(n)])
                    U_sub = U[u_indices, :]
                    w_indices = np.array([k*n + i for i in range(n)])
                    W_sub = W[w_indices, :]
                    A = U_sub * W_sub
                    b = np.ones(n)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    u_indices = np.array([i*m + j for j in range(m)])
                    U_sub = U[u_indices, :]
                    v_indices = np.array([j*p + k for j in range(m)])
                    V_sub = V[v_indices, :]
                    A = U_sub * V_sub
                    b = np.ones(m)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        W[k*n+i, :] = x
                    except:
                        pass
            
            error = compute_error(U, V, W)
            if error < 1e-8:
                result = round_and_verify(U, V, W, tol=1e-8)
                if result is not None:
                    return result
        
        return None
    
    # Try multiple initialization strategies
    for trial in range(40):
        if time.time() - start_time > time_limit - 5:
            break
        
        np.random.seed(123 + trial * 7)
        
        # Different initialization strategies
        if trial < 8:
            # Random initialization with varying scales
            scale = 0.3 + 0.1 * (trial % 4)
            U = np.random.randn(n_m, rank) * scale
            V = np.random.randn(m_p, rank) * scale
            W = np.random.randn(p_n, rank) * scale
        elif trial < 16:
            # Perturbed rank-71 with varying noise
            noise_scale = 0.05 + 0.02 * (trial % 4)
            U = U71[:, :rank].copy()
            V = V71[:, :rank].copy()
            W = W71[:, :rank].copy()
            U += np.random.randn(n_m, rank) * noise_scale
            V += np.random.randn(m_p, rank) * noise_scale
            W += np.random.randn(p_n, rank) * noise_scale
        else:
            # Hybrid: mix of random and structured
            mix_ratio = 0.5 + 0.1 * (trial % 4)
            U_rand = np.random.randn(n_m, rank) * 0.5
            V_rand = np.random.randn(m_p, rank) * 0.5
            W_rand = np.random.randn(p_n, rank) * 0.5
            U = mix_ratio * U71[:, :rank] + (1 - mix_ratio) * U_rand
            V = mix_ratio * V71[:, :rank] + (1 - mix_ratio) * V_rand
            W = mix_ratio * W71[:, :rank] + (1 - mix_ratio) * W_rand
        
        # Run alternating minimization
        for iteration in range(150):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    w_indices = np.array([k*n + i for k in range(p)])
                    W_sub = W[w_indices, :]
                    V_sub = V[j*p:(j+1)*p, :]
                    A = V_sub * W_sub
                    b = np.ones(p)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    u_indices = np.array([i*m + j for i in range(n)])
                    U_sub = U[u_indices, :]
                    w_indices = np.array([k*n + i for i in range(n)])
                    W_sub = W[w_indices, :]
                    A = U_sub * W_sub
                    b = np.ones(n)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    u_indices = np.array([i*m + j for j in range(m)])
                    U_sub = U[u_indices, :]
                    v_indices = np.array([j*p + k for j in range(m)])
                    V_sub = V[v_indices, :]
                    A = U_sub * V_sub
                    b = np.ones(m)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check error periodically
            if iteration % 5 == 0:
                error = compute_error(U, V, W)
                if error < 1e-6:
                    result = round_and_verify(U, V, W)
                    if result is not None:
                        return result
                elif error < 1e-4:
                    # Try polishing
                    result = polish(U.copy(), V.copy(), W.copy(), max_iters=30)
                    if result is not None:
                        return result
        
    return None
