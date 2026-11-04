"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 110  # leave safety margin before 120s kill

    def verify_np(U, V, W, n, m, p, rank, tol=1e-10):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def rational_round(arr, denom=2):
        """Round to multiples of 1/denom."""
        return np.round(arr * denom) / denom

    def get_3x3x3_decomp(rank_333, seed=0):
        """Get a rank-rank_333 decomposition of 3x3x3 matrix multiplication.
        Returns U3 (9, rank_333), V3 (9, rank_333), W3 (9, rank_333) where
        U3[i*3+j, t], V3[j*3+k, t], W3[k*3+i, t] for i,j,k in 0..2.
        """
        rng = np.random.RandomState(seed)
        
        if rank_333 >= 27:
            # Schoolbook
            U3 = np.zeros((9, rank_333))
            V3 = np.zeros((9, rank_333))
            W3 = np.zeros((9, rank_333))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U3[i*3+j, col] = 1.0
                        V3[j*3+k, col] = 1.0
                        W3[k*3+i, col] = 1.0
                        col += 1
            return U3, V3, W3
        
        # Use Strassen-based construction for 3x3x3
        # Embed 2x2x2 Strassen (rank 7) and add schoolbook for remaining
        U3 = np.zeros((9, rank_333))
        V3 = np.zeros((9, rank_333))
        W3 = np.zeros((9, rank_333))
        
        # Strassen 2x2x2 for the (0,1)x(0,1)x(0,1) sub-block
        U_s = np.array([
            [1, 0, -1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, -1, 0]
        ], dtype=float)
        V_s = np.array([
            [0, 0, 1, -1, 1, 0, -1],
            [-1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0],
            [-1, 1, 0, 1, 1, 1, 0]
        ], dtype=float)
        W_s = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1]
        ], dtype=float)
        
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U3[i*3+j, t] = U_s[i, j]
            for j in range(2):
                for k in range(2):
                    V3[j*3+k, t] = V_s[j, k]
            for k in range(2):
                for i in range(2):
                    W3[k*3+i, t] = W_s[k, i]
        
        # Add schoolbook terms for remaining indices
        col = 7
        # For i=2 (third row of A)
        for j in range(3):
            for k in range(3):
                if col >= rank_333:
                    break
                U3[2*3+j, col] = 1.0
                V3[j*3+k, col] = 1.0
                W3[k*3+2, col] = 1.0
                col += 1
        # For j=2 (third col of A / third row of B) with i=0,1
        for i in range(2):
            for k in range(3):
                if col >= rank_333:
                    break
                U3[i*3+2, col] = 1.0
                V3[2*3+k, col] = 1.0
                W3[k*3+i, col] = 1.0
                col += 1
        # For k=2 (third col of B / third row of C) with i=0,1 and j=0,1
        for i in range(2):
            for j in range(2):
                if col >= rank_333:
                    break
                U3[i*3+j, col] = 1.0
                V3[j*3+2, col] = 1.0
                W3[2*3+i, col] = 1.0
                col += 1
        
        # Add small random perturbation for ALS to work with
        if rank_333 > 7:
            U3[:, 7:] += rng.randn(9, max(0, rank_333-7)) * 0.1
            V3[:, 7:] += rng.randn(9, max(0, rank_333-7)) * 0.1
            W3[:, 7:] += rng.randn(9, max(0, rank_333-7)) * 0.1
        
        return U3, V3, W3

    def project_to_234(U3, V3, W3, rank_333):
        """Project a 3x3x3 decomposition to 2x3x4 by zeroing out
        indices that don't exist in the 2x3x4 tensor.
        Returns U (6, rank_333), V (12, rank_333), W (8, rank_333).
        """
        U = np.zeros((n*m, rank_333))
        V = np.zeros((m*p, rank_333))
        W = np.zeros((p*n, rank_333))
        
        # Map 3x3x3 indices to 2x3x4
        # A: 2x3 -> i in 0,1; j in 0,1,2
        # B: 3x4 -> j in 0,1,2; k in 0,1,2,3
        # C: 2x4 -> i in 0,1; k in 0,1,2,3
        
        # U3[i3*3+j3, t] -> U[i*3+j, t] for i in 0,1; j in 0,1,2
        for i in range(2):
            for j in range(3):
                U[i*3+j, :] = U3[i*3+j, :]
        
        # V3[j3*3+k3, t] -> V[j*4+k, t] for j in 0,1,2; k in 0,1,2,3
        # Only k=0,1,2 come from 3x3x3
        for j in range(3):
            for k in range(3):
                V[j*4+k, :] = V3[j*3+k, :]
        
        # W3[k3*3+i3, t] -> W[k*2+i, t] for k in 0,1,2,3; i in 0,1
        # Only k=0,1,2 come from 3x3x3
        for k in range(3):
            for i in range(2):
                W[k*2+i, :] = W3[k*3+i, :]
        
        return U, V, W

    def try_als(U_init, V_init, W_init, rank, max_iter=800, tol=1e-14, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                if verify_np(U, V, W, n, m, p, rank):
                    return U, V, W
        
        if verify_np(U, V, W, n, m, p, rank):
            return U, V, W
        
        return None

    def rational_polish(U, V, W, rank, max_iters=200, seed=0):
        """Try to find a rational (half-integer) decomposition near the given solution."""
        rng = np.random.RandomState(seed)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Round to half-integers
            U_r = rational_round(U, 2)
            V_r = rational_round(V, 2)
            W_r = rational_round(W, 2)
            
            if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                return U_r, V_r, W_r
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        err += (s - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            # Perturb and retry
            scale = max(0.1, 1.0 - it / max_iters)
            U = U_r + rng.randn(n * m, rank) * 0.01 * scale
            V = V_r + rng.randn(m * p, rank) * 0.01 * scale
            W = W_r + rng.randn(p * n, rank) * 0.01 * scale
        
        return best_U, best_V, best_W

    def try_border_rank_lift(target_rank, seed=0):
        """Try to find a rank-target_rank decomposition of 2x3x4 by:
        1. Getting a higher-rank 3x3x3 decomposition
        2. Projecting to 2x3x4
        3. Using ALS to reduce to target_rank
        4. Applying Gaussian elimination to remove redundant terms
        """
        if time.time() > deadline - 10:
            return None
        
        # Start with a 3x3x3 decomposition of rank >= target_rank
        rank_333 = max(target_rank, 18)
        U3, V3, W3 = get_3x3x3_decomp(rank_333, seed=seed)
        
        # Project to 2x3x4
        U_proj, V_proj, W_proj = project_to_234(U3, V3, W3, rank_333)
        
        # Truncate to target_rank by keeping the most important columns
        # Use SVD to identify important directions
        try:
            # Combine all matrices to find important directions
            all_cols = np.hstack([U_proj, V_proj, W_proj])
            U_svd, S, Vt = np.linalg.svd(all_cols, full_matrices=False)
            
            # Keep top target_rank directions
            U_keep = U_svd[:, :target_rank]
            V_keep = Vt[:target_rank, :].T
            
            # Project each matrix onto the kept subspace
            U_red = U_proj @ V_keep  # (6, target_rank)
            V_red = V_proj @ V_keep  # (12, target_rank)
            W_red = W_proj @ V_keep  # (8, target_rank)
        except:
            U_red = U_proj[:, :target_rank]
            V_red = V_proj[:, :target_rank]
            W_red = W_proj[:, :target_rank]
        
        # Run ALS to refine
        result = try_als(U_red, V_red, W_red, target_rank, max_iter=300, tol=1e-13, seed=seed)
        if result is not None:
            return result
        
        # Try with different seeds
        for s in range(5):
            if time.time() > deadline - 5:
                break
            U3, V3, W3 = get_3x3x3_decomp(rank_333, seed=seed + 100 + s)
            U_proj, V_proj, W_proj = project_to_234(U3, V3, W3, rank_333)
            
            U_red = U_proj[:, :target_rank] + np.random.RandomState(seed+s).randn(6, target_rank) * 0.05
            V_red = V_proj[:, :target_rank] + np.random.RandomState(seed+s).randn(12, target_rank) * 0.05
            W_red = W_proj[:, :target_rank] + np.random.RandomState(seed+s).randn(8, target_rank) * 0.05
            
            result = try_als(U_red, V_red, W_red, target_rank, max_iter=200, tol=1e-13, seed=seed+s)
            if result is not None:
                return result
        
        return None

    best_result = None
    best_rank = n * m * p  # 24

    # First, try border rank lifting from low to high rank
    for rank in range(14, 20):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = try_border_rank_lift(rank, seed=seed)
            if result is not None:
                U, V, W = result
                
                # Try rational polish
                U_r, V_r, W_r = rational_polish(U, V, W, rank, max_iters=50, seed=seed)
                
                if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
                
                # Also try without polish
                U_list = [[round(float(x), 12) for x in row] for row in U]
                V_list = [[round(float(x), 12) for x in row] for row in V]
                W_list = [[round(float(x), 12) for x in row] for row in W]
                U_check = np.array(U_list, dtype=float)
                V_check = np.array(V_list, dtype=float)
                W_check = np.array(W_list, dtype=float)
                if verify_np(U_check, V_check, W_check, n, m, p, rank, tol=1e-9):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    # If border rank lifting didn't work, try direct ALS
    if best_result is None:
        rng = np.random.RandomState(42)
        for rank in range(14, 24):
            if time.time() > deadline - 10:
                break
            
            found = False
            for seed in range(15):
                if time.time() > deadline - 5:
                    break
                
                U_init = rng.randn(n * m, rank) * 0.5
                V_init = rng.randn(m * p, rank) * 0.5
                W_init = rng.randn(p * n, rank) * 0.5
                
                result = try_als(U_init, V_init, W_init, rank, max_iter=400, tol=1e-14, seed=seed)
                if result is not None:
                    U, V, W = result
                    
                    # Try rational polish
                    U_r, V_r, W_r = rational_polish(U, V, W, rank, max_iters=80, seed=seed)
                    
                    if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                        U_list = [[float(x) for x in row] for row in U_r]
                        V_list = [[float(x) for x in row] for row in V_r]
                        W_list = [[float(x) for x in row] for row in W_r]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
                    
                    # Also try without polish
                    U_list = [[round(float(x), 12) for x in row] for row in U]
                    V_list = [[round(float(x), 12) for x in row] for row in V]
                    W_list = [[round(float(x), 12) for x in row] for row in W]
                    U_check = np.array(U_list, dtype=float)
                    V_check = np.array(V_list, dtype=float)
                    W_check = np.array(W_list, dtype=float)
                    if verify_np(U_check, V_check, W_check, n, m, p, rank, tol=1e-9):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                break
    
    if best_result is not None:
        return best_result

    # Fallback: schoolbook rank 24
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
