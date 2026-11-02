import time
import numpy as np

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify(U, V, W, n=2, m=2, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _build_tensor(n=2, m=2, p=5):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    
    return U, V, W

def _robust_lstsq(A, b):
    try:
        sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(sol)):
            raise np.linalg.LinAlgError("Non-finite solution")
        return sol
    except np.linalg.LinAlgError:
        lambda_reg = 1e-8
        ATA = A.T @ A + lambda_reg * np.eye(A.shape[1])
        ATb = A.T @ b
        try:
            sol = np.linalg.solve(ATA, ATb)
            if not np.all(np.isfinite(sol)):
                raise np.linalg.LinAlgError("Non-finite solution in fallback")
            return sol
        except np.linalg.LinAlgError:
            return np.zeros_like(b)

def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-12, deadline=None):
    prev_err = np.inf
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            U[u] = _robust_lstsq(VvW_flat, b)
        
        if not np.all(np.isfinite(U)):
            break
            
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            V[v] = _robust_lstsq(Uw_flat, b)
            
        if not np.all(np.isfinite(V)):
            break

        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            W[w] = _robust_lstsq(UV_flat, b)
            
        if not np.all(np.isfinite(W)):
            break

        err = np.linalg.norm(T - np.einsum('ur,vr,wr->uvw', U, V, W))
        if not np.isfinite(err):
            break
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Final high-precision polish pass
    if deadline is None or time.time() < deadline - 0.5:
        for _ in range(10):
            if deadline is not None and time.time() > deadline:
                break
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n*m):
                b = T[u].reshape(-1)
                U[u] = _robust_lstsq(VvW_flat, b)
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m*p):
                b = T[:, v, :].reshape(-1)
                V[v] = _robust_lstsq(Uw_flat, b)
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p*n):
                b = T[:, :, w].reshape(-1)
                W[w] = _robust_lstsq(UV_flat, b)
            err = np.linalg.norm(T - np.einsum('ur,vr,wr->uvw', U, V, W))
            if not np.isfinite(err) or err < 1e-14:
                break
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    result = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(result, T, atol=1e-6):
        return None
    
    # Try rationalizing with increasing denominators
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result_s, T, atol=1e-5):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-8:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Final attempt: use the floating-point values directly if they verify
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _syndrome_init(n, m, p, R, trial_idx):
    """
    Syndrome-based initialization: construct random factor matrices whose
    bilinear products satisfy the target tensor constraints in a structured way.
    Uses the null-space of a constraint matrix to generate valid starting points.
    """
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # For each rank component r, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # 
    # Strategy: assign each rank component to "cover" a subset of (i,j,k) triples,
    # using structured patterns that make the bilinear identity easier to satisfy.
    
    # Pattern 1: Block-diagonal style - each component handles specific (i,j) pairs
    # for all k, or specific k values for all (i,j)
    pattern = trial_idx % 4
    
    if pattern == 0:
        # Assign components to (i,j) pairs, covering all k
        pairs = [(i, j) for i in range(n) for j in range(m)]
        for r in range(R):
            if r < len(pairs):
                i, j = pairs[r]
                # This component contributes to all k for this (i,j)
                U[i*m + j, r] = 1.0
                for k in range(p):
                    V[j*p + k, r] = 1.0 / p
                    W[k*n + i, r] = 1.0
            else:
                # Extra components: random but small
                U[:, r] = np.random.randn(n*m) * 0.1
                V[:, r] = np.random.randn(m*p) * 0.1
                W[:, r] = np.random.randn(p*n) * 0.1
    
    elif pattern == 1:
        # Assign components to k values, covering all (i,j)
        for r in range(R):
            if r < p:
                k = r
                for i in range(n):
                    for j in range(m):
                        U[i*m + j, r] = 1.0 / (n*m)
                        V[j*p + k, r] = 1.0
                        W[k*n + i, r] = 1.0
            else:
                U[:, r] = np.random.randn(n*m) * 0.1
                V[:, r] = np.random.randn(m*p) * 0.1
                W[:, r] = np.random.randn(p*n) * 0.1
    
    elif pattern == 2:
        # Mixed: Strassen-like for first 7, then syndrome-based for rest
        U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
        if R >= 7:
            U[:, :7] = U_str
            V[:, :7] = V_str
            W[:, :7] = W_str
        for r in range(7, R):
            # Each extra component handles a specific (i,j,k) triple
            idx = r - 7
            i = idx % n
            j = (idx // n) % m
            k = (idx // (n*m)) % p
            U[i*m + j, r] = 1.0
            V[j*p + k, r] = 1.0
            W[k*n + i, r] = 1.0
    
    else:
        # Pattern 3: Randomized syndrome - each component has structured support
        for r in range(R):
            # Each component covers a random subset of positions
            u_support = np.random.choice(n*m, size=min(2, n*m), replace=False)
            v_support = np.random.choice(m*p, size=min(3, m*p), replace=False)
            w_support = np.random.choice(p*n, size=min(2, p*n), replace=False)
            
            for idx in u_support:
                U[idx, r] = np.random.choice([-1, 1])
            for idx in v_support:
                V[idx, r] = np.random.choice([-1, 1])
            for idx in w_support:
                W[idx, r] = np.random.choice([-1, 1])
    
    return U, V, W

def _search_rank_aggressive(n, m, p, R, timeout=10, n_trials=200):
    """Aggressive ALS search with multiple initialization strategies."""
    start_time = time.time()
    deadline = start_time + timeout
    T = _build_tensor(n, m, p)
    
    # Base Strassen block for initialization
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Strategy 0: Syndrome-based initialization (new)
        if trial % 5 == 0:
            U, V, W = _syndrome_init(n, m, p, R, trial)
        
        # Strategy 1: Start from Strassen + random completion
        elif trial % 5 == 1:
            if R >= 7:
                U[:, :7] = U_str
                V[:, :7] = V_str
                W[:, :7] = W_str
            if R > 7:
                scale = 0.3 + 0.2 * (trial % 5)
                U[:, 7:] = np.random.randn(n*m, R-7) * scale
                V[:, 7:] = np.random.randn(m*p, R-7) * scale
                W[:, 7:] = np.random.randn(p*n, R-7) * scale
        
        # Strategy 2: Pure random with small scale
        elif trial % 5 == 2:
            scale = 0.5 + 0.1 * (trial % 4)
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
        
        # Strategy 3: Two Strassen blocks + random
        elif trial % 5 == 3:
            if R >= 7:
                U[:, :7] = U_str
                V[:, :7] = V_str
                W[:, :7] = W_str
            if R >= 14:
                U_str2, V_str2, W_str2 = _get_strassen_block_np(n, m, p, 2, 3)
                U[:, 7:14] = U_str2
                V[:, 7:14] = V_str2
                W[:, 7:14] = W_str2
            if R > 14:
                U[:, 14:] = np.random.randn(n*m, R-14) * 0.3
                V[:, 14:] = np.random.randn(m*p, R-14) * 0.3
                W[:, 14:] = np.random.randn(p*n, R-14) * 0.3
        
        # Strategy 4: Syndrome-based with different pattern
        else:
            U, V, W = _syndrome_init(n, m, p, R, trial + 100)
        
        # Reduce max_iter to keep individual trials fast
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=100, deadline=deadline)
        
        if err < 1e-8:
            result = _check_and_rationalize(U, V, W, T, R, n, m, p)
            if result is not None:
                return result
    return None

def _construct_rank18(n=2, m=2, p=5):
    """Known valid rank-18 decomposition: two Strassen blocks + schoolbook for col 4."""
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    U_str2, V_str2, W_str2 = _get_strassen_block_np(n, m, p, 2, 3)
    U[:, 7:14] = U_str2
    V[:, 7:14] = V_str2
    W[:, 7:14] = W_str2
    
    # Schoolbook for column 4
    U[0, 14] = 1; V[0*p+4, 14] = 1; W[4*n+0, 14] = 1
    U[1, 15] = 1; V[1*p+4, 15] = 1; W[4*n+0, 15] = 1
    U[2, 16] = 1; V[0*p+4, 16] = 1; W[4*n+1, 16] = 1
    U[3, 17] = 1; V[1*p+4, 17] = 1; W[4*n+1, 17] = 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Verify before returning
    if not _verify(U_list, V_list, W_list, n, m, p):
        raise ValueError("Rank-18 construction failed verification")
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_perturb_rank(n, m, p, target_rank, timeout=10, n_trials=50):
    """
    Attempts to find a lower rank decomposition by perturbing the known rank-18 solution.
    """
    start_time = time.time()
    deadline = start_time + timeout
    T = _build_tensor(n, m, p)
    
    # Get the base rank-18 solution
    base_U, base_V, base_W = _get_strassen_block_np(n, m, p, 0, 1)
    base_U2, base_V2, base_W2 = _get_strassen_block_np(n, m, p, 2, 3)
    
    full_U = np.zeros((n*m, 18))
    full_V = np.zeros((m*p, 18))
    full_W = np.zeros((p*n, 18))
    full_U[:, :7] = base_U
    full_V[:, :7] = base_V
    full_W[:, :7] = base_W
    full_U[:, 7:14] = base_U2
    full_V[:, 7:14] = base_V2
    full_W[:, 7:14] = base_W2
    # Schoolbook col 4
    full_U[0, 14] = 1; full_V[0*p+4, 14] = 1; full_W[4*n+0, 14] = 1
    full_U[1, 15] = 1; full_V[1*p+4, 15] = 1; full_W[4*n+0, 15] = 1
    full_U[2, 16] = 1; full_V[0*p+4, 16] = 1; full_W[4*n+1, 16] = 1
    full_U[3, 17] = 1; full_V[1*p+4, 17] = 1; full_W[4*n+1, 17] = 1
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        if trial % 3 == 0:
            # Keep a random subset of the 18 components
            indices = np.random.choice(18, target_rank, replace=False)
            U[:, :] = full_U[:, indices]
            V[:, :] = full_V[:, indices]
            W[:, :] = full_W[:, indices]
        elif trial % 3 == 1:
            # Start with first target_rank components and add small noise
            U[:, :] = full_U[:, :target_rank]
            V[:, :] = full_V[:, :target_rank]
            W[:, :] = full_W[:, :target_rank]
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
        else:
            # Syndrome-based initialization for target rank
            U, V, W = _syndrome_init(n, m, p, target_rank, trial)
            
        U, V, W, err = _als_refine(U, V, W, T, target_rank, n, m, p, max_iter=150, deadline=deadline)
        
        if err < 1e-8:
            result = _check_and_rationalize(U, V, W, T, target_rank, n, m, p)
            if result is not None:
                return result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try Rank 15 - aggressive search with tight budget
    remaining = deadline - time.time() - 20
    if remaining > 5:
        t15_budget = min(20, remaining)
        result = _search_rank_aggressive(n, m, p, 15, timeout=t15_budget, n_trials=50)
        if result is not None:
            return result
    
    # Try Rank 16
    remaining = deadline - time.time() - 20
    if remaining > 5:
        t16_budget = min(20, remaining)
        result = _search_rank_aggressive(n, m, p, 16, timeout=t16_budget, n_trials=50)
        if result is not None:
            return result
            
    # Try Rank 17 via Perturbation
    remaining = deadline - time.time() - 15
    if remaining > 5:
        t17_budget = min(20, remaining)
        result = _try_perturb_rank(n, m, p, 17, timeout=t17_budget, n_trials=100)
        if result is not None:
            return result

    # Try Rank 17 - standard aggressive search
    remaining = deadline - time.time() - 15
    if remaining > 5:
        t17_budget = min(15, remaining)
        result = _search_rank_aggressive(n, m, p, 17, timeout=t17_budget, n_trials=30)
        if result is not None:
            return result
    
    # Fallback Rank 18 (Fast and Guaranteed)
    return _construct_rank18(n, m, p)
