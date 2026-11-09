"""Matrix multiplication 4x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _err_fast(U, V, W, idx_u, idx_v, idx_w):
    U_sel = U[idx_u, :]
    V_sel = V[idx_v, :]
    W_sel = W[idx_w, :]
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    return float(np.max(np.abs(vals - 1.0)))

def _err_sq(U, V, W, idx_u, idx_v, idx_w):
    U_sel = U[idx_u, :]
    V_sel = V[idx_v, :]
    W_sel = W[idx_w, :]
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    return float(np.sum((vals - 1.0) ** 2))

def _kronecker_init(target_rank, n, m, p, variant=0):
    """Structured initialization using Kronecker/block structure."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if variant == 0:
        # Block-structured: group by (i, j, k) with systematic assignment
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
    elif variant == 1:
        # Kronecker-inspired: separate row and column structure
        # U ~ I_n ⊗ J_m, V ~ J_m ⊗ I_p
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j, c] = 1.0 + 0.3 * (j % 2)
            V[j * p + k, c] = 1.0 + 0.3 * (k % 2)
            W[k * n + i, c] = 1.0
    elif variant == 2:
        # Diagonal emphasis with off-diagonal coupling
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            if i + 1 < n:
                U[(i + 1) * m + j, c] += 0.25
            if j + 1 < m:
                U[i * m + j + 1, c] += 0.25
            if k + 1 < p:
                V[j * p + k + 1, c] += 0.25
    elif variant == 3:
        # Two-scale: fine + coarse
        half = target_rank // 2
        for c in range(min(half, target_rank)):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        for c in range(half, target_rank):
            c2 = c - half
            i = c2 % n
            j = (c2 // n) % m
            k = (c2 // (n * m)) % p
            U[i * m + j, c] = 0.7
            V[j * p + k, c] = 0.7
            W[k * n + i, c] = 0.7
            if i < n - 1:
                U[(i + 1) * m + j, c] += 0.2
            if j < m - 1:
                U[i * m + j + 1, c] += 0.2
    elif variant == 4:
        # Random structured with block bias
        rng = np.random.RandomState(42 * variant + target_rank)
        for c in range(target_rank):
            r = rng.random()
            if r < 0.65:
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            else:
                ui = rng.randint(0, n * m)
                vi = rng.randint(0, m * p)
                wi = rng.randint(0, p * n)
                U[ui, c] = 1.0
                V[vi, c] = 1.0
                W[wi, c] = 1.0
    else:
        # Pure random
        rng = np.random.RandomState(variant * 7919 + target_rank * 31)
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
    
    return U, V, W

def _als_step_U(U, V, W, n, m, p, R):
    """Update U using ALS."""
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # b[k] = 1
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            # Solve A x = b
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                U_new[i * m + j, :] = x
            except np.linalg.LinAlgError:
                pass
    return U_new

def _als_step_V(U, V, W, n, m, p, R):
    """Update V using ALS."""
    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # c[i] = 1
            B = np.zeros((n, R))
            for i in range(n):
                B[i, :] = U[i * m + j, :] * W[k * n + i, :]
            c = np.ones(n)
            try:
                x, residuals, rank, sv = np.linalg.lstsq(B, c, rcond=None)
                V_new[j * p + k, :] = x
            except np.linalg.LinAlgError:
                pass
    return V_new

def _als_step_W(U, V, W, n, m, p, R):
    """Update W using ALS."""
    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # d[j] = 1
            C = np.zeros((m, R))
            for j in range(m):
                C[j, :] = U[i * m + j, :] * V[j * p + k, :]
            d = np.ones(m)
            try:
                x, residuals, rank, sv = np.linalg.lstsq(C, d, rcond=None)
                W_new[k * n + i, :] = x
            except np.linalg.LinAlgError:
                pass
    return W_new

def _alternating_minimize(U, V, W, idx_u, idx_v, idx_w, deadline, max_iter=500):
    """Alternating Least Squares optimization."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    cur_err = _err_fast(U, V, W, idx_u, idx_v, idx_w)
    best_err = cur_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 8:
            break
        if cur_err < 1e-13:
            break
        
        # ALS steps
        U = _als_step_U(U, V, W, n, m, p, R)
        V = _als_step_V(U, V, W, n, m, p, R)
        W = _als_step_W(U, V, W, n, m, p, R)
        
        new_err = _err_fast(U, V, W, idx_u, idx_v, idx_w)
        if new_err < best_err:
            best_err = new_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if abs(new_err - cur_err) < 1e-10:
            break
        cur_err = new_err
    
    return best_U, best_V, best_W, best_err

def _polish(U, V, W, idx_u, idx_v, idx_w, deadline, max_iter=300):
    """High-precision polish using ALS with smaller steps or just more ALS."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    cur_err = _err_fast(U, V, W, idx_u, idx_v, idx_w)
    best_err = cur_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if cur_err < 1e-14:
            break
        
        # Use ALS again for polish
        U = _als_step_U(U, V, W, n, m, p, R)
        V = _als_step_V(U, V, W, n, m, p, R)
        W = _als_step_W(U, V, W, n, m, p, R)
        
        new_err = _err_fast(U, V, W, idx_u, idx_v, idx_w)
        if new_err < best_err:
            best_err = new_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if abs(new_err - cur_err) < 1e-12:
            break
        cur_err = new_err
        
    return best_U, best_V, best_W, best_err

def _try_rank(target_rank, num_attempts=8, deadline=None):
    """Try to find a rank decomposition at target_rank."""
    n, m, p = DIMENSIONS
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    best_overall_err = float('inf')
    best_overall_U = best_overall_V = best_overall_W = None
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 12:
            break
        
        variant = attempt % 5
        U, V, W = _kronecker_init(target_rank, n, m, p, variant)
        
        # Add small perturbation to break symmetry
        rng = np.random.RandomState(attempt * 12345 + target_rank * 67)
        U += rng.randn(n * m, target_rank) * 0.08
        V += rng.randn(m * p, target_rank) * 0.08
        W += rng.randn(p * n, target_rank) * 0.08
        
        # Phase 1: coarse optimization
        U, V, W, err = _alternating_minimize(U, V, W, idx_u, idx_v, idx_w, deadline, max_iter=400)
        
        if err < best_overall_err:
            best_overall_err = err
            best_overall_U, best_overall_V, best_overall_W = U.copy(), V.copy(), W.copy()
        
        # Phase 2: polish if close
        if err < 1e-3:
            U2, V2, W2, err2 = _polish(U, V, W, idx_u, idx_v, idx_w, deadline, max_iter=200)
            if err2 < best_overall_err:
                best_overall_err = err2
                best_overall_U, best_overall_V, best_overall_W = U2.copy(), V2.copy(), W2.copy()
        
        if best_overall_err < 1e-8:
            break
    
    if best_overall_err < 1e-7:
        # Round to half-integers and verify
        U_r = np.round(best_overall_U * 2) / 2
        V_r = np.round(best_overall_V * 2) / 2
        W_r = np.round(best_overall_W * 2) / 2
        
        err_r = _err_fast(U_r, V_r, W_r, idx_u, idx_v, idx_w)
        if err_r < 1e-9:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    
    # Try ranks from lower bound upward
    # Prioritize the range where solutions are likely to exist (90-110)
    ranks_to_try = list(range(90, 111)) + list(range(111, 121)) + list(range(50, 90))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        if 90 <= target_rank <= 110:
            num_attempts = 15
        elif target_rank < 90:
            num_attempts = 5
        else:
            num_attempts = 8
        
        result = _try_rank(target_rank, num_attempts=num_attempts, deadline=deadline)
        if result is not None:
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
