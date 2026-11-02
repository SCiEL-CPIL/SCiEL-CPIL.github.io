"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify_full(U, V, W, n, m, p):
    """Full verification: T[i,j,k]=1 iff indices match, else 0."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    Ue = U_arr.reshape(n, m, 1, r)
    Ve = V_arr.reshape(1, m, p, r)
    We = W_arr.T.reshape(n, 1, p, r)
    T = np.sum(Ue * Ve * We, axis=3)
    # T[i,j,k] should be 1 for all valid i,j,k
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix multiplication."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    # For each (i,j), compute sum over k of (sum_r U[i,j,r]*V[j,k,r])*B_ref[j,k]
    # where B_ref is identity-like, but we need T[i,j,k]=1
    # Alternative: compute C[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Use einsum for speed
    T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U_arr, V_arr, W_arr)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _random_rank1_attempt(n, m, p, rng, deadline):
    """
    Try to find a rank-1 tensor (u,v,w) that contributes to the decomposition.
    u: shape (n*m,), v: shape (m*p,), w: shape (p*n,)
    The product u[i*m+j] * v[j*p+k] * w[k*n+i] should approximate 1 for all (i,j,k)
    """
    # Strategy: pick random sparse patterns
    density = rng.choice([0.3, 0.5, 0.7])
    u = np.zeros(n * m)
    v = np.zeros(m * p)
    w = np.zeros(p * n)
    
    # Assign random values to some positions
    for idx in range(n * m):
        if rng.random() < density:
            u[idx] = rng.choice([-1, 1])
    for idx in range(m * p):
        if rng.random() < density:
            v[idx] = rng.choice([-1, 1])
    for idx in range(p * n):
        if rng.random() < density:
            w[idx] = rng.choice([-1, 1])
    
    # Compute the resulting tensor
    T = np.einsum('ij,kl,ki->ijk', u.reshape(n, m), v.reshape(m, p), w.reshape(p, n))
    # Check if it's close to ones (it won't be, but we track correlation)
    corr = np.sum(T) / (n * m * p)
    return u, v, w, corr

def _greedy_build(n, m, p, max_rank, deadline, rng):
    """
    Greedy construction: iteratively add rank-1 terms to minimize residual.
    """
    target = np.ones((n, m, p))
    residual = target.copy()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    start_time = time.time()
    
    for step in range(max_rank):
        if time.time() > deadline:
            break
        if np.allclose(residual, 0, atol=1e-10):
            break
        
        # Try several random rank-1 terms and pick the best
        best_corr = -1
        best_u, best_v, best_w = None, None, None
        
        n_attempts = 50
        for _ in range(n_attempts):
            u, v, w, corr = _random_rank1_attempt(n, m, p, rng, deadline)
            if corr > best_corr:
                best_corr = corr
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None or best_corr <= 0:
            break
        
        # Compute the tensor contribution
        contrib = np.einsum('ij,kl,ki->ijk',
                           best_u.reshape(n, m),
                           best_v.reshape(m, p),
                           best_w.reshape(p, n))
        
        # Scale to match residual best: find alpha that minimizes ||residual - alpha*contrib||^2
        alpha = np.sum(residual * contrib) / (np.sum(contrib * contrib) + 1e-12)
        
        # Update residual
        residual -= alpha * contrib
        
        # Store scaled factors
        U_cols.append(best_u * np.sqrt(abs(alpha)) if alpha > 0 else -best_u * np.sqrt(abs(alpha)))
        V_cols.append(best_v * np.sqrt(abs(alpha)) if alpha > 0 else -best_v * np.sqrt(abs(alpha)))
        W_cols.append(best_w * np.sign(alpha))
    
    if not U_cols:
        return None
    
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return U, V, W, len(U_cols)

def _linear_algebra_approach(n, m, p, max_rank, deadline, rng):
    """
    Use linear algebra to find dependencies in random rank-r tensors.
    Generate random rank-1 terms, stack them, find null space to get relations.
    """
    start_time = time.time()
    target = np.ones((n, m, p))
    
    # Generate many random rank-1 tensors
    n_samples = min(200, max_rank * 10)
    if time.time() > deadline - 10:
        n_samples = min(n_samples, 20)
    
    samples = []
    for _ in range(n_samples):
        if time.time() > deadline:
            break
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        # Random sparse with values in {-1, 0, 1}
        for idx in range(n * m):
            if rng.random() < 0.5:
                u[idx] = rng.choice([-1, 1])
        for idx in range(m * p):
            if rng.random() < 0.5:
                v[idx] = rng.choice([-1, 1])
        for idx in range(p * n):
            if rng.random() < 0.5:
                w[idx] = rng.choice([-1, 1])
        
        T = np.einsum('ij,kl,ki->ijk', u.reshape(n, m), v.reshape(m, p), w.reshape(p, n))
        samples.append((u, v, w, T))
    
    if not samples:
        return None
    
    # We want to find coefficients c_1, ..., c_s such that sum c_s * T_s = target
    # Flatten all tensors
    M = np.zeros((n * m * p, len(samples)))
    for s_idx, (u, v, w, T) in enumerate(samples):
        M[:, s_idx] = T.flatten()
    
    b = target.flatten()
    
    # Solve least squares: M @ coeffs = b
    coeffs, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
    
    # Check residual
    reconstructed = M @ coeffs
    if not np.allclose(reconstructed, b, atol=1e-8):
        return None
    
    # Find a sparse subset: use greedy selection
    abs_coeffs = np.abs(coeffs)
    order = np.argsort(-abs_coeffs)
    
    U_cols = []
    V_cols = []
    W_cols = []
    residual = b.copy()
    used = set()
    
    for idx in order:
        if time.time() > deadline:
            break
        if abs(coeffs[idx]) < 1e-6:
            continue
        if idx in used:
            continue
        
        u, v, w, T = samples[idx]
        T_flat = T.flatten()
        
        # Check if this sample helps reduce residual
        corr = np.sum(residual * T_flat) / (np.sum(T_flat * T_flat) + 1e-12)
        if abs(corr) < 1e-6:
            continue
        
        residual -= corr * T_flat
        used.add(idx)
        
        U_cols.append(u * np.sqrt(abs(corr)) if corr > 0 else -u * np.sqrt(abs(corr)))
        V_cols.append(v * np.sqrt(abs(corr)) if corr > 0 else -v * np.sqrt(abs(corr)))
        W_cols.append(w * np.sign(corr))
        
        if len(used) >= max_rank:
            break
        if np.allclose(residual, 0, atol=1e-8):
            break
    
    if not U_cols:
        return None
    
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return U, V, W, len(U_cols)

def _structured_decomposition(n, m, p):
    """
    Try specific structured decompositions known to be efficient for small matrices.
    For 3x4x4, exploit that n=3 is small.
    
    Strategy: decompose by fixing i (3 values).
    For each i, C[i,:] = A[i,:] @ B is a 1x4 times 4x4 = 1x4 vector.
    This requires 4 multiplications per i, total 3*4 = 12 if we can share.
    But B is 4x4, so each row of A (length 4) times B (4x4) needs 16 mults.
    Total schoolbook: 3*4*4 = 48.
    
    Try to share computation across i values.
    """
    # Baseline schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    best_rank = rank
    
    # Try greedy build with various rank limits
    for max_r in [24, 20, 18, 16, 14, 12]:
        if time.time() > deadline - 20:
            break
        result = _greedy_build(n, m, p, max_r, deadline - 10, rng)
        if result:
            U_c, V_c, W_c, r_c = result
            if _verify_full(U_c, V_c, W_c, n, m, p):
                if r_c < best_rank:
                    best = (U_c, V_c, W_c, r_c)
                    best_rank = r_c
                break
    
    # Try linear algebra approach
    if time.time() < deadline - 30:
        result = _linear_algebra_approach(n, m, p, 24, deadline - 10, rng)
        if result:
            U_c, V_c, W_c, r_c = result
            if _verify_full(U_c, V_c, W_c, n, m, p):
                if r_c < best_rank:
                    best = (U_c, V_c, W_c, r_c)
                    best_rank = r_c
    
    # Try structured
    if time.time() < deadline - 5:
        U_s, V_s, W_s, r_s = _structured_decomposition(n, m, p)
        if r_s < best_rank and _verify_full(U_s, V_s, W_s, n, m, p):
            best = (U_s, V_s, W_s, r_s)
            best_rank = r_s
    
    U_best, V_best, W_best, _ = best
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
