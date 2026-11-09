# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _optimize_weights_on_support(support, idx_a, idx_b, idx_c, idx_d, timeout=5.0):
    """
    Given a fixed set of indices (support), optimize the weights on them to maximize D/M.
    We use a parameterization w_i = exp(z_i) / sum(exp(z_j)) restricted to the support.
    Since the support size is small (K), this is a low-dimensional optimization.
    """
    K = len(support)
    if K == 0:
        return None, 0.0
        
    # Pre-compute bin counts for the support
    # For each target variable, we need to know which bins the support cells fall into
    # and how many cells in the support fall into each bin.
    # Actually, it's easier to just compute the full bincounts for the 100x1 vector
    # but only the non-zero parts matter.
    
    def objective(z):
        # z is length K
        z = z - np.max(z)
        exp_z = np.exp(z)
        w = exp_z / exp_z.sum()
        
        P = np.zeros(100)
        P[support] = w
        
        pf = P
        la = idx_a.max() + 1
        lb = idx_b.max() + 1
        lc = idx_c.max() + 1
        ld = idx_d.max() + 1
        
        # Efficient bincount for sparse P
        # We can just use np.bincount on the full index arrays with weights P
        # Since P is mostly 0, this is fast.
        qa = np.bincount(idx_a, weights=pf, minlength=la)
        qb = np.bincount(idx_b, weights=pf, minlength=lb)
        qc = np.bincount(idx_c, weights=pf, minlength=lc)
        qd = np.bincount(idx_d, weights=pf, minlength=ld)
        
        D = _entropy(qc)
        M = max(_entropy(qa), _entropy(qb), _entropy(qd))
        
        if M < 1e-12:
            return -1e9
        return -D / M # Minimize negative ratio
        
    # Initial guess: uniform
    z0 = np.zeros(K)
    
    # Use Nelder-Mead for small K (robust to non-smoothness of max)
    # Or L-BFGS-B if we implement gradient, but NM is safer for small K and no grad needed.
    # Given K <= 30, NM might be slow if K is large. Let's use L-BFGS-B with numerical grad? 
    # No, let's stick to a simple local search or NM for K < 15, and maybe uniform for larger?
    # Actually, let's try L-BFGS-B with a gradient-free wrapper if K is small, or just NM.
    # NM is O(K) evaluations per step. K=10 is fine. K=30 is okay.
    
    try:
        res = minimize(objective, z0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        best_val = -res.fun
        P_opt = np.zeros(100)
        z_opt = res.x
        z_opt = z_opt - np.max(z_opt)
        exp_z = np.exp(z_opt)
        w = exp_z / exp_z.sum()
        P_opt[support] = w
        return P_opt, best_val
    except Exception:
        return None, 0.0

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0
    
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    best_P = None
    best_val = -1e9
    
    # Strategy: Explore different support sizes and random supports.
    # Also try structured supports (diagonals, etc.)
    
    # 1. Try small supports (K=1 to 15)
    # A support of size K can have at most K distinct values for x, x+y, etc.
    # To maximize D (entropy of x-y), we want the x-y values to be as spread out as possible.
    # Max D is log2(K) if all x-y values are distinct.
    # To minimize M, we want the x, x+y, x+2y values to be concentrated.
    
    # Let's generate random supports of various sizes
    rng = np.random.default_rng(42)
    
    support_sizes = [5, 10, 15, 20, 25, 30]
    
    for K in support_sizes:
        if time.time() > deadline - 30.0:
            break
            
        # Number of trials per size
        # More trials for smaller K, fewer for larger
        trials = 200 if K <= 15 else 50
        
        for _ in range(trials):
            if time.time() > deadline - 10.0:
                break
                
            # Random support
            support = rng.choice(100, size=K, replace=False)
            
            P_opt, val = _optimize_weights_on_support(support, idx_a, idx_b, idx_c, idx_d, timeout=2.0)
            
            if P_opt is not None and val > best_val:
                best_val = val
                best_P = P_opt.copy()
                
            # Also try uniform weights on this support (quick check)
            P_uni = np.zeros(100)
            P_uni[support] = 1.0 / K
            val_uni = _value_from_P(P_uni, idx_a, idx_b, idx_c, idx_d)
            if val_uni > best_val:
                best_val = val_uni
                best_P = P_uni.copy()

    # 2. Try structured supports: Diagonals of x-y
    # Cells with same x-y are on a diagonal.
    # If we pick one cell from each diagonal, x-y values are all distinct (0..9 shifted).
    # This maximizes D = log2(10) = ~3.32.
    # We want to pick 10 cells, one from each diagonal (x-y = const), such that
    # x, x+y, x+2y are concentrated.
    
    # Diagonals: x - y = d for d in -9..9. Shifted: 0..18.
    # There are 19 diagonals. We can only have at most 10 cells if we want distinct x-y?
    # No, x-y ranges from -9 to 9. So 19 possible values.
    # If we pick 10 cells with distinct x-y, D <= log2(10).
    # If we pick cells such that x is constant (vertical line), M = 0 (invalid).
    # If we pick cells such that x+y is constant (diagonal), M = 0 (invalid).
    
    # Let's try to pick one cell from each of 10 distinct diagonals.
    # We can enumerate subsets of diagonals? 19 choose 10 is too big.
    # Let's use a greedy/random approach for structured supports.
    
    # Generate all cells
    cells = np.array([(x, y) for y in range(10) for x in range(10)])
    diag_ids = cells[:, 0] - cells[:, 1] + 9 # 0..18
    
    # For each diagonal, we have multiple cells.
    # Let's try to build a support by picking one cell from each of K distinct diagonals.
    # We'll use a random selection of K distinct diagonals, then pick a random cell from each.
    
    for K in [10, 12, 15]:
        if time.time() > deadline - 10.0:
            break
            
        trials = 100
        for _ in range(trials):
            # Pick K distinct diagonals
            diags = rng.choice(19, size=K, replace=False)
            support = []
            for d in diags:
                # Find cells with this diagonal
                mask = diag_ids == d
                cell_indices = np.where(mask)[0]
                if len(cell_indices) > 0:
                    # Pick a random cell from this diagonal
                    c = cell_indices[rng.integers(len(cell_indices))]
                    support.append(c)
            
            support = np.array(support)
            P_uni = np.zeros(100)
            P_uni[support] = 1.0 / K
            val_uni = _value_from_P(P_uni, idx_a, idx_b, idx_c, idx_d)
            
            if val_uni > best_val:
                best_val = val_uni
                best_P = P_uni.copy()
                
            # Refine with optimizer
            P_opt, val = _optimize_weights_on_support(support, idx_a, idx_b, idx_c, idx_d, timeout=2.0)
            if P_opt is not None and val > best_val:
                best_val = val
                best_P = P_opt.copy()

    # 3. Fallback to uniform if nothing found
    if best_P is None:
        best_P = np.full(100, 1.0/100.0)
        
    # Finalize
    P = best_P.reshape(10, 10)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.full((10, 10), 1.0 / 100.0)
    else:
        P = P / s
        
    # Verify validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        P = np.full((10, 10), 1.0 / 100.0)
        
    return P.astype(np.float64)

# EVOLVE-BLOCK-END
