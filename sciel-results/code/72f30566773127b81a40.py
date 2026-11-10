# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    idx_a = X.astype(np.int64).ravel()
    idx_b = (X + Y).astype(np.int64).ravel()
    idx_c = (X - Y + 9).astype(np.int64).ravel()
    idx_d = (X + 2 * Y).astype(np.int64).ravel()
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

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def _soft_max_entropies(qa, qb, qd, temp=0.05):
    ea = _entropy(qa)
    eb = _entropy(qb)
    ed = _entropy(qd)
    arr = np.array([ea, eb, ed])
    if np.max(arr) < 1e-12:
        return 1e-12
    scaled = arr / temp
    max_s = np.max(scaled)
    logsumexp = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return temp * logsumexp

def _optimize_weights_for_support(mask, idx_a, idx_b, idx_c, idx_d, max_iter=2000):
    support_indices = np.where(mask.ravel() > 0)[0]
    k = len(support_indices)
    if k == 0:
        return 0.0, None
    
    z0 = np.zeros(k)
    
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    def _compute_q(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        w = exp_z / exp_z.sum()
        pf = np.zeros(100)
        pf[support_indices] = w
        qa = np.bincount(idx_a, weights=pf, minlength=la)
        qb = np.bincount(idx_b, weights=pf, minlength=lb)
        qc = np.bincount(idx_c, weights=pf, minlength=lc)
        qd = np.bincount(idx_d, weights=pf, minlength=ld)
        return qa, qb, qc, qd, w, pf

    def objective(z):
        qa, qb, qc, qd, w, pf = _compute_q(z)
        D = _entropy(qc)
        M = _soft_max_entropies(qa, qb, qd, temp=0.05)
        if M < 1e-12:
            return 1e9
        if D < 1e-12:
            return 1e9
        return -(np.log(D) - np.log(M))

    try:
        res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            z_opt = res.x
            z_shifted = z_opt - np.max(z_opt)
            exp_z = np.exp(z_shifted)
            w = exp_z / exp_z.sum()
            pf = np.zeros(100)
            pf[support_indices] = w
            P = pf.reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return val, P
        else:
            w = np.ones(k) / k
            pf = np.zeros(100)
            pf[support_indices] = w
            P = pf.reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return val, P
    except Exception:
        return 0.0, None

def _uniform_value(mask, idx_a, idx_b, idx_c, idx_d):
    pf = np.zeros(100)
    m_flat = mask.ravel()
    cnt = m_flat.sum()
    if cnt == 0:
        return -1e9
    pf[m_flat] = 1.0 / cnt
    P = pf.reshape(10, 10)
    return _value_from_P(P, idx_a, idx_b, idx_c, idx_d)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    best_val = -1e9
    best_P = np.full((10, 10), 1.0 / 100.0)
    
    # Initialize with uniform distribution
    val_u = _value_from_P(best_P, idx_a, idx_b, idx_c, idx_d)
    if val_u > best_val:
        best_val = val_u
        best_P = best_P.copy()
        
    # Deterministic Global Coordinate Ascent
    # Strategy: 
    # 1. Start with a reasonable mask (e.g., all cells, or a specific pattern)
    # 2. Iteratively:
    #    a. Optimize weights for the current mask
    #    b. Try all pairwise swaps (i in mask, j not in mask) to find the best improvement
    #    c. If a swap improves the value (after re-optimizing weights for the new mask), accept it
    #    d. Repeat until no improvement or time limit
    
    # Initial mask: Start with all cells active? Or maybe a subset?
    # Let's try starting with a mask that has high D/M potential.
    # Uniform on all cells gives some baseline.
    # Let's try a few initial masks to find a good starting point for the coordinate ascent.
    
    rng = np.random.default_rng(42)
    
    initial_masks = []
    # 1. All ones
    initial_masks.append(np.ones((10, 10), dtype=bool))
    # 2. Main diagonal
    m_diag = np.zeros((10, 10), dtype=bool)
    m_diag[np.arange(10), np.arange(10)] = True
    initial_masks.append(m_diag)
    # 3. Anti-diagonal
    m_anti = np.zeros((10, 10), dtype=bool)
    m_anti[np.arange(10), 9 - np.arange(10)] = True
    initial_masks.append(m_anti)
    # 4. Random sparse
    m_rand = rng.random((10, 10)) > 0.7
    if m_rand.sum() < 2: m_rand.flat[0] = True; m_rand.flat[1] = True
    initial_masks.append(m_rand)
    # 5. Random dense
    m_dense = rng.random((10, 10)) > 0.2
    initial_masks.append(m_dense)

    best_start_mask = None
    best_start_val = -1e9
    best_start_P = None

    for mask in initial_masks:
        if time.time() > deadline - 100.0: break
        val, P = _optimize_weights_for_support(mask, idx_a, idx_b, idx_c, idx_d)
        if P is not None and val > best_start_val:
            best_start_val = val
            best_start_mask = mask.copy()
            best_start_P = P.copy()
            if val > best_val:
                best_val = val
                best_P = P.copy()

    if best_start_mask is None:
        best_start_mask = np.ones((10, 10), dtype=bool)
        val, P = _optimize_weights_for_support(best_start_mask, idx_a, idx_b, idx_c, idx_d)
        best_start_val = val
        best_start_P = P

    # Coordinate Ascent Loop
    current_mask = best_start_mask.copy()
    current_P = best_start_P.copy()
    current_val = best_start_val
    
    # Precompute indices for speed
    all_indices = np.arange(100)
    
    while time.time() < deadline - 20.0:
        # 1. Optimize weights for current mask
        val_opt, P_opt = _optimize_weights_for_support(current_mask, idx_a, idx_b, idx_c, idx_d)
        if P_opt is not None and val_opt > current_val:
            current_val = val_opt
            current_P = P_opt
            if current_val > best_val:
                best_val = current_val
                best_P = current_P.copy()
        
        # 2. Find best swap
        # Get list of active and inactive indices
        active_indices = np.where(current_mask.ravel())[0]
        inactive_indices = np.where(~current_mask.ravel())[0]
        
        if len(active_indices) == 0 or len(inactive_indices) == 0:
            break
            
        best_swap_val = current_val
        best_swap_i = -1
        best_swap_j = -1
        best_swap_P = None
        
        # To save time, we can't try ALL pairs if the sets are large (50*50=2500)
        # 2500 evaluations * (L-BFGS cost) might be slow.
        # L-BFGS takes ~0.1s? 2500 * 0.1s = 250s. That's too slow for one iteration.
        # We need to be smarter.
        
        # Strategy: 
        # - Use a cheaper surrogate for the swap evaluation?
        # - Or limit the number of swaps tried per iteration?
        # - Or use the uniform value for the new mask to pre-filter?
        
        # Let's try a heuristic: 
        # Evaluate uniform value for new mask first. If uniform value is not promising, skip L-BFGS.
        # But uniform value is a lower bound on the optimized value? Not necessarily, but often correlated.
        
        # Actually, let's just limit the number of swaps we test per iteration to keep it fast.
        # Or, we can test a random subset of swaps.
        
        # Let's try testing up to 50 random swaps per iteration.
        num_swaps_to_test = 50
        if len(active_indices) * len(inactive_indices) <= num_swaps_to_test:
            pairs_to_test = list(itertools.product(active_indices, inactive_indices))
        else:
            # Random sampling
            pairs_to_test = []
            for _ in range(num_swaps_to_test):
                i = active_indices[rng.integers(0, len(active_indices))]
                j = inactive_indices[rng.integers(0, len(inactive_indices))]
                pairs_to_test.append((i, j))
        
        for i, j in pairs_to_test:
            if time.time() > deadline - 20.0: break
            
            new_mask = current_mask.copy()
            new_mask.ravel()[i] = False
            new_mask.ravel()[j] = True
            
            # Quick check: uniform value
            val_uni = _uniform_value(new_mask, idx_a, idx_b, idx_c, idx_d)
            
            # If uniform value is very bad, skip optimization
            # What is a "good" uniform value? 
            # Let's say if val_uni < current_val * 0.8, skip?
            # Or just try optimizing if val_uni is not terrible.
            
            # To be safe and fast, let's only optimize if val_uni > best_swap_val * 0.9 
            # OR if we are early in the search.
            
            # Actually, let's just optimize a few of the best uniform candidates.
            # This requires sorting.
            
            # Alternative: Just optimize the first few swaps that look promising based on uniform value.
            # Let's collect candidates.
            
        # Refining the swap search:
        # Collect candidates based on uniform value
        candidates = []
        for i, j in pairs_to_test:
            new_mask = current_mask.copy()
            new_mask.ravel()[i] = False
            new_mask.ravel()[j] = True
            val_uni = _uniform_value(new_mask, idx_a, idx_b, idx_c, idx_d)
            candidates.append((val_uni, i, j, new_mask))
            
        # Sort by uniform value descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Try optimizing the top 5 candidates
        for val_uni, i, j, new_mask in candidates[:5]:
            if time.time() > deadline - 20.0: break
            
            # If the uniform value is already worse than the current best, and the gap is large, skip?
            # Let's just try optimizing the top ones.
            
            val_opt, P_opt = _optimize_weights_for_support(new_mask, idx_a, idx_b, idx_c, idx_d)
            if P_opt is not None and val_opt > best_swap_val:
                best_swap_val = val_opt
                best_swap_i = i
                best_swap_j = j
                best_swap_P = P_opt
                best_swap_mask = new_mask.copy()
        
        if best_swap_i != -1:
            # Accept the swap
            current_mask = best_swap_mask
            current_P = best_swap_P
            current_val = best_swap_val
            if current_val > best_val:
                best_val = current_val
                best_P = current_P.copy()
        else:
            # No improvement found in this iteration
            # Try a different random subset of swaps? Or break?
            # Let's try a few more random subsets before breaking to avoid local minima.
            # But to keep it simple, let's break if we don't find an improvement.
            # Actually, let's try one more time with different random swaps.
            # If that fails, break.
            
            # Simple break for now to ensure termination
            break

    # Final polish
    if time.time() < deadline - 10.0:
        best_support = best_P > 1e-6
        if best_support.sum() > 0:
            val_final, P_final_opt = _optimize_weights_for_support(best_support, idx_a, idx_b, idx_c, idx_d, max_iter=5000)
            if P_final_opt is not None and val_final > best_val:
                best_val = val_final
                best_P = P_final_opt.copy()

    P_final = _finalize(best_P, idx_a, idx_b, idx_c, idx_d)
    return P_final

# EVOLVE-BLOCK-END
