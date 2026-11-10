import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes D/M where D = H(x-y), M = max(H(x), H(x+y), H(x+2y)).
    Combines structured initializations, smooth surrogate Nelder-Mead, 
    and exact SLSQP polishing.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]

    # Flatten for binning
    d_val = (xs - ys).flatten()       # x - y
    m0_val = xs.flatten()              # x
    m1_val = (xs + ys).flatten()       # x + y
    m2_val = (xs + 2 * ys).flatten()   # x + 2y

    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)

    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def shannon(p_dist):
        p_nonzero = p_dist[p_dist > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    def log_softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    def objective_smooth(z, temperature=0.1):
        """Objective using smooth max surrogate in log-space"""
        p_flat = log_softmax(z)
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        # Smooth max using log-sum-exp
        entropies = np.array([H_m0, H_m1, H_m2])
        max_e = np.max(entropies)
        if max_e == 0:
            M_smooth = 0.0
        else:
            # Temperature annealing helps navigate the kink
            M_smooth = temperature * np.log(np.sum(np.exp((entropies - max_e) / temperature))) + max_e
            
        if M_smooth < 1e-9:
            return 1e9
        return -H_d / M_smooth

    def objective_exact(z):
        """Exact objective in log-space"""
        p_flat = log_softmax(z)
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return 1e9
        return -H_d / M

    # --- Build structured starting points ---
    starts = []
    rng = np.random.default_rng(42)

    # 1. Uniform
    starts.append(np.zeros(100))  # log-softmax of zeros = uniform

    # 2. Single x-line (columns)
    for x0 in range(n):
        z = np.full(100, -100.0)
        z[x0 * 10:(x0 + 1) * 10] = 0.0
        starts.append(z)

    # 3. Single y-line (rows)
    for y0 in range(n):
        z = np.full(100, -100.0)
        z[y0 * 10:(y0 + 1) * 10] = 0.0
        starts.append(z)

    # 4. Main diagonal
    z_diag = np.full(100, -100.0)
    for i in range(n):
        z_diag[i * 10 + i] = 0.0
    starts.append(z_diag)

    # 5. Anti-diagonal
    z_adiag = np.full(100, -100.0)
    for i in range(n):
        z_adiag[i * 10 + (9 - i)] = 0.0
    starts.append(z_adiag)

    # 6. Two x-lines (spread in x, fixed in y? No, fixed x means spread y)
    # Trying to balance entropies
    for x0 in range(n):
        for x1 in range(x0 + 1, n):
            z = np.full(100, -100.0)
            z[x0 * 10:(x0 + 1) * 10] = 0.0
            z[x1 * 10:(x1 + 1) * 10] = 0.0
            starts.append(z)
            
    # 7. Two y-lines
    for y0 in range(n):
        for y1 in range(y0 + 1, n):
            z = np.full(100, -100.0)
            z[y0 * 10:(y0 + 1) * 10] = 0.0
            z[y1 * 10:(y1 + 1) * 10] = 0.0
            starts.append(z)

    # 8. Random Dirichlet converted to log-softmax
    for trial in range(80):
        alpha = rng.uniform(0.1, 5.0, 100)
        p_flat = rng.dirichlet(alpha)
        z = np.log(p_flat + 1e-15)
        starts.append(z)

    # 9. Checkerboard
    for offset in range(2):
        z = np.full(100, -100.0)
        for y in range(n):
            for x in range(n):
                if (x + y) % 2 == offset:
                    z[y * 10 + x] = 0.0
        starts.append(z)

    # 10. Slope patterns: x + r*y = c
    for r in [-2, -1, 0, 1, 2, 3]:
        for c in range(-10, 20):
            z = np.full(100, -100.0)
            count = 0
            for y in range(n):
                for x in range(n):
                    if x + r * y == c:
                        z[y * 10 + x] = 0.0
                        count += 1
            if count > 1: # Avoid single point
                starts.append(z)

    # 11. Sparse random subsets
    for subset_size in [3, 5, 8, 12]:
        for trial in range(10):
            indices = rng.choice(100, size=subset_size, replace=False)
            z = np.full(100, -100.0)
            z[indices] = 0.0
            starts.append(z)

    # 12. Greedy construction heuristic
    # We try to build a set of points that maximizes H(x-y) while keeping max(H(x), H(x+y), H(x+2y)) low.
    # This provides discrete, sparse starting points which can be smoothed by the optimizer.
    
    def greedy_start(target_size, seed):
        rng_local = np.random.default_rng(seed)
        selected = set()
        # Pre-compute bin indices for each cell
        # cell (y, x) -> d_idx, m0_idx, m1_idx, m2_idx
        cell_bins = []
        for y in range(n):
            for x in range(n):
                idx = y * n + x
                d_v = x - y
                m0_v = x
                m1_v = x + y
                m2_v = x + 2 * y
                # Map to bin index
                d_idx = np.searchsorted(d_vals, d_v)
                m0_idx = np.searchsorted(m0_vals, m0_v)
                m1_idx = np.searchsorted(m1_vals, m1_v)
                m2_idx = np.searchsorted(m2_vals, m2_v)
                cell_bins.append((idx, d_idx, m0_idx, m1_idx, m2_idx))
                
        # State: counts in each bin
        d_counts = np.zeros(num_d)
        m0_counts = np.zeros(num_m0)
        m1_counts = np.zeros(num_m1)
        m2_counts = np.zeros(num_m2)
        
        # We want to maximize D and minimize M.
        # Let's score a point by: gain in H_d - lambda * gain in M
        # Since H is concave, adding mass to a bin with low count gives higher marginal gain.
        # We want high marginal gain for d, low marginal gain for m0, m1, m2.
        
        for _ in range(target_size):
            best_score = -np.inf
            best_cell = None
            for idx, d_idx, m0_idx, m1_idx, m2_idx in cell_bins:
                if idx in selected:
                    continue
                # Current entropies
                # To estimate marginal gain, we can look at current counts.
                # Approximation: Marginal gain in H is roughly proportional to -log(p) or similar.
                # Let's just use the current counts to estimate "desirability".
                # High count in d_bin -> low marginal gain for D.
                # High count in m_bin -> high marginal gain for M (bad).
                
                # Simple heuristic: 
                # Score = -d_counts[d_idx] + 0.1 * (m0_counts[m0_idx] + m1_counts[m1_idx] + m2_counts[m2_idx])
                # We want to pick cells where d_count is low (to spread D) and m_counts are high (to concentrate M? No, we want M low, so we want to pick cells that DON'T increase M much. 
                # Wait. If we pick a cell that is already in a crowded m_bin, the increase in H_m is small. 
                # If we pick a cell in a new m_bin, the increase in H_m is large. 
                # So we prefer cells where m_counts are ALREADY HIGH (so adding 1 doesn't increase entropy much) 
                # AND d_counts are LOW (so adding 1 increases D entropy a lot).
                
                score = -d_counts[d_idx] + 0.5 * (m0_counts[m0_idx] + m1_counts[m1_idx] + m2_counts[m2_idx])
                
                if score > best_score:
                    best_score = score
                    best_cell = (idx, d_idx, m0_idx, m1_idx, m2_idx)
            
            if best_cell is None:
                break
                
            idx, d_idx, m0_idx, m1_idx, m2_idx = best_cell
            selected.add(idx)
            d_counts[d_idx] += 1
            m0_counts[m0_idx] += 1
            m1_counts[m1_idx] += 1
            m2_counts[m2_idx] += 1
            
        z = np.full(100, -100.0)
        for idx in selected:
            z[idx] = 0.0
        return z

    for size in [5, 10, 15, 20]:
        for seed in range(5):
            starts.append(greedy_start(size, 1000 + seed))

    # --- Evaluate all starts and find best ---
    best_score = -1e9
    z_best = starts[0].copy()
    best_p = log_softmax(z_best)

    for z_flat in starts:
        p_flat = log_softmax(z_flat)
        score = get_score(p_flat)
        if score > best_score:
            best_score = score
            z_best = z_flat.copy()
            best_p = p_flat.copy()

    # --- Multi-phase optimization ---
    
    # Phase 1: Smooth Nelder-Mead from top diverse starts
    # We pick the best start and a few others with different structures
    candidates = [z_best.copy()]
    # Add some diverse starts: first 20
    for i in range(min(20, len(starts))):
        if i != 0: # skip duplicate of best if it was first
             candidates.append(starts[i].copy())
    
    # Also add random perturbations of the best
    for _ in range(5):
        z_pert = z_best + rng.normal(0, 0.5, 100)
        candidates.append(z_pert)

    temperatures = [0.5, 0.2, 0.1, 0.05]
    
    for temp in temperatures:
        if time.time() > deadline - 120.0:
            break
        for z_init in candidates:
            if time.time() > deadline - 120.0:
                break
            try:
                res = minimize(
                    lambda z: objective_smooth(z, temperature=temp),
                    z_init,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'fatol': 1e-10, 'xatol': 1e-12}
                )
                p_res = log_softmax(res.x)
                score = get_score(p_res)
                if score > best_score:
                    best_score = score
                    z_best = res.x.copy()
                    best_p = p_res.copy()
            except Exception:
                pass

    # Phase 2: Exact SLSQP on probability space
    # Convert best z to p
    p_current = log_softmax(z_best)
    
    bounds_p = [(0.0, 1.0) for _ in range(100)]
    constraints_p = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    
    # Run SLSQP from current best and a few other good candidates
    slsqp_starts = [p_current.copy()]
    # Add top 5 starts from initial evaluation? We didn't store scores, so just use a few random good ones
    # Let's just use the current best and a couple of random restarts from the pool
    for _ in range(3):
        idx = rng.integers(0, len(starts))
        slsqp_starts.append(log_softmax(starts[idx]))

    for p_init in slsqp_starts:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(
                lambda p: -get_score(p),
                p_init,
                method='SLSQP',
                bounds=bounds_p,
                constraints=constraints_p,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            p_res = np.clip(res.x, 0, None)
            s = np.sum(p_res)
            if s > 1e-9:
                p_res /= s
            score = get_score(p_res)
            if score > best_score:
                best_score = score
                best_p = p_res.copy()
                z_best = np.log(p_res + 1e-15)
        except Exception:
            pass

    # Phase 3: Final Nelder-Mead polish in log-space with exact objective
    if time.time() < deadline - 15.0:
        try:
            res_nm = minimize(
                lambda z: objective_exact(z),
                z_best,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-15}
            )
            p_nm = log_softmax(res_nm.x)
            score_nm = get_score(p_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_p = p_nm.copy()
        except Exception:
            pass

    # Final validation and normalization
    P = best_p.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)

    return P
