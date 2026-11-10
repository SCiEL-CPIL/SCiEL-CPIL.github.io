import numpy as np
import time

def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 weight matrix P."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_values(x_flat - y_flat, w)
    H0 = entropy_from_values(x_flat, w)
    H1 = entropy_from_values(x_flat + y_flat, w)
    H2 = entropy_from_values(x_flat + 2 * y_flat, w)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _compute_ratio_fast(w, diff_bins, h0_bins, h1_bins, h2_bins, n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2):
    """Fast ratio computation using precomputed bin indices."""
    def entropy_from_bins(bins, weights, n_bins):
        counts = np.zeros(n_bins)
        np.add.at(counts, bins, weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_bins(diff_bins, w, n_bins_diff)
    H0 = entropy_from_bins(h0_bins, w, n_bins_h0)
    H1 = entropy_from_bins(h1_bins, w, n_bins_h1)
    H2 = entropy_from_bins(h2_bins, w, n_bins_h2)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _soft_max_entropy_ratio(w, diff_bins, h0_bins, h1_bins, h2_bins, n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2, beta=10.0):
    """Compute D/M with soft-max for the M term, suitable for gradient-based optimization."""
    def entropy_from_bins(bins, weights, n_bins):
        counts = np.zeros(n_bins)
        np.add.at(counts, bins, weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_bins(diff_bins, w, n_bins_diff)
    H0 = entropy_from_bins(h0_bins, w, n_bins_h0)
    H1 = entropy_from_bins(h1_bins, w, n_bins_h1)
    H2 = entropy_from_bins(h2_bins, w, n_bins_h2)
    
    # Soft-max: smooth approximation of max(H0, H1, H2)
    H_vals = np.array([H0, H1, H2])
    H_max = np.max(H_vals)
    if H_max <= 1e-9:
        return 0.0
    # Log-sum-exp style soft max
    exp_scaled = np.exp(beta * (H_vals - H_max))
    soft_max = H_max + np.log(np.sum(exp_scaled)) / beta
    
    if soft_max <= 1e-9:
        return 0.0
    return D / soft_max

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    # Precompute bin indices for fast entropy computation
    diff_vals = x_flat - y_flat  # range -9..9, shift by 9 -> 0..18
    diff_bins = diff_vals + 9
    n_bins_diff = 19
    
    h0_vals = x_flat  # range 0..9
    h0_bins = h0_vals
    n_bins_h0 = 10
    
    h1_vals = x_flat + y_flat  # range 0..18
    h1_bins = h1_vals
    n_bins_h1 = 19
    
    h2_vals = x_flat + 2 * y_flat  # range 0..27
    h2_bins = h2_vals
    n_bins_h2 = 28
    
    def compute_ratio(w):
        return _compute_ratio_fast(w, diff_bins, h0_bins, h1_bins, h2_bins,
                                   n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2)
    
    def compute_ratio_soft(w, beta=10.0):
        return _soft_max_entropy_ratio(w, diff_bins, h0_bins, h1_bins, h2_bins,
                                       n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2, beta)
    
    # Generate diverse initial candidates focused on anti-diagonal structure
    candidates = []
    
    # Candidate 1: uniform
    candidates.append(np.ones((n, n), dtype=np.float64) / 100.0)
    
    # Candidate 2: Each anti-diagonal (x+y = k) with equal weight
    for k in range(19):
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                w[i, j] = 1.0
        if w.sum() > 0:
            w /= w.sum()
            candidates.append(w)
    
    # Candidate 3: Concentrate on middle anti-diagonals where x-y varies most
    w3 = np.zeros((n, n), dtype=np.float64)
    for k in [8, 9, 10]:
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                w3[i, j] = 1.0
    w3 /= w3.sum()
    candidates.append(w3)
    
    # Candidate 4: Weight anti-diagonals to maximize x-y entropy
    w4 = np.zeros((n, n), dtype=np.float64)
    for k in range(19):
        diag_cells = []
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                diag_cells.append(j - i)
        if len(diag_cells) > 0:
            unique_diffs = len(set(diag_cells))
            weight = unique_diffs / len(diag_cells)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    w4[i, j] = weight
    w4 /= w4.sum()
    candidates.append(w4)
    
    # Candidate 5: Make H(x) low by concentrating x, while spreading x-y
    for x_target in [4, 5]:
        w5 = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            w5[i, x_target] = 1.0
        w5 /= w5.sum()
        candidates.append(w5)
    
    # Candidate 6: Combine anti-diagonal concentration with x-y spreading
    for alpha in [0.5, 1.0, 2.0]:
        w6 = np.zeros((n, n), dtype=np.float64)
        for k in range(19):
            diag_cells = []
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    diag_cells.append((i, j))
            if len(diag_cells) > 0:
                center_dist = abs(k - 9)
                weight = np.exp(-alpha * center_dist / 10.0)
                for i, j in diag_cells:
                    w6[i, j] = weight
        w6 /= w6.sum()
        candidates.append(w6)
    
    # Candidate 7: Optimal anti-diagonal selection
    best_single_diag = None
    best_single_score = -1
    for k in range(19):
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                w[i, j] = 1.0
        if w.sum() > 0:
            w /= w.sum()
            sc = compute_ratio(w.flatten())
            if sc > best_single_score:
                best_single_score = sc
                best_single_diag = w.copy()
    if best_single_diag is not None:
        candidates.append(best_single_diag)
    
    # Evaluate all candidates
    best_score = -1
    best_P = candidates[0]
    for c in candidates:
        sc = compute_ratio(c.flatten())
        if sc > best_score:
            best_score = sc
            best_P = c.copy()
    
    # Phase 1: Multi-cell local search with restarts
    np.random.seed(42)
    current_P = best_P.copy()
    current_score = best_score
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 200
    phase1_end = start_time + 120
    
    while time.time() < phase1_end:
        if no_improve_count > max_no_improve:
            noise_scale = 0.2 + 0.1 * (no_improve_count // max_no_improve)
            noise = np.random.normal(0, noise_scale, (n, n))
            current_P = best_P + noise
            current_P = np.clip(current_P, 0, None)
            s = current_P.sum()
            if s > 1e-9:
                current_P /= s
            else:
                current_P = best_P.copy()
            current_score = compute_ratio(current_P.flatten())
            no_improve_count = 0
            step_size = 0.15
            max_no_improve = 100
            continue
        
        n_cells = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        indices = np.random.randint(0, n * n, n_cells)
        deltas = np.random.uniform(-step_size, step_size, n_cells)
        
        new_P = current_P.copy()
        new_P_flat = new_P.flatten()
        new_P_flat[indices] += deltas
        new_P_flat = np.clip(new_P_flat, 0, None)
        s = new_P_flat.sum()
        if s <= 1e-9:
            continue
        new_P_flat /= s
        new_P = new_P_flat.reshape(n, n)
        
        new_score = compute_ratio(new_P_flat)
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.97
        else:
            no_improve_count += 1
            step_size = max(0.01, step_size * 1.03)
    
    # Phase 2: Gradient-based optimization with soft-max surrogate
    from scipy.optimize import minimize
    
    P_for_log = best_P.copy()
    P_for_log[P_for_log < 1e-12] = 1e-12
    log_P_init = np.log(P_for_log.flatten())
    
    def neg_soft_ratio_from_log(log_p_flat, beta=10.0):
        log_p = log_p_flat.reshape(n, n)
        log_p_max = np.max(log_p)
        p = np.exp(log_p - log_p_max)
        p_sum = p.sum()
        if p_sum <= 1e-15:
            return 1e10
        p = p / p_sum
        w = p.flatten()
        ratio = compute_ratio_soft(w, beta)
        return -ratio
    
    phase2_end = start_time + timeout - safety_margin
    gradient_starts = [log_P_init]
    
    # Add perturbed starts
    for _ in range(8):
        perturbed = log_P_init + np.random.normal(0, 0.8, n * n)
        gradient_starts.append(perturbed)
    
    # Add starts from other good candidates
    for c in candidates[:6]:
        c_log = c.copy()
        c_log[c_log < 1e-12] = 1e-12
        gradient_starts.append(np.log(c_log.flatten()))
    
    best_gradient_score = best_score
    best_gradient_P = best_P.copy()
    
    # Use multiple beta values for soft-max
    beta_values = [5.0, 10.0, 20.0]
    
    for beta in beta_values:
        if time.time() > phase2_end - 30:
            break
        
        for log_start in gradient_starts:
            if time.time() > phase2_end - 60:
                break
            
            try:
                result = minimize(
                    neg_soft_ratio_from_log,
                    log_start,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                log_p_opt = result.x.reshape(n, n)
                log_p_max = np.max(log_p_opt)
                p_opt = np.exp(log_p_opt - log_p_max)
                p_opt = p_opt / p_opt.sum()
                
                # Evaluate with hard max for true score
                score_opt = compute_ratio(p_opt.flatten())
                if score_opt > best_gradient_score:
                    best_gradient_score = score_opt
                    best_gradient_P = p_opt.copy()
            except Exception:
                continue
    
    if best_gradient_score > best_score:
        best_score = best_gradient_score
        best_P = best_gradient_P.copy()
    
    # Phase 3: Final high-precision polish with adaptive steps
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.003
    no_improve_count = 0
    max_no_improve = 150
    
    # Precompute anti-diagonal cell lists for structured moves
    anti_diag_cells = {}
    for k in range(19):
        cells = [(a, b) for a in range(n) for b in range(n) if a + b == k]
        if len(cells) >= 2:
            anti_diag_cells[k] = cells
    
    while time.time() < phase2_end - 10:
        if no_improve_count > max_no_improve:
            break
        
        move_type = np.random.choice(['single', 'pair', 'diag', 'row_col'], p=[0.3, 0.25, 0.25, 0.2])
        
        if move_type == 'single':
            i, j = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            new_P = current_P.copy()
            new_P[i, j] += delta
        elif move_type == 'pair':
            i1, j1 = np.random.randint(0, n, 2)
            i2, j2 = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            new_P = current_P.copy()
            new_P[i1, j1] += delta
            new_P[i2, j2] -= delta
        elif move_type == 'diag':
            k = np.random.choice(list(anti_diag_cells.keys()))
            cells = anti_diag_cells[k]
            i1, j1 = cells[np.random.randint(len(cells))]
            i2, j2 = cells[np.random.randint(len(cells))]
            if i1 == i2 and j1 == j2:
                continue
            delta = np.random.uniform(-step_size, step_size)
            new_P = current_P.copy()
            new_P[i1, j1] += delta
            new_P[i2, j2] -= delta
        else:  # row_col: move along row or column
            if np.random.rand() < 0.5:
                # Move along row
                i = np.random.randint(n)
                j1, j2 = np.random.choice(n, 2, replace=False)
                delta = np.random.uniform(-step_size, step_size)
                new_P = current_P.copy()
                new_P[i, j1] += delta
                new_P[i, j2] -= delta
            else:
                # Move along column
                j = np.random.randint(n)
                i1, i2 = np.random.choice(n, 2, replace=False)
                delta = np.random.uniform(-step_size, step_size)
                new_P = current_P.copy()
                new_P[i1, j] += delta
                new_P[i2, j] -= delta
        
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        
        new_score = compute_ratio(new_P.flatten())
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.94
        else:
            no_improve_count += 1
            step_size = max(0.0003, step_size * 1.04)
    
    # Phase 4: Ultra-fine polish
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.0005
    no_improve_count = 0
    max_no_improve = 200
    phase4_end = phase2_end - 5
    
    while time.time() < phase4_end:
        if no_improve_count > max_no_improve:
            break
        
        i, j = np.random.randint(0, n, 2)
        delta = np.random.uniform(-step_size, step_size)
        new_P = current_P.copy()
        new_P[i, j] += delta
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        
        new_score = compute_ratio(new_P.flatten())
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.96
        else:
            no_improve_count += 1
            step_size = max(0.00005, step_size * 1.05)
    
    # Final validation and normalization
    result = np.clip(best_P, 0, None)
    s = result.sum()
    if s <= 1e-9:
        result = np.ones((n, n)) / 100.0
    else:
        result = result / s
    
    return result
