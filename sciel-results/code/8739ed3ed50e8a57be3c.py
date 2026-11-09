<<<<<<< SEARCH
def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 310.0  # Leave margin before 330s limit
    
    # Pre-compute slack matrix
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    def _eval_heuristic(base):
        """Evaluate a heuristic matrix with 3 seeds, return mean score."""
        scores = []
        for seed in [42, 123, 456]:
            score = _run_aco(d, capacity, base, n_ants=20, n_iters=30, seed=seed)
            scores.append(score)
        return np.mean(scores)
    
    def _try_heuristic(base):
        """Try a heuristic, update best if better."""
        nonlocal best_score, best_heuristic
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        avg_score = _eval_heuristic(base)
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = base.copy()
        return avg_score
    
    # Phase 1: Coarse search over best-fit alpha values
    alphas = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        
        if alpha == 0:
            base = np.where(valid, 1.0, 1e-9)
        else:
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        
        _try_heuristic(base)
    
    # Phase 2: Worst-fit style
    betas = [0.5, 1, 1.5, 2, 3]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, (slack + 1e-9)**beta, 1e-9)
        _try_heuristic(base)
    
    # Phase 3: Target-distance heuristics (prefer pairs summing near a target)
    targets = [capacity * 0.5, capacity * 0.55, capacity * 0.6, capacity * 0.65, capacity * 0.7, capacity * 0.8]
    
    for target in targets:
        if time.time() - start_time > time_limit:
            break
        
        dist = np.abs(pair_sum - target)
        base = np.where(valid, 1.0 / (dist + 1.0), 1e-9)
        _try_heuristic(base)
    
    # Phase 4: Complementarity heuristic - reward pairs that fill bins well
    # Use exponential of negative slack (strongly prefer tight fits)
    for gamma in [0.05, 0.1, 0.15, 0.2, 0.3]:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, np.exp(-gamma * slack), 1e-9)
        _try_heuristic(base)
    
    # Phase 5: Combined - product of best-fit and complementarity
    if time.time() - start_time < time_limit:
        for alpha in [1, 2, 3]:
            for gamma in [0.1, 0.15, 0.2]:
                if time.time() - start_time > time_limit:
                    break
                bf = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
                comp = np.where(valid, np.exp(-gamma * slack), 1e-9)
                base = bf * comp
                _try_heuristic(base)
    
    # Phase 6: Fine-grained search around best alpha if we have time
    if time.time() - start_time < time_limit * 0.7:
        # Fine grid for best-fit
        fine_alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
        for alpha in fine_alphas:
            if time.time() - start_time > time_limit:
                break
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            _try_heuristic(base)
    
    # Phase 7: Size-based heuristic - prefer placing larger items with smaller items
    if time.time() - start_time < time_limit:
        # Heuristic: 1 / (d[i] * d[j]) for valid pairs - favors complementary sizes
        size_prod = d[:, None] * d[None, :]
        base = np.where(valid, 1.0 / (size_prod + 1e-9), 1e-9)
        _try_heuristic(base)
        
        # Heuristic: (capacity - d[i] - d[j])^(-1) * d[i] * d[j]
        # This rewards pairs that are both large and fit well
        base2 = np.where(valid, (d[:, None] * d[None, :]) / (slack + 1.0), 1e-9)
        _try_heuristic(base2)
    
    if best_heuristic is None:
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
=======
def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 310.0  # Leave margin before 330s limit
    
    # Pre-compute slack matrix
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    def _eval_heuristic(base):
        """Evaluate a heuristic matrix with 3 seeds, return mean score."""
        scores = []
        for seed in [42, 123, 456]:
            score = _run_aco_fast(d, capacity, base, n_ants=20, n_iters=30, seed=seed)
            scores.append(score)
        return np.mean(scores)
    
    def _try_heuristic(base):
        """Try a heuristic, update best if better."""
        nonlocal best_score, best_heuristic
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        avg_score = _eval_heuristic(base)
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = base.copy()
        return avg_score
    
    def _ffd_bin_assignment():
        """
        Compute FFD bin assignment and return a dict mapping item index -> bin index.
        """
        sorted_indices = np.argsort(-d)  # descending order
        bins = []  # list of (fill, [item_indices])
        
        item_to_bin = {}
        for idx in sorted_indices:
            placed = False
            for b_idx, (fill, items) in enumerate(bins):
                if fill + d[idx] <= capacity:
                    bins[b_idx] = (fill + d[idx], items + [int(idx)])
                    item_to_bin[int(idx)] = b_idx
                    placed = True
                    break
            if not placed:
                item_to_bin[int(idx)] = len(bins)
                bins.append((d[idx], [int(idx)]))
        
        return item_to_bin
    
    def _ffd_compatibility_matrix(strength=1.0, base_level=1e-9):
        """
        Build a heuristic matrix where pairs in the same FFD bin get high attractiveness.
        Adjacent items in the same bin get the highest values.
        """
        item_to_bin = _ffd_bin_assignment()
        mat = np.full((n, n), base_level, dtype=float)
        
        # Group items by bin
        bin_items = {}
        for item, b in item_to_bin.items():
            if b not in bin_items:
                bin_items[b] = []
            bin_items[b].append(item)
        
        for b, items in bin_items.items():
            items_sorted = sorted(items)  # by original index for determinism
            m = len(items_sorted)
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    # Distance in the bin sequence
                    dist = abs(i - j)
                    # Higher attractiveness for closer items in the same bin
                    if dist == 1:
                        mat[items_sorted[i], items_sorted[j]] = strength
                    elif dist == 2:
                        mat[items_sorted[i], items_sorted[j]] = strength * 0.5
                    else:
                        mat[items_sorted[i], items_sorted[j]] = strength * 0.1
        
        return mat
    
    # Phase 0: FFD-derived sparse compatibility heuristics
    for strength in [1.0, 5.0, 10.0, 50.0, 100.0]:
        if time.time() - start_time > time_limit:
            break
        base = _ffd_compatibility_matrix(strength=strength, base_level=1e-9)
        _try_heuristic(base)
    
    # FFD combined with best-fit
    if time.time() - start_time < time_limit:
        for strength in [1.0, 10.0]:
            for alpha in [0.5, 1.0, 2.0]:
                if time.time() - start_time > time_limit:
                    break
                ffd_mat = _ffd_compatibility_matrix(strength=strength, base_level=1e-9)
                bf = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
                # Combine: use FFD structure but modulate by best-fit within same bin
                base = ffd_mat * bf
                base = np.where(valid, base, 1e-9)
                _try_heuristic(base)
    
    # Phase 1: Coarse search over best-fit alpha values
    alphas = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        
        if alpha == 0:
            base = np.where(valid, 1.0, 1e-9)
        else:
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        
        _try_heuristic(base)
    
    # Phase 2: Worst-fit style
    betas = [0.5, 1, 1.5, 2, 3]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, (slack + 1e-9)**beta, 1e-9)
        _try_heuristic(base)
    
    # Phase 3: Target-distance heuristics (prefer pairs summing near a target)
    targets = [capacity * 0.5, capacity * 0.55, capacity * 0.6, capacity * 0.65, capacity * 0.7, capacity * 0.8]
    
    for target in targets:
        if time.time() - start_time > time_limit:
            break
        
        dist = np.abs(pair_sum - target)
        base = np.where(valid, 1.0 / (dist + 1.0), 1e-9)
        _try_heuristic(base)
    
    # Phase 4: Complementarity heuristic - reward pairs that fill bins well
    for gamma in [0.05, 0.1, 0.15, 0.2, 0.3]:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, np.exp(-gamma * slack), 1e-9)
        _try_heuristic(base)
    
    # Phase 5: Combined - product of best-fit and complementarity
    if time.time() - start_time < time_limit:
        for alpha in [1, 2, 3]:
            for gamma in [0.1, 0.15, 0.2]:
                if time.time() - start_time > time_limit:
                    break
                bf = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
                comp = np.where(valid, np.exp(-gamma * slack), 1e-9)
                base = bf * comp
                _try_heuristic(base)
    
    # Phase 6: Fine-grained search around best alpha if we have time
    if time.time() - start_time < time_limit * 0.7:
        fine_alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
        for alpha in fine_alphas:
            if time.time() - start_time > time_limit:
                break
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            _try_heuristic(base)
    
    # Phase 7: Size-based heuristic
    if time.time() - start_time < time_limit:
        size_prod = d[:, None] * d[None, :]
        base = np.where(valid, 1.0 / (size_prod + 1e-9), 1e-9)
        _try_heuristic(base)
        
        base2 = np.where(valid, (d[:, None] * d[None, :]) / (slack + 1.0), 1e-9)
        _try_heuristic(base2)
    
    # Phase 8: FFD with different base levels and strengths
    if time.time() - start_time < time_limit:
        for strength in [2.0, 5.0, 20.0, 50.0]:
            for base_level in [1e-9, 0.01, 0.1]:
                if time.time() - start_time > time_limit:
                    break
                base = _ffd_compatibility_matrix(strength=strength, base_level=base_level)
                _try_heuristic(base)
    
    if best_heuristic is None:
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
>>>>>>> REPLACE

