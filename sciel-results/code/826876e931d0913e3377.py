import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 115.0

def _get_divisors(n):
    divs = set()
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.add(n)
    return sorted(divs)

_DIVS = _get_divisors(_BASE)

def construct():
    start_time = time.time()
    time_limit = _DEADLINE
    
    N = _BASE
    divisors = _DIVS
    n_divs = len(divisors)
    
    # Precompute indices for each modulus and residue
    indices_cache = {}
    for m in divisors:
        indices_cache[m] = [np.arange(r, N, m) for r in range(m)]
    
    def compute_uncovered(residues):
        """Compute uncovered count from a residues dict."""
        counts = np.zeros(N, dtype=np.int8)
        for m in divisors:
            counts[indices_cache[m][residues[m]]] += 1
        return int(np.sum(counts == 0))
    
    def greedy_build(order):
        """Greedy: process moduli in given order, pick best residue."""
        covered = np.zeros(N, dtype=bool)
        residues = {}
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                idx = indices_cache[m][r]
                gain = int((~covered[idx]).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[m] = best_r
            covered[indices_cache[m][best_r]] = True
        return residues
    
    # 1. Initialization: Try diverse greedy orders
    rng = random.Random(12345)
    
    # Multiple structured orders
    orders = [
        divisors.copy(),                          # ascending
        sorted(divisors, reverse=True),           # descending
    ]
    
    # Add some structured random orders
    for _ in range(8):
        order = divisors.copy()
        rng.shuffle(order)
        orders.append(order)
    
    # Sort by a heuristic: larger moduli first tend to cover more
    orders.append(sorted(divisors, key=lambda m: -N // m))
    orders.append(sorted(divisors, key=lambda m: m % 3))  # group by residue class
    
    best_res = None
    best_unc = N + 1
    
    for order in orders:
        res = greedy_build(order)
        unc = compute_uncovered(res)
        if unc < best_unc:
            best_unc = unc
            best_res = res
    
    best_residues = best_res.copy()
    best_uncovered = best_unc
    
    # 2. Build incremental state
    counts = np.zeros(N, dtype=np.int8)
    for m in divisors:
        counts[indices_cache[m][best_residues[m]]] += 1
    
    # Precompute cnt1 and cnt0 for all (m, r)
    cnt1 = {}
    cnt0 = {}
    for m in divisors:
        cnt1[m] = np.zeros(m, dtype=np.int32)
        cnt0[m] = np.zeros(m, dtype=np.int32)
        for r in range(m):
            idx = indices_cache[m][r]
            cnt1[m][r] = np.sum(counts[idx] == 1)
            cnt0[m][r] = np.sum(counts[idx] == 0)
    
    current_uncovered = best_uncovered
    
    def recompute_cnt_for_m(m):
        """Recompute cnt1[m] and cnt0[m] after counts change."""
        for r in range(m):
            idx = indices_cache[m][r]
            cnt1[m][r] = np.sum(counts[idx] == 1)
            cnt0[m][r] = np.sum(counts[idx] == 0)
    
    def apply_single_move(m, r_old, r_new):
        """Apply a single residue change, return delta."""
        nonlocal counts, current_uncovered
        idx_old = indices_cache[m][r_old]
        idx_new = indices_cache[m][r_new]
        
        L = int(cnt1[m][r_old])
        G = int(cnt0[m][r_new])
        delta = L - G
        
        counts[idx_old] -= 1
        counts[idx_new] += 1
        
        # Recompute cnt for this modulus
        recompute_cnt_for_m(m)
        
        current_uncovered += delta
        best_residues[m] = r_new
        
        return delta
    
    def try_best_single_move(m):
        """Try to find best single move for modulus m. Returns (r_new, delta) or (None, 0)."""
        r_old = best_residues[m]
        L = int(cnt1[m][r_old])
        
        # Find r_new that minimizes delta = L - cnt0[m][r_new]
        # i.e., maximize cnt0[m][r_new]
        max_G = np.max(cnt0[m])
        if max_G <= L:
            return None, 0
        
        candidates = np.where(cnt0[m] == max_G)[0]
        candidates = candidates[candidates != r_old]
        if len(candidates) == 0:
            return None, 0
        
        # Pick randomly among best candidates
        r_new = int(rng.choice(candidates))
        delta = L - int(max_G)
        return r_new, delta
    
    # 3. Local search with multi-move escapes
    no_improve_steps = 0
    max_no_improve = 50
    step_count = 0
    
    # Weighted selection: larger moduli have more potential impact
    # Weight proportional to N/m (number of positions covered)
    weights = np.array([N // m for m in divisors], dtype=np.float64)
    weights /= weights.sum()
    
    while time.time() - start_time < time_limit - 10:
        step_count += 1
        
        # Select modulus with probability proportional to its weight
        m = divisors[rng.choice(n_divs, p=weights)]
        
        r_new, delta = try_best_single_move(m)
        
        if r_new is not None and delta < 0:
            apply_single_move(m, best_residues[m], r_new)
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        # Escape local minimum with multi-move perturbation
        if no_improve_steps > max_no_improve:
            # Try a 2-move swap: change residues of two moduli simultaneously
            if n_divs >= 2:
                m1_idx = rng.randrange(n_divs)
                m2_idx = rng.randrange(n_divs)
                if m1_idx != m2_idx:
                    m1 = divisors[m1_idx]
                    m2 = divisors[m2_idx]
                    
                    r1_old = best_residues[m1]
                    r2_old = best_residues[m2]
                    
                    # Try random new residues for both
                    r1_new = rng.randint(0, m1 - 1)
                    r2_new = rng.randint(0, m2 - 1)
                    
                    if r1_new != r1_old or r2_new != r2_old:
                        # Compute combined delta
                        idx1_old = indices_cache[m1][r1_old]
                        idx1_new = indices_cache[m1][r1_new]
                        idx2_old = indices_cache[m2][r2_old]
                        idx2_new = indices_cache[m2][r2_new]
                        
                        # Temporarily apply both changes
                        old_counts = counts.copy()
                        
                        counts[idx1_old] -= 1
                        counts[idx1_new] += 1
                        counts[idx2_old] -= 1
                        counts[idx2_new] += 1
                        
                        new_unc = int(np.sum(counts == 0))
                        delta_combined = new_unc - current_uncovered
                        
                        if delta_combined < 0:
                            # Accept the move
                            best_residues[m1] = r1_new
                            best_residues[m2] = r2_new
                            current_uncovered = new_unc
                            
                            # Recompute cnt for both moduli
                            recompute_cnt_for_m(m1)
                            recompute_cnt_for_m(m2)
                            
                            if current_uncovered < best_uncovered:
                                best_uncovered = current_uncovered
                            no_improve_steps = 0
                        else:
                            # Revert
                            counts = old_counts
                            no_improve_steps = 0
            else:
                no_improve_steps = 0
    
    # 4. Final polish: exhaustive single-move improvement until convergence
    polish_deadline = start_time + 118.0
    
    # Recompute state
    counts = np.zeros(N, dtype=np.int8)
    for m in divisors:
        counts[indices_cache[m][best_residues[m]]] += 1
    
    cnt1 = {}
    cnt0 = {}
    for m in divisors:
        cnt1[m] = np.zeros(m, dtype=np.int32)
        cnt0[m] = np.zeros(m, dtype=np.int32)
        for r in range(m):
            idx = indices_cache[m][r]
            cnt1[m][r] = np.sum(counts[idx] == 1)
            cnt0[m][r] = np.sum(counts[idx] == 0)
    
    current_uncovered = compute_uncovered(best_residues)
    
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for m in divisors:
            if time.time() >= polish_deadline:
                break
            r_old = best_residues[m]
            L = int(cnt1[m][r_old])
            max_G = np.max(cnt0[m])
            if max_G > L:
                candidates = np.where(cnt0[m] == max_G)[0]
                candidates = candidates[candidates != r_old]
                if len(candidates) > 0:
                    best_r = int(candidates[0])
                    best_delta = L - int(max_G)
                    
                    idx_old = indices_cache[m][r_old]
                    idx_new = indices_cache[m][best_r]
                    
                    counts[idx_old] -= 1
                    counts[idx_new] += 1
                    recompute_cnt_for_m(m)
                    
                    current_uncovered += best_delta
                    best_residues[m] = best_r
                    improved = True
    
    # Final check
    final_unc = compute_uncovered(best_residues)
    if final_unc < best_uncovered:
        best_uncovered = final_unc
    
    result = [[m, best_residues[m]] for m in divisors]
    return result
