import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129

    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using numpy convolution-like approach"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=int)
        if k <= 50:
            A_i = A_arr[:, None]
            A_j = A_arr[None, :]
            sums = A_i + A_j
            mask = np.triu(np.ones((k, k), dtype=bool), k=0)
            valid_sums = sums[mask]
            valid_sums = valid_sums[valid_sums <= max_sum]
            if len(valid_sums) > 0:
                counts = np.bincount(valid_sums, minlength=max_sum + 1)
            else:
                counts = np.zeros(max_sum + 1, dtype=int)
            return counts
        else:
            counts = np.zeros(max_sum + 1, dtype=int)
            for i in range(k):
                ai = A_arr[i]
                s = ai + ai
                if s <= max_sum:
                    counts[s] += 1
                for j in range(i + 1, k):
                    aj = A_arr[j]
                    s = ai + aj
                    if s <= max_sum:
                        counts[s] += 1
            return counts

    def get_cost_congestion(A, t):
        """
        Returns (max_mult, uncovered_count, congestion_penalty, counts_array)
        Uses a smooth congestion penalty to encourage uniform distribution.
        """
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, 0, counts)
        max_mult = int(np.max(target_counts))
        excess = np.maximum(target_counts - t, 0)
        congestion = int(np.sum(excess ** 2))
        return (max_mult, 0, congestion, counts)

    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        max_mult, uncovered, _, counts = get_cost_congestion(A, 0)
        return (max_mult, uncovered, counts)

    def incremental_swap_cost(A, x_in, x_out, current_counts, t):
        """
        Compute new counts after removing x_in and adding x_out.
        Returns (new_counts, new_cost, new_uncovered, new_congestion)
        """
        new_counts = current_counts.copy()

        # Subtract contributions involving x_in
        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1

        # Add contributions involving x_out
        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
            new_congestion = 0
        else:
            new_cost = int(np.max(target_counts))
            excess = np.maximum(target_counts - t, 0)
            new_congestion = int(np.sum(excess ** 2))

        return new_counts, new_cost, uncovered, new_congestion

    def incremental_add_cost(A, x_add, current_counts, t):
        """
        Compute new counts after adding x_add to A.
        Returns (new_counts, new_cost, new_uncovered, new_congestion)
        """
        new_counts = current_counts.copy()

        s = 2 * x_add
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            s = x_add + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
            new_congestion = 0
        else:
            new_cost = int(np.max(target_counts))
            excess = np.maximum(target_counts - t, 0)
            new_congestion = int(np.sum(excess ** 2))

        return new_counts, new_cost, uncovered, new_congestion

    def incremental_remove_cost(A, x_rem, current_counts, t):
        """
        Compute new counts after removing x_rem from A.
        Returns (new_counts, new_cost, new_uncovered, new_congestion)
        """
        new_counts = current_counts.copy()

        s = 2 * x_rem
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_rem:
                continue
            s = x_rem + y
            if s <= max_sum:
                new_counts[s] -= 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
            new_congestion = 0
        else:
            new_cost = int(np.max(target_counts))
            excess = np.maximum(target_counts - t, 0)
            new_congestion = int(np.sum(excess ** 2))

        return new_counts, new_cost, uncovered, new_congestion

    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')

    candidates = []

    # Candidate 1: k=16-20, spread out evenly
    for k in [16, 17, 18, 19, 20]:
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) < k:
            base_set = set()
            used = set()
            for i in range(k):
                x = round(i * 128 / (k - 1))
                while x in used:
                    x += 1
                    if x > 128:
                        x = 0
                base_set.add(x)
                used.add(x)
            base_set = set(x for x in base_set if 0 <= x <= 128)
        candidates.append(set(base_set))

        for offset in range(-3, 4):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128 / (k - 1))) + offset
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 2:
                candidates.append(perturbed)

    # Candidate: Two clusters approach
    for a in range(5, 15):
        for b in range(113, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)

    # Candidate: Mix of small and large
    for m in range(6, 12):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid = set(range(60 - m // 2, 60 + m // 2 + 1))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)

    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost, uncovered, _ = get_cost_detailed(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)

    # --- Differential Evolution phase: global search to escape local minima ---
    # Model A as binary vector of length 129. DE explores the discrete landscape.
    # We use a smooth epigraph surrogate: minimize max(counts) via soft-max.
    
    def de_objective(binary_vec):
        """
        Objective for DE: takes a binary vector (0/1) of length 129,
        returns a smooth approximation of the peak multiplicity.
        Uses log-sum-exp for soft-max of the target counts.
        """
        if time.time() > deadline - 10.0:
            return 9999.0
        
        A = set(i for i, v in enumerate(binary_vec) if v > 0.5)
        if len(A) < 10 or len(A) > 40:
            return 9999.0
        
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1].astype(float)
        
        # Check coverage
        if np.any(target_counts == 0):
            uncovered = np.sum(target_counts == 0)
            return 1000.0 + uncovered
        
        # Soft-max: log(sum(exp(c_i / beta))) * beta approximates max(c_i)
        # For beta small, this is close to max; for beta large, it's smoother
        beta = 2.0
        max_tc = np.max(target_counts)
        if max_tc == 0:
            return 0.0
        # Shift for numerical stability
        shifted = target_counts / beta - max_tc / beta
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        soft_max = (max_tc + beta * log_sum_exp)
        
        # Add congestion penalty for uniformity
        t = max(2.0, max_tc - 1.0)
        excess = np.maximum(target_counts - t, 0)
        congestion = 0.1 * np.sum(excess ** 2)
        
        return soft_max + congestion

    # Run DE with multiple restarts to explore different regions
    if best_A is not None:
        # Use the best structured candidate as a starting point for DE
        # DE works in continuous space [0,1]^129, then we threshold at 0.5
        
        # Try DE with different population sizes and tolerances
        de_configs = [
            {'popsize': 15, 'maxiter': 80, 'tol': 0.01, 'mutation': (0.5, 1.0), 'recombination': 0.7},
            {'popsize': 12, 'maxiter': 60, 'tol': 0.05, 'mutation': (0.3, 0.8), 'recombination': 0.8},
        ]
        
        for cfg in de_configs:
            if time.time() > deadline - 15.0:
                break
            
            try:
                # Bounds: each coordinate in [0, 1]
                bounds = [(0, 1)] * 129
                
                # Use the current best as a hint via initial population
                # Create initial population centered around good solutions
                best_vec = np.zeros(129)
                for x in best_A:
                    best_vec[x] = 1.0
                
                result = differential_evolution(
                    de_objective,
                    bounds,
                    maxiter=cfg['maxiter'],
                    popsize=cfg['popsize'],
                    tol=cfg['tol'],
                    mutation=cfg['mutation'],
                    recombination=cfg['recombination'],
                    seed=42,
                    init='latinhypercube',
                    polish=False,
                    update_interval=10,
                    callback=None
                )
                
                if result.success or result.fun < 1000:
                    de_solution = set(i for i, v in enumerate(result.x) if v > 0.5)
                    if len(de_solution) >= 10 and len(de_solution) <= 40:
                        de_cost, de_uncovered, _ = get_cost_detailed(de_solution)
                        if de_uncovered == 0 and de_cost < best_cost:
                            best_cost = de_cost
                            best_A = set(de_solution)
            except Exception:
                pass

    # Now optimize the best candidate with local search using incremental updates
    if best_A is not None:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)
        
        # Start with a moderate congestion threshold
        t = max(2, current_cost - 1)
        
        # Local search: try swaps and additions/removals with incremental cost
        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)
        
        # Track best and use simulated annealing-like acceptance
        best_local_cost = current_cost
        best_local_A = set(A)
        
        # Multi-phase local search with decreasing "temperature"
        for phase in range(3):
            if time.time() > deadline - 5.0:
                break
            
            if phase == 0:
                max_iter = 20000
                accept_equal = True
            elif phase == 1:
                max_iter = 15000
                accept_equal = True
            else:
                max_iter = 10000
                accept_equal = False
            
            for iteration in range(max_iter):
                if time.time() > deadline - 5.0:
                    break

                move_type = rng.random()

                if move_type < 0.6 and len(outside) > 0:
                    # Swap
                    x_in = rng.choice(A_list)
                    x_out = rng.choice(outside)

                    new_counts, new_cost, _, new_congestion = incremental_swap_cost(A, x_in, x_out, current_counts, t)

                    # Accept if cost improves, or if congestion improves significantly
                    if new_cost < current_cost or (new_cost == current_cost and (new_congestion < 50 or (accept_equal and new_congestion < 100))):
                        A.remove(x_in)
                        A.add(x_out)
                        outside.remove(x_out)
                        outside.append(x_in)
                        A_list.remove(x_in)
                        A_list.append(x_out)
                        current_counts = new_counts
                        current_cost = new_cost
                        if current_cost < best_local_cost:
                            best_local_cost = current_cost
                            best_local_A = set(A)
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A)
                                t = max(2, best_cost - 1)

                elif move_type < 0.8 and len(A) > 14:
                    # Removal
                    x = rng.choice(A_list)

                    new_counts, new_cost, _, new_congestion = incremental_remove_cost(A, x, current_counts, t)

                    if new_cost < current_cost or (new_cost == current_cost and (new_congestion < 30 or (accept_equal and new_congestion < 60))):
                        A.remove(x)
                        outside.append(x)
                        A_list.remove(x)
                        current_counts = new_counts
                        current_cost = new_cost
                        if current_cost < best_local_cost:
                            best_local_cost = current_cost
                            best_local_A = set(A)
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A)
                                t = max(2, best_cost - 1)

                elif move_type < 1.0 and len(A) < 30:
                    # Addition
                    x = rng.choice(outside)

                    new_counts, new_cost, _, new_congestion = incremental_add_cost(A, x, current_counts, t)

                    if new_cost < current_cost or (new_cost == current_cost and (new_congestion < 30 or (accept_equal and new_congestion < 60))):
                        A.add(x)
                        outside.remove(x)
                        A_list.append(x)
                        current_counts = new_counts
                        current_cost = new_cost
                        if current_cost < best_local_cost:
                            best_local_cost = current_cost
                            best_local_A = set(A)
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A)
                                t = max(2, best_cost - 1)

                # Early exit if we reach a very good cost
                if best_cost <= 2:
                    break

        # After local search, if we improved, use the best local solution
        if best_local_cost < current_cost:
            A = set(best_local_A)
            current_counts = get_counts_fast(A)
            current_cost = best_local_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = set(A)

    # Final verification and cleanup
    if best_A is None:
        # Fallback: return a valid set
        best_A = set(range(0, 129, 8))

    return list(best_A)
