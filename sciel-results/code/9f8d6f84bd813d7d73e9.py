"""Sparse squares of twelve-term integer polynomials - DFT spectral surrogate + collision search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    best_poly = None
    best_nz = float('inf')

    N_DFT = 8192  # 2^13, enough to avoid circular convolution for degree <= 40

    def square_nonzero_fast(exponents, coeffs):
        """Fast exact square nonzero count."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def spectral_l1(exponents, coeffs):
        """Compute L1 norm of DFT spectrum as surrogate for square sparsity."""
        c = np.asarray(coeffs, dtype=np.float64)
        e = np.asarray(exponents, dtype=np.int64)
        # Build dense polynomial
        max_deg = max(e) if len(e) > 0 else 0
        poly = np.zeros(max_deg + 1, dtype=np.float64)
        for i, (ex, co) in enumerate(zip(e, c)):
            poly[ex] = co
        # Pad to N_DFT
        padded = np.zeros(N_DFT, dtype=np.float64)
        padded[:len(poly)] = poly
        # DFT
        spectrum = np.fft.fft(padded)
        # L1 norm of squared spectrum magnitude
        return float(np.sum(np.abs(spectrum ** 2)))

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def try_candidate(exponents, coeffs):
        nonlocal best_poly, best_nz
        if len(exponents) != 12 or len(set(exponents)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exponents):
            return
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return
        nz = square_nonzero_fast(exponents, coeffs)
        if nz < best_nz:
            best_nz = nz
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    random.seed(42)

    def spectral_guided_sa(initial_exps, initial_coefs, max_iters=5000, t0=10.0, t_final=0.01, cooling=0.9995):
        """SA with spectral L1 as secondary objective to guide moves."""
        nonlocal best_poly, best_nz
        cur_exps = sorted(initial_exps)
        cur_coefs = list(initial_coefs)
        cur_nz = square_nonzero_fast(cur_exps, cur_coefs)
        cur_spec = spectral_l1(cur_exps, cur_coefs)
        
        if cur_nz < best_nz:
            best_nz = cur_nz
            max_deg = max(cur_exps)
            poly = [0] * (max_deg + 1)
            for e, c in zip(cur_exps, cur_coefs):
                poly[e] = c
            best_poly = poly

        best_in_run_exps = list(cur_exps)
        best_in_run_coefs = list(cur_coefs)
        best_in_run_nz = cur_nz
        best_in_run_spec = cur_spec

        # Track recent best spectral values for adaptive weighting
        spec_history = []

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Temperature schedule
            t = t0 * (cooling ** it)
            if t < t_final:
                t = t_final

            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)

            r = random.random()
            if r < 0.20:
                # Move one exponent
                idx = random.randint(0, 11)
                shift = random.randint(-8, 8)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif r < 0.35:
                # Structural swap: exchange two exponents to create sum collisions
                i, j = random.sample(range(12), 2)
                si = random.randint(-3, 3)
                sj = random.randint(-3, 3)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            elif r < 0.45:
                # Swap two exponents exactly (reorder)
                i, j = random.sample(range(12), 2)
                new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                new_exps.sort()
            elif r < 0.60:
                # Modify a coefficient
                idx = random.randint(0, 11)
                new_val = random.choice([-8,-6,-4,-3,-2,-1,1,2,3,4,5,6,8,12])
                if new_val != new_coefs[idx]:
                    new_coefs[idx] = new_val
            elif r < 0.70:
                # Modify two coefficients
                i, j = random.sample(range(12), 2)
                new_coefs[i] = random.choice([-4,-3,-2,-1,1,2,3,4,5])
                new_coefs[j] = random.choice([-4,-3,-2,-1,1,2,3,4,5])
            elif r < 0.80:
                # Swap two coefficients
                i, j = random.sample(range(12), 2)
                new_coefs[i], new_coefs[j] = new_coefs[j], new_coefs[i]
            elif r < 0.90:
                # Move two exponents
                i, j = random.sample(range(12), 2)
                si = random.choice([-5,-4,-3,-2,-1,1,2,3,4,5])
                sj = random.choice([-5,-4,-3,-2,-1,1,2,3,4,5])
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            else:
                # Large jump: reset one exponent
                idx = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue

            # Validate
            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue

            new_nz = square_nonzero_fast(new_exps, new_coefs)
            new_spec = spectral_l1(new_exps, new_coefs)

            # Combined objective: primary is nz, secondary is spectral L1
            # Use spectral info to break ties and guide exploration
            delta_nz = new_nz - cur_nz
            delta_spec = new_spec - cur_spec

            # Acceptance: prefer lower nz, then lower spectral L1
            if delta_nz < 0:
                accept = True
            elif delta_nz == 0:
                # Same nz: accept if spectral improves, or with probability based on spectral delta
                if delta_spec < 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta_spec / (t * scale))
                    spec_scale = max(cur_spec * 0.01, 1.0)
                    accept = random.random() < np.exp(-delta_spec / max(t * spec_scale, 1e-10))
            else:
                # Worse nz: accept with probability exp(-delta_nz/t) * spectral bonus
                spec_bonus = 1.0 if new_spec < cur_spec else 0.5
                accept = random.random() < np.exp(-delta_nz / max(t, 1e-10)) * spec_bonus

            if accept:
                cur_nz = new_nz
                cur_exps = list(new_exps)
                cur_coefs = list(new_coefs)
                cur_spec = new_spec

                spec_history.append(cur_spec)
                if len(spec_history) > 100:
                    spec_history.pop(0)

                if cur_nz < best_in_run_nz or (cur_nz == best_in_run_nz and cur_spec < best_in_run_spec):
                    best_in_run_nz = cur_nz
                    best_in_run_spec = cur_spec
                    best_in_run_exps = list(cur_exps)
                    best_in_run_coefs = list(cur_coefs)

                if cur_nz < best_nz:
                    best_nz = cur_nz
                    max_deg = max(cur_exps)
                    poly = [0] * (max_deg + 1)
                    for e, c in zip(cur_exps, cur_coefs):
                        poly[e] = c
                    best_poly = poly

        # Return best found in this run
        return best_in_run_exps, best_in_run_coefs, best_in_run_nz

    # Phase 1: Generate diverse starting points
    random.seed(42)
    starting_points = []

    # AP-based starts
    for d in [1, 2, 3, 4, 5]:
        max_a = max(0, 40 - 11 * d)
        for a in range(0, min(max_a + 1, 5)):
            exps = [a + i * d for i in range(12)]
            if exps[-1] < 41:
                coefs = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coefs))
                coefs2 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                starting_points.append((exps, coefs2))

    # Two-AP starts
    for d1, d2 in [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]:
        for a1 in [0, 2, 4]:
            for gap in [5, 8, 12]:
                a2 = a1 + gap
                exp1 = [a1 + i * d1 for i in range(6)]
                exp2 = [a2 + i * d2 for i in range(6)]
                all_exps = sorted(exp1 + exp2)
                if len(set(all_exps)) == 12 and all(e < 41 for e in all_exps):
                    coefs = [1] * 6 + [-1] * 6
                    starting_points.append((all_exps, coefs))
                    coefs_alt = [1, -1, 1, -1, 1, -1] + [1, -1, 1, -1, 1, -1]
                    starting_points.append((all_exps, coefs_alt))

    # Symmetric starts
    for center in [10, 15, 20]:
        for offsets in [[1,2,3,4,5,6], [1,2,3,4,6,8], [2,3,4,5,6,7]]:
            exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
            if all(0 <= e < 41 for e in exps) and len(set(exps)) == 12:
                coefs = [1] * 6 + [-1] * 6
                starting_points.append((exps, coefs))
                coefs2 = [1, -1, 1, -1, 1, -1] * 2
                starting_points.append((exps, coefs2))

    # Random starts
    for _ in range(15):
        exps = sorted(random.sample(range(41), 12))
        coefs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        starting_points.append((exps, coefs))

    # Phase 2: Run SA from multiple starting points
    # Evaluate initial candidates first
    for exps, coefs in starting_points:
        if time.time() > deadline - 10:
            break
        try_candidate(exps, coefs)

    # Run spectral-guided SA from each starting point with time budget
    num_sa_runs = min(len(starting_points), 30)
    time_per_run = (deadline - time.time() - 5) / max(num_sa_runs, 1)
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 5:
            break
        exps, coefs = starting_points[i]
        # Estimate iterations based on time budget (DFT adds overhead)
        est_iters = int(time_per_run * 1500)  # ~1500 iters/sec estimate with DFT
        est_iters = max(300, min(est_iters, 5000))
        spectral_guided_sa(exps, coefs, max_iters=est_iters)

    # Phase 3: Continue SA from the best solution found
    if best_poly is not None and time.time() < deadline - 8:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]
        
        # Multiple SA runs from perturbed best
        remaining_time = deadline - time.time() - 5
        num_extra = max(1, int(remaining_time / 3.0))
        
        for restart in range(min(num_extra, 20)):
            if time.time() > deadline - 5:
                break
            
            pert_exp = list(best_exp)
            pert_coef = list(best_coef)
            
            # Perturb
            num_pert = random.randint(1, 3)
            for _ in range(num_pert):
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    shift = random.randint(-5, 5)
                    new_e = pert_exp[idx] + shift
                    if 0 <= new_e < 41 and new_e not in pert_exp:
                        pert_exp[idx] = new_e
                        pert_exp.sort()
                else:
                    idx = random.randint(0, 11)
                    pert_coef[idx] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                    if pert_coef[idx] == 0:
                        pert_coef[idx] = 1
            
            if len(set(pert_exp)) == 12 and all(0 <= e < 41 for e in pert_exp) and \
               all(c != 0 and abs(c) <= 64 for c in pert_coef):
                remaining = deadline - time.time() - 5
                est_iters = max(300, int(remaining * 1500 / max(remaining_time / 3.0, 1)))
                spectral_guided_sa(pert_exp, pert_coef, max_iters=min(est_iters, 4000))

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
