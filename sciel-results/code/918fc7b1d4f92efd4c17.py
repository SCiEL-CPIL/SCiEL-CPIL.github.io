import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay well under 90s kill

    N = 256

    # Precompute all APs and build incidence structure
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    # Iterate over all valid (start, step) pairs
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d

            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)

    num_aps = len(ap_indices)

    def thue_morse(N):
        """Generate Thue-Morse sequence as +/-1 signs."""
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def compute_sums(signs):
        """Compute sums for all APs given a signs array."""
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(signs[idx_list])
        return sums

    def greedy_optimize(s_init, time_budget, rng):
        """
        Improved Greedy Coordinate Descent with:
        - Fast sum updates using precomputed affected AP lists
        - Smarter tie-breaking based on number of tight constraints
        - Periodic perturbation to escape local minima
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))

        no_improve_sweeps = 0
        max_no_improve = 5

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved_this_sweep = False

            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = aps_containing[i]

                # Save old sums for affected APs
                old_sums = sums[affected].copy()

                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    improved_this_sweep = True
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                elif new_max == current_max:
                    # Count how many APs are at the boundary (|sum| == current_max)
                    old_tight = np.sum(np.abs(old_sums) == current_max)
                    new_tight = np.sum(np.abs(sums[affected]) == current_max)
                    if new_tight < old_tight:
                        improved_this_sweep = True
                    elif new_tight == old_tight:
                        if rng.random() < 0.02:
                            improved_this_sweep = True
                        else:
                            sums[affected] = old_sums
                            s[i] = old_val
                    else:
                        sums[affected] = old_sums
                        s[i] = old_val
                else:
                    sums[affected] = old_sums
                    s[i] = old_val

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    if rng.random() < 0.5:
                        num_flips = rng.randint(3, 10)
                        for _ in range(num_flips):
                            j = rng.choice(indices_list)
                            s[j] = -s[j]
                            sums[aps_containing[j]] += -2 * s[j]
                        current_max = int(np.max(np.abs(sums)))
                        no_improve_sweeps = 0
                    else:
                        break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    # 2. SAT-based approach using Max-SAT formulation
    # We'll use a simple CDCL-inspired approach with clause learning
    # For each target D, we want to check if there exists signs such that |sum(ap)| <= D for all APs
    
    def sat_check(D, max_clauses=None, time_budget=5.0):
        """
        Check if there exists a sign assignment with max AP sum <= D.
        Uses a simple local search with clause-based constraints.
        Returns (success, signs) or (False, None).
        """
        deadline = time.time() + time_budget
        
        # Filter to "critical" APs: those with length >= 2 (singletons always have |sum|=1 <= D for D>=1)
        # Also, for large D, many APs are automatically satisfied
        critical_aps = []
        for ap_idx, idx_list in enumerate(ap_indices):
            if len(idx_list) <= D:
                # This AP can never violate |sum| > D since max possible sum is len(idx_list)
                continue
            critical_aps.append((ap_idx, idx_list))
        
        if not critical_aps:
            return True, None
        
        # Start from a random initialization
        rng = random.Random(hash(D) % 2**32)
        s = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        
        # Compute initial violations
        def get_violations(signs):
            """Return list of (ap_idx, sum) where |sum| > D"""
            violations = []
            for ap_idx, idx_list in critical_aps:
                sm = int(np.sum(signs[idx_list]))
                if abs(sm) > D:
                    violations.append((ap_idx, sm))
            return violations
        
        violations = get_violations(s)
        if not violations:
            return True, s.copy()
        
        # Local search: repeatedly flip a sign that reduces the number of violations
        best_violations = len(violations)
        best_s = s.copy()
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Pick a random violated AP
            if not violations:
                return True, s.copy()
            
            ap_idx, sm = violations[rng.randint(0, len(violations) - 1)]
            idx_list = ap_indices[ap_idx]
            
            # Try flipping each element in this AP to see if it reduces violations
            best_flip = None
            best_new_violations = len(violations)
            
            for pos in idx_list:
                if time.time() > deadline:
                    break
                
                # Flip this position
                old_val = s[pos]
                s[pos] = -old_val
                
                # Recompute violations (expensive but we limit iterations)
                new_violations = get_violations(s)
                new_count = len(new_violations)
                
                if new_count < best_new_violations:
                    best_new_violations = new_count
                    best_flip = pos
                    # Keep the flip
                else:
                    # Revert
                    s[pos] = old_val
            
            if best_flip is not None:
                # Apply the best flip
                s[best_flip] = -s[best_flip]
                violations = get_violations(s)
                if len(violations) < best_violations:
                    best_violations = len(violations)
                    best_s = s.copy()
                    if best_violations == 0:
                        return True, best_s
            else:
                # No improvement found; try a random perturbation
                if rng.random() < 0.3:
                    num_flips = rng.randint(1, 5)
                    for _ in range(num_flips):
                        j = rng.randint(0, N - 1)
                        s[j] = -s[j]
                    violations = get_violations(s)
                    if len(violations) < best_violations:
                        best_violations = len(violations)
                        best_s = s.copy()
                else:
                    # Stuck; break out
                    break
        
        if best_violations == 0:
            return True, best_s
        return False, best_s

    # 3. Main loop: combine greedy and SAT-based approaches
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Run 1: Thue-Morse + greedy
    tm = thue_morse(N)
    s1, score1 = greedy_optimize(tm, time_limit * 0.25, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Try SAT-based approach for small D values
    # Start from D=3 and go up
    for D in [3, 4, 5]:
        if time.time() - start_time > time_limit * 0.7:
            break
        
        success, sat_s = sat_check(D, time_budget=5.0)
        if success:
            # Verify and refine
            sums = compute_sums(sat_s)
            actual_max = int(np.max(np.abs(sums)))
            if actual_max < best_global_score:
                best_global_score = actual_max
                best_global_s = sat_s.copy()
                # Try to improve further with greedy
                refined_s, refined_score = greedy_optimize(sat_s, 3.0, rng_global)
                if refined_score < best_global_score:
                    best_global_score = refined_score
                    best_global_s = refined_s.copy()
                break  # Found a valid solution for this D; no need to try larger D

    # Run 3: Random init + greedy
    if time.time() - start_time < time_limit * 0.8:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.15, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Remaining time: more greedy restarts
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 1.0:
        num_extra = min(5, int(remaining_budget / 2.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            budget = min(2.0, time_limit - (time.time() - start_time) - 1.0)
            if budget < 0.5:
                break
            s_k, score_k = greedy_optimize(rand_init, budget, rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
