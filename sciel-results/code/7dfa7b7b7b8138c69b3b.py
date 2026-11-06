# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    start_time = time.time()
    timeout = 315.0  # safety margin before the 330s kill

    m = len(clauses)
    if m == 0:
        return [True] * n

    # ---- Precompute clause structure (0-based) ----
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)  # 1 if positive, 0 if negative
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else 0

    # Precompute flattened arrays for vectorized delta computation
    vars_occ = clause_vars.flatten()
    signs_occ = clause_signs.flatten()
    clause_idx_occ = np.repeat(np.arange(m), 3)

    # Variable -> list of (clause_index, position_in_clause) for flip updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))

    # Convert to numpy arrays for faster vectorized ops where possible
    var_clause_idx_arr = [np.array([ci for ci, j in var_to_clauses[v]], dtype=np.int32) if var_to_clauses[v] else np.array([], dtype=np.int32) for v in range(n)]
    var_clause_sign_arr = [np.array([clause_signs[ci, j] for ci, j in var_to_clauses[v]], dtype=np.int8) if var_to_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]

    # ---- State helpers ----
    def init_state(assignment):
        """Vectorized initialization of clause satisfaction counts."""
        a = np.array(assignment, dtype=np.int8)
        sat = (a[clause_vars] == clause_signs).astype(np.int32)
        csc = sat.sum(axis=1)
        return csc

    def compute_all_deltas(assignment, csc):
        """
        Compute the change in unsat count for flipping each variable.
        Returns an array of length n.
        """
        a = np.array(assignment, dtype=np.int8)
        # current_matches[k] is True if literal k is currently satisfied
        current_matches = (a[vars_occ] == signs_occ)
        
        # For each occurrence, determine contribution to delta if flipped
        # If currently satisfied (True): flipping makes it unsatisfied.
        #   If csc[clause] == 1, clause becomes unsat -> delta +1
        #   If csc[clause] > 1, clause stays sat -> delta 0
        # If currently unsatisfied (False): flipping makes it satisfied.
        #   If csc[clause] == 0, clause becomes sat -> delta -1
        #   If csc[clause] > 0, clause stays sat -> delta 0
        
        csc_occ = csc[clause_idx_occ]
        
        delta_sat_to_unsat = (csc_occ == 1).astype(np.int8)
        delta_unsat_to_sat = (csc_occ == 0).astype(np.int8) * -1
        
        # Select based on current match status
        # If current_matches is True, use delta_sat_to_unsat
        # If current_matches is False, use delta_unsat_to_sat
        delta_contrib = np.where(current_matches, delta_sat_to_unsat, delta_unsat_to_sat)
        
        # Sum contributions for each variable
        all_deltas = np.bincount(vars_occ, weights=delta_contrib, minlength=n)
        
        return all_deltas

    def flip_var(v, assignment, csc):
        """Flip variable v, update csc. Returns new unsat_count change."""
        assignment[v] = not assignment[v]
        nv = 1 if assignment[v] else 0
        idxs = var_clause_idx_arr[v]
        signs = var_clause_sign_arr[v]
        
        unsat_change = 0
        for k in range(len(idxs)):
            ci = idxs[k]
            old = csc[ci]
            if nv == signs[k]:
                csc[ci] = old + 1
                if old == 0:
                    unsat_change -= 1
            else:
                csc[ci] = old - 1
                if old == 1:
                    unsat_change += 1
        return unsat_change

    def greedy_pass(assignment, csc, unsat_count, max_iters=100):
        """Repeatedly flip any variable that strictly reduces unsat count using vectorized deltas."""
        deadline = timeout - 2.0
        for _ in range(max_iters):
            if (time.time() - start_time) > deadline:
                break
            if unsat_count == 0:
                return 0
            
            # Compute deltas for all variables
            all_deltas = compute_all_deltas(assignment, csc)
            
            # Find the variable with the most negative delta
            min_delta = np.min(all_deltas)
            
            if min_delta < 0:
                # Find all variables with this min delta
                candidates = np.where(all_deltas == min_delta)[0]
                v = int(candidates[np.random.randint(0, len(candidates))])
                
                # Flip
                unsat_change = flip_var(v, assignment, csc)
                unsat_count += unsat_change
                
                if unsat_count == 0:
                    return 0
            else:
                break
        return unsat_count

    def walksat_phase(assignment, csc, current_unsat, max_flips, noise_start, noise_end, rng):
        """WalkSAT with linearly decaying noise, using vectorized deltas for scoring."""
        deadline = timeout - 5.0
        for i in range(max_flips):
            if (time.time() - start_time) > deadline:
                break
            if current_unsat == 0:
                return 0

            noise = noise_start + (noise_end - noise_start) * (i / max(1, max_flips))
            
            if rng.random() < noise:
                # Random move: pick a random unsat clause, then a random var in it
                unsat = np.where(csc == 0)[0]
                if len(unsat) == 0:
                    return 0
                ci = int(rng.choice(unsat))
                v = int(clause_vars[ci][rng.randint(0, 2)])
            else:
                # Score-based move: pick var with min delta
                all_deltas = compute_all_deltas(assignment, csc)
                min_delta = np.min(all_deltas)
                candidates = np.where(all_deltas == min_delta)[0]
                v = int(candidates[rng.randint(0, len(candidates))])

            unsat_change = flip_var(v, assignment, csc)
            current_unsat += unsat_change

            if current_unsat == 0:
                return 0

        return current_unsat

    def sa_phase(assignment, csc, current_unsat, max_flips, rng):
        """Simulated annealing."""
        deadline = timeout - 5.0
        T_start = 1.0
        T_end = 0.05
        for i in range(max_flips):
            if (time.time() - start_time) > deadline:
                break
            if current_unsat == 0:
                return 0

            T = T_start + (T_end - T_start) * (i / max(1, max_flips))
            
            # Pick a random variable from an unsat clause
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0
            ci = int(rng.choice(unsat))
            v = int(clause_vars[ci][rng.randint(0, 2)])

            # Compute delta for this specific flip
            all_deltas = compute_all_deltas(assignment, csc)
            d = all_deltas[v]
            
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-6)):
                unsat_change = flip_var(v, assignment, csc)
                current_unsat += unsat_change

            if current_unsat == 0:
                return 0

        return current_unsat

    # ---- Main search ----
    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(42)
    np.random.seed(42)

    restart_num = 0
    while (time.time() - start_time) < timeout - 10.0:
        restart_num += 1

        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        csc = init_state(assignment)
        current_unsat = int(np.sum(csc == 0))

        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Quick greedy pass
        current_unsat = greedy_pass(assignment, csc, current_unsat, max_iters=50)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 1: WalkSAT with high noise (exploration)
        max_flips_p1 = 3000
        current_unsat = walksat_phase(assignment, csc, current_unsat, max_flips_p1, 0.5, 0.1, rng)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 2: Greedy polish
        current_unsat = greedy_pass(assignment, csc, current_unsat, max_iters=50)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 3: WalkSAT with low noise (exploitation)
        max_flips_p2 = 3000
        current_unsat = walksat_phase(assignment, csc, current_unsat, max_flips_p2, 0.15, 0.05, rng)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 4: Simulated annealing
        max_flips_p3 = 2000
        current_unsat = sa_phase(assignment, csc, current_unsat, max_flips_p3, rng)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 5: Final greedy polish
        current_unsat = greedy_pass(assignment, csc, current_unsat, max_iters=100)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        if best_unsat > 50 and restart_num >= 8:
            break

    # Final deep polish on the best assignment found
    if best_assignment is not None:
        csc = init_state(best_assignment)
        current_unsat = int(np.sum(csc == 0))

        deadline = timeout - 2.0
        rng_local = random.Random(999)
        best_local = current_unsat
        best_local_assign = best_assignment.copy()

        for _ in range(500):
            if (time.time() - start_time) > deadline:
                break
            if current_unsat == 0:
                return best_assignment

            # Greedy improvement
            for _g in range(30):
                if (time.time() - start_time) > deadline:
                    break
                if current_unsat == 0:
                    return best_assignment
                all_deltas = compute_all_deltas(best_assignment, csc)
                min_delta = np.min(all_deltas)
                if min_delta < 0:
                    candidates = np.where(all_deltas == min_delta)[0]
                    v = int(candidates[rng_local.randint(0, len(candidates))])
                    unsat_change = flip_var(v, best_assignment, csc)
                    current_unsat += unsat_change
                    if current_unsat == 0:
                        return best_assignment
                else:
                    break

            if current_unsat < best_local:
                best_local = current_unsat
                best_local_assign = best_assignment.copy()

            if current_unsat > best_local:
                best_assignment[:] = best_local_assign
                csc = init_state(best_assignment)
                current_unsat = best_local

            # Small random perturbation
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return best_assignment
            n_perturb = rng_local.randint(1, 3)
            for _p in range(n_perturb):
                if len(unsat) == 0:
                    break
                ci = unsat[rng_local.randint(0, len(unsat) - 1)]
                v = int(clause_vars[ci][rng_local.randint(0, 2)])
                unsat_change = flip_var(v, best_assignment, csc)
                current_unsat += unsat_change
                unsat = np.where(csc == 0)[0]

            if current_unsat == 0:
                return best_assignment

        # Final greedy
        current_unsat = greedy_pass(best_assignment, csc, current_unsat, max_iters=200)
        if current_unsat == 0:
            return best_assignment

        return best_assignment

    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
