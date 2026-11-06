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

    # Variable -> list of (clause_index, position_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))

    # Precompute, per variable, the list of (clause_index, sign_at_that_position)
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        idxs = []
        signs = []
        for ci, j in var_to_clauses[v]:
            idxs.append(ci)
            signs.append(clause_signs[ci, j])
        var_clause_idx.append(idxs)
        var_clause_sign.append(signs)

    # Convert to numpy arrays for faster vectorized ops where possible
    var_clause_idx_arr = [np.array(idxs, dtype=np.int32) if idxs else np.array([], dtype=np.int32) for idxs in var_clause_idx]
    var_clause_sign_arr = [np.array(signs, dtype=np.int8) if signs else np.array([], dtype=np.int8) for signs in var_clause_sign]

    # ---- State helpers ----
    def init_state(assignment):
        """Vectorized initialization of clause satisfaction counts."""
        a = np.array(assignment, dtype=np.int8)
        sat = (a[clause_vars] == clause_signs).astype(np.int32)
        csc = sat.sum(axis=1)
        return csc

    def flip_var(v, assignment, csc, unsat_count):
        """Flip variable v, update csc and unsat_count. Returns new unsat_count."""
        assignment[v] = not assignment[v]
        nv = 1 if assignment[v] else 0
        idxs = var_clause_idx[v]
        signs = var_clause_sign[v]
        for k in range(len(idxs)):
            ci = idxs[k]
            old = csc[ci]
            if nv == signs[k]:
                csc[ci] = old + 1
                if old == 0:
                    unsat_count -= 1
            else:
                csc[ci] = old - 1
                if old == 1:
                    unsat_count += 1
        return unsat_count

    def delta_if_flip(v, assignment, csc):
        """Net change in unsat count if we flip v (positive = more unsat)."""
        nv = 0 if assignment[v] else 1
        idxs = var_clause_idx[v]
        signs = var_clause_sign[v]
        delta = 0
        for k in range(len(idxs)):
            ci = idxs[k]
            old = csc[ci]
            if nv == signs[k]:
                if old == 0:
                    delta -= 1
            else:
                if old == 1:
                    delta += 1
        return delta

    def vectorized_best_flip(csc, unsat_indices, assignment):
        """
        Score-based clause selection: for each unsatisfied clause, compute the
        delta of flipping each of its 3 variables. Return the variable with
        the best (most negative) delta, with random tie-breaking.
        Uses numpy where possible for the scoring.
        """
        if len(unsat_indices) == 0:
            return -1, 0

        # For each unsat clause, compute deltas for its 3 variables
        # We'll iterate over unsat clauses but use vectorized ops per clause
        best_var = -1
        best_delta = 0  # we want most negative delta

        # Sample up to 50 unsat clauses for scoring (avoids O(m) per step)
        sample_size = min(len(unsat_indices), 50)
        if sample_size < len(unsat_indices):
            sampled = np.random.choice(unsat_indices, size=sample_size, replace=False)
        else:
            sampled = unsat_indices

        for ci in sampled:
            vs = clause_vars[ci]
            for j in range(3):
                v = int(vs[j])
                d = delta_if_flip(v, assignment, csc)
                if d < best_delta:
                    best_delta = d
                    best_var = v
                elif d == best_delta and best_var >= 0:
                    # Random tie-breaking
                    if np.random.random() < 0.5:
                        best_var = v

        return best_var, best_delta

    def greedy_pass(assignment, csc, unsat_count, max_iters=100):
        """Repeatedly flip any variable that strictly reduces unsat count."""
        deadline = timeout - 2.0
        for _ in range(max_iters):
            if (time.time() - start_time) > deadline:
                break
            if unsat_count == 0:
                return 0
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0
            # Use score-based selection
            v, d = vectorized_best_flip(csc, unsat, assignment)
            if v >= 0 and d < 0:
                unsat_count = flip_var(v, assignment, csc, unsat_count)
                if unsat_count == 0:
                    return 0
            else:
                break
        return unsat_count

    def walksat_phase(assignment, csc, current_unsat, max_flips, noise_start, noise_end, rng):
        """WalkSAT with linearly decaying noise from noise_start to noise_end."""
        deadline = timeout - 5.0
        for i in range(max_flips):
            if (time.time() - start_time) > deadline:
                break
            if current_unsat == 0:
                return 0

            noise = noise_start + (noise_end - noise_start) * (i / max(1, max_flips))
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0

            ci = int(rng.choice(unsat))
            vs = clause_vars[ci]

            if rng.random() < noise:
                v = int(vs[rng.randint(0, 2)])
            else:
                # Score-based: pick variable from this clause that minimizes total unsat
                best_var = -1
                best_score = float('inf')
                for j in range(3):
                    v = int(vs[j])
                    d = delta_if_flip(v, assignment, csc)
                    # Add small random tiebreaker
                    score = d + rng.random() * 0.01
                    if score < best_score:
                        best_score = score
                        best_var = v
                v = best_var

            if v >= 0:
                current_unsat = flip_var(v, assignment, csc, current_unsat)

            if current_unsat == 0:
                return 0

        return current_unsat

    def sa_phase(assignment, csc, current_unsat, max_flips, rng):
        """Simulated annealing: accept worse moves with probability exp(-delta/T)."""
        deadline = timeout - 5.0
        T_start = 1.0
        T_end = 0.05
        for i in range(max_flips):
            if (time.time() - start_time) > deadline:
                break
            if current_unsat == 0:
                return 0

            T = T_start + (T_end - T_start) * (i / max(1, max_flips))
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0

            ci = int(rng.choice(unsat))
            vs = clause_vars[ci]
            v = int(vs[rng.randint(0, 2)])

            d = delta_if_flip(v, assignment, csc)
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-6)):
                current_unsat = flip_var(v, assignment, csc, current_unsat)

            if current_unsat == 0:
                return 0

        return current_unsat

    def vectorized_walksat(assignment, csc, current_unsat, max_flips, rng):
        """
        Vectorized WalkSAT: each step, compute deltas for all variables in all
        unsat clauses simultaneously, then pick the best flip globally.
        This is the key structural change: global score-based selection.
        """
        deadline = timeout - 5.0
        for i in range(max_flips):
            if (time.time() - start_time) > deadline:
                break
            if current_unsat == 0:
                return 0

            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0

            # Compute delta for each variable in each unsat clause
            # Use a sampling approach for speed: consider up to 200 unsat clauses
            sample_size = min(len(unsat), 200)
            if sample_size < len(unsat):
                sampled = rng.sample(list(unsat), sample_size)
            else:
                sampled = list(unsat)

            # Collect all (variable, delta) pairs
            var_deltas = {}
            for ci in sampled:
                vs = clause_vars[ci]
                for j in range(3):
                    v = int(vs[j])
                    d = delta_if_flip(v, assignment, csc)
                    if v not in var_deltas or d < var_deltas[v]:
                        var_deltas[v] = d

            if not var_deltas:
                break

            # Pick best variable (most negative delta), with random tie-breaking
            min_delta = min(var_deltas.values())
            candidates = [v for v, d in var_deltas.items() if d == min_delta]
            v = rng.choice(candidates)

            # Accept if it improves, or with some probability if it's neutral
            d = var_deltas[v]
            if d <= 0 or rng.random() < 0.1:  # 10% chance to accept neutral/worse
                current_unsat = flip_var(v, assignment, csc, current_unsat)

            if current_unsat == 0:
                return 0

        return current_unsat

    # ---- Main search: multiple strategies per restart ----
    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(42)
    np.random.seed(42)

    # Precompute numpy arrays for vectorized init
    clause_vars_arr = clause_vars  # already numpy
    clause_signs_arr = clause_signs  # already numpy

    restart_num = 0
    while (time.time() - start_time) < timeout - 10.0:
        restart_num += 1

        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        a_arr = np.array(assignment, dtype=np.int8)
        sat = (a_arr[clause_vars_arr] == clause_signs_arr).astype(np.int32)
        csc = sat.sum(axis=1)
        current_unsat = int(np.sum(csc == 0))

        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Quick greedy pass using score-based selection
        current_unsat = greedy_pass(assignment, csc, current_unsat, max_iters=50)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 1: Vectorized WalkSAT with global score-based selection
        max_flips_p1 = 3000
        current_unsat = vectorized_walksat(assignment, csc, current_unsat, max_flips_p1, rng)
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

        # Phase 3: Traditional WalkSAT with lower noise (exploitation)
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

        # If stuck, break out
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

            # Greedy improvement using score-based selection
            for _g in range(30):
                if (time.time() - start_time) > deadline:
                    break
                if current_unsat == 0:
                    return best_assignment
                unsat = np.where(csc == 0)[0]
                if len(unsat) == 0:
                    return best_assignment
                v, d = vectorized_best_flip(csc, unsat, best_assignment)
                if v >= 0 and d < 0:
                    current_unsat = flip_var(v, best_assignment, csc, current_unsat)
                    if current_unsat == 0:
                        return best_assignment
                else:
                    break

            if current_unsat < best_local:
                best_local = current_unsat
                best_local_assign = best_assignment.copy()

            if current_unsat > best_local:
                best_assignment[:] = best_local_assign
                csc[:] = init_state(best_assignment)
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
                current_unsat = flip_var(v, best_assignment, csc, current_unsat)
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
