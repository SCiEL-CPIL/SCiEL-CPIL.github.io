# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    start_time = time.time()
    timeout = 315.0  # safety margin before the 330s kill

    # ---- Precompute clause structure (0-based) ----
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vs, ss = [], []
        for lit in clause:
            vs.append(abs(lit) - 1)
            ss.append(lit > 0)
        clause_vars.append(vs)
        clause_signs.append(ss)

    m = len(clauses)

    # Variable -> list of (clause_index, position_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, vs in enumerate(clause_vars):
        for j, v in enumerate(vs):
            var_to_clauses[v].append((i, j))

    # Precompute, per variable, the list of (clause_index, sign_at_that_position)
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        idxs = []
        signs = []
        for ci, j in var_to_clauses[v]:
            idxs.append(ci)
            signs.append(clause_signs[ci][j])
        var_clause_idx.append(idxs)
        var_clause_sign.append(signs)

    # ---- State helpers ----
    def init_state(assignment):
        """Vectorized initialization of clause satisfaction counts."""
        a = np.array(assignment, dtype=np.int8)
        vars_arr = np.array(clause_vars, dtype=np.int32)
        signs_arr = np.array(clause_signs, dtype=np.int8)
        sat = (a[vars_arr] == signs_arr).astype(np.int32)
        csc = sat.sum(axis=1)
        return csc

    def flip_var(v, assignment, csc, unsat_count):
        """Flip variable v, update csc and unsat_count. Returns new unsat_count."""
        assignment[v] = not assignment[v]
        nv = assignment[v]
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
        nv = not assignment[v]
        idxs = var_clause_idx[v]
        signs = var_clause_sign[v]
        delta = 0
        for k in range(len(idxs)):
            ci = idxs[k]
            old = csc[ci]
            if nv == signs[k]:
                # becomes satisfied
                if old == 0:
                    delta -= 1
            else:
                # becomes unsatisfied
                if old == 1:
                    delta += 1
        return delta

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
            cand = set()
            freq = {}
            for ci in unsat:
                for v in clause_vars[ci]:
                    cand.add(v)
                    freq[v] = freq.get(v, 0) + 1
            ordered = sorted(cand, key=lambda v: -freq.get(v, 0))
            improved = False
            for v in ordered:
                if (time.time() - start_time) > deadline:
                    break
                d = delta_if_flip(v, assignment, csc)
                if d < 0:
                    unsat_count = flip_var(v, assignment, csc, unsat_count)
                    improved = True
                    if unsat_count == 0:
                        return 0
            if not improved:
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
                v = vs[rng.randint(0, 2)]
            else:
                # Greedy: satisfy most other unsat clauses
                best_var = -1
                best_score = -1.0
                for j in range(3):
                    v = vs[j]
                    score = 0
                    idxs = var_clause_idx[v]
                    signs = var_clause_sign[v]
                    for k in range(len(idxs)):
                        oci = idxs[k]
                        if oci != ci and csc[oci] == 0:
                            score += 1
                    score += rng.random() * 0.05
                    if score > best_score:
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
            v = vs[rng.randint(0, 2)]

            d = delta_if_flip(v, assignment, csc)
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-6)):
                current_unsat = flip_var(v, assignment, csc, current_unsat)

            if current_unsat == 0:
                return 0

        return current_unsat

    # ---- Main search: multiple strategies per restart ----
    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(42)

    # Precompute numpy arrays for vectorized init
    clause_vars_arr = np.array(clause_vars, dtype=np.int32)
    clause_signs_arr = np.array(clause_signs, dtype=np.int8)

    # Adaptive restart budget: use more time for later restarts if going well
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

        # Quick greedy pass
        current_unsat = greedy_pass(assignment, csc, current_unsat, max_iters=50)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 1: WalkSAT with high noise (exploration)
        max_flips_p1 = 3000
        current_unsat = walksat_phase(assignment, csc, current_unsat, max_flips_p1, 0.35, 0.20, rng)
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

        # Phase 3: WalkSAT with lower noise (exploitation)
        max_flips_p2 = 4000
        current_unsat = walksat_phase(assignment, csc, current_unsat, max_flips_p2, 0.18, 0.08, rng)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Phase 4: Simulated annealing
        max_flips_p3 = 3000
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

        # If we're doing well, use shorter restarts to try more
        # If stuck, use longer restarts
        if best_unsat <= 10:
            pass  # already solving fast
        elif best_unsat > 50 and restart_num >= 5:
            # We're struggling; break out to final polish
            break

    # Final deep polish on the best assignment found
    if best_assignment is not None:
        csc = init_state(best_assignment)
        current_unsat = int(np.sum(csc == 0))

        # Deep polish with perturbations to escape local minima
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
            improved = False
            for _g in range(50):
                if (time.time() - start_time) > deadline:
                    break
                if current_unsat == 0:
                    return best_assignment
                unsat = np.where(csc == 0)[0]
                if len(unsat) == 0:
                    return best_assignment
                cand = set()
                freq = {}
                for ci in unsat:
                    for v in clause_vars[ci]:
                        cand.add(v)
                        freq[v] = freq.get(v, 0) + 1
                ordered = sorted(cand, key=lambda v: -freq.get(v, 0))
                any_improved = False
                for v in ordered:
                    d = delta_if_flip(v, best_assignment, csc)
                    if d < 0:
                        current_unsat = flip_var(v, best_assignment, csc, current_unsat)
                        any_improved = True
                        if current_unsat == 0:
                            return best_assignment
                if not any_improved:
                    break

            if current_unsat < best_local:
                best_local = current_unsat
                best_local_assign = best_assignment.copy()

            if current_unsat > best_local:
                # Restore best and perturb
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
                v = clause_vars[ci][rng_local.randint(0, 2)]
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
