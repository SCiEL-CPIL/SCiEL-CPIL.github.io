# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    start_time = time.time()
    timeout = 320.0  # safety margin before the 330s kill

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
        # assignment is a list of bools; convert to numpy for vectorized ops
        a = np.array(assignment, dtype=np.int8)
        # For each clause, check if each literal is satisfied
        # clause_vars is list of lists, clause_signs is list of lists
        # Build arrays: vars_arr shape (m, 3), signs_arr shape (m, 3)
        vars_arr = np.array(clause_vars, dtype=np.int32)
        signs_arr = np.array(clause_signs, dtype=np.int8)
        # a[vars_arr] gives the assignment values for each literal's variable
        # A literal is satisfied if (a[var] == 1) == sign
        # i.e., (a[var] * 2 - 1) * (sign * 2 - 1) > 0
        # More simply: satisfied if (a[var] == 1) XOR (sign == 0) is False
        # i.e., a[var] == sign
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
                if old == 0:
                    delta -= 1
            else:
                if old == 1:
                    delta += 1
        return delta

    def unit_propagation(assignment, csc, unsat_count):
        """Apply unit propagation: if a clause has exactly one unassigned/unsatisfied
        literal, force that variable. Returns (assignment, csc, unsat_count, changed)."""
        changed = False
        deadline = timeout - 3.0
        # Find clauses with csc == 1 (exactly one literal satisfied)
        # Actually for unit prop we need clauses with exactly one UNSATISFIED literal
        # i.e., csc == 2 (two satisfied, one not)
        # Wait: csc counts how many literals are satisfied.
        # A clause is unit if exactly 1 literal is still "free" (unsatisfied).
        # In our binary assignment, every literal is either sat or not.
        # So unit clause = csc == 2 (2 sat, 1 not sat)
        # The unsatisfied literal's variable should be flipped to satisfy it.
        for _ in range(100):  # max iterations to avoid infinite loop
            if (time.time() - start_time) > deadline:
                break
            # Find clauses with exactly 2 satisfied literals
            unit_clauses = np.where(csc == 2)[0]
            if len(unit_clauses) == 0:
                break
            any_flipped = False
            for ci in unit_clauses:
                if (time.time() - start_time) > deadline:
                    break
                vs = clause_vars[ci]
                ss = clause_signs[ci]
                # Find the unsatisfied literal
                for j in range(3):
                    if assignment[vs[j]] != ss[j]:
                        # Flip this variable to satisfy the clause
                        assignment[vs[j]] = ss[j]
                        # Update csc
                        v = vs[j]
                        idxs = var_clause_idx[v]
                        signs = var_clause_sign[v]
                        nv = assignment[v]
                        for k in range(len(idxs)):
                            oci = idxs[k]
                            old = csc[oci]
                            if nv == signs[k]:
                                csc[oci] = old + 1
                                if old == 0:
                                    unsat_count -= 1
                            else:
                                csc[oci] = old - 1
                                if old == 1:
                                    unsat_count += 1
                        any_flipped = True
                        changed = True
                        if unsat_count == 0:
                            return assignment, csc, 0, True
                        break  # Only flip one variable per clause
            if not any_flipped:
                break
        return assignment, csc, unsat_count, changed

    def polish(assignment, csc, unsat_count, max_passes=100):
        """High-precision polish: repeatedly flip any variable that strictly
        reduces the unsat count. Use a worklist of variables in unsat clauses.
        Continue until a full pass makes no improvement (local optimum)."""
        deadline = timeout - 3.0
        for _ in range(max_passes):
            if (time.time() - start_time) > deadline:
                break
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0
            # Candidate variables: those in at least one unsat clause
            cand = set()
            freq = {}
            for ci in unsat:
                for v in clause_vars[ci]:
                    cand.add(v)
                    freq[v] = freq.get(v, 0) + 1
            # Sort candidates by frequency in unsat clauses (heuristic)
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
            # Try unit propagation after greedy pass
            if improved:
                assignment, csc, unsat_count, _ = unit_propagation(assignment, csc, unsat_count)
                if unsat_count == 0:
                    return 0
            if not improved:
                break
        return int(np.sum(csc == 0))

    def deep_polish(assignment, csc, unsat_count, max_passes=300):
        """Deeper polish: alternate between greedy flips, unit propagation, and
        small random perturbations to escape local minima."""
        deadline = timeout - 2.0
        rng_local = random.Random(12345)
        best_local = unsat_count
        best_local_assign = assignment.copy()
        
        for _ in range(max_passes):
            if (time.time() - start_time) > deadline:
                break
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return 0
            # Try greedy improvement
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
            # Apply unit propagation
            if improved:
                assignment, csc, unsat_count, _ = unit_propagation(assignment, csc, unsat_count)
                if unsat_count == 0:
                    return 0
            if not improved:
                # Stuck: try a small random perturbation
                if unsat_count < best_local:
                    best_local = unsat_count
                    best_local_assign = assignment.copy()
                # Flip 1-3 random variables in unsat clauses
                n_perturb = rng_local.randint(1, 3)
                for _p in range(n_perturb):
                    if len(unsat) == 0:
                        break
                    ci = unsat[rng_local.randint(0, len(unsat) - 1)]
                    v = clause_vars[ci][rng_local.randint(0, 2)]
                    unsat_count = flip_var(v, assignment, csc, unsat_count)
                    unsat = np.where(csc == 0)[0]
                if unsat_count == 0:
                    return 0
        # Restore best if we got worse
        if unsat_count > best_local:
            assignment[:] = best_local_assign
            csc[:] = init_state(assignment)
            return best_local
        return unsat_count

    # ---- Main WalkSAT search with restarts + adaptive noise ----
    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(42)
    max_flips_per_restart = 20000
    noise = 0.20

    # Precompute numpy arrays for vectorized init
    clause_vars_arr = np.array(clause_vars, dtype=np.int32)
    clause_signs_arr = np.array(clause_signs, dtype=np.int8)

    while (time.time() - start_time) < timeout - 5.0:
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

        # Quick greedy pass to improve initial assignment
        for _ in range(3):
            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return assignment
            cand = set()
            for ci in unsat:
                cand.update(clause_vars[ci])
            any_improved = False
            for v in cand:
                d = delta_if_flip(v, assignment, csc)
                if d < 0:
                    current_unsat = flip_var(v, assignment, csc, current_unsat)
                    any_improved = True
                    if current_unsat == 0:
                        return assignment
            if not any_improved:
                break
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # WalkSAT loop
        for _ in range(max_flips_per_restart):
            if (time.time() - start_time) > timeout - 5.0:
                break

            if current_unsat == 0:
                return assignment

            unsat = np.where(csc == 0)[0]
            if len(unsat) == 0:
                return assignment

            ci = int(rng.choice(unsat))
            vs = clause_vars[ci]

            if rng.random() < noise:
                v = vs[rng.randint(0, 2)]
            else:
                # Vectorized greedy: score each of the 3 variables by how many
                # other unsat clauses they appear in (plus tiny jitter)
                best_var = -1
                best_score = -1.0
                unsat_set = set(unsat.tolist())
                for j in range(3):
                    v = vs[j]
                    idxs = var_clause_idx[v]
                    # Count how many of v's clauses (other than ci) are unsat
                    score = 0
                    for k in range(len(idxs)):
                        oci = idxs[k]
                        if oci != ci and oci in unsat_set:
                            score += 1
                    score += rng.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_var = v
                v = best_var

            if v >= 0:
                current_unsat = flip_var(v, assignment, csc, current_unsat)

            if current_unsat == 0:
                return assignment
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()

            # Adaptive noise: widen if stuck, tighten if improving
            if current_unsat > best_unsat:
                noise = min(0.5, noise + 0.02)
            else:
                noise = max(0.08, noise - 0.01)

    # Final deep polish on the best assignment found
    if best_assignment is not None:
        csc = init_state(best_assignment)
        current_unsat = int(np.sum(csc == 0))
        # First try unit propagation
        best_assignment, csc, current_unsat, _ = unit_propagation(best_assignment, csc, current_unsat)
        if current_unsat == 0:
            return best_assignment
        final_unsat = deep_polish(best_assignment, csc, current_unsat, max_passes=300)
        if final_unsat == 0:
            return best_assignment
        # Also try standard polish as backup
        csc2 = init_state(best_assignment)
        unsat2 = polish(best_assignment, csc2, int(np.sum(csc2 == 0)))
        if unsat2 == 0:
            return best_assignment
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
