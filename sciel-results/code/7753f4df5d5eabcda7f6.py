import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 325.0  # 5s margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [False] * n

    # Precompute clause structure
    clause_vars = []
    clause_signs = []

    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    # Build var_to_clauses: for each variable, list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            var_to_clauses[v].append((i, clause_signs[i][j]))

    # Precompute for each variable: numpy arrays of clause indices and signs
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        cidx = [ci for ci, s in var_to_clauses[v]]
        csign = [s for ci, s in var_to_clauses[v]]
        var_clause_idx.append(np.array(cidx, dtype=np.int32) if cidx else np.array([], dtype=np.int32))
        var_clause_sign.append(np.array(csign, dtype=bool) if csign else np.array([], dtype=bool))

    def init_state(assignment_arr):
        """Compute clause_sat_count from a numpy bool array assignment."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            vs = clause_vars[i]
            ss = clause_signs[i]
            count = 0
            for j in range(3):
                if assignment_arr[vs[j]] == ss[j]:
                    count += 1
            clause_sat_count[i] = count
        return clause_sat_count

    def flip_var(v, assignment_arr, clause_sat_count):
        """Flip variable v and update clause_sat_count using precomputed arrays."""
        old_val = assignment_arr[v]
        assignment_arr[v] = not old_val
        new_val = assignment_arr[v]
        cidx = var_clause_idx[v]
        csign = var_clause_sign[v]
        if len(cidx) > 0:
            matched = (csign == new_val)
            deltas = np.where(matched, 1, -1).astype(np.int32)
            np.add.at(clause_sat_count, cidx, deltas)

    def unit_propagate(assignment_arr, clause_sat_count):
        """Set variables in clauses with 0 satisfied literals."""
        changed = True
        max_rounds = 20
        rounds = 0
        while changed and rounds < max_rounds:
            changed = False
            rounds += 1
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                break
            unsat_indices = np.where(unsat_mask)[0]
            for i in unsat_indices:
                if clause_sat_count[i] != 0:
                    continue
                # Pick variable with least damage
                best_v = clause_vars[i][0]
                best_damage = float('inf')
                for v in clause_vars[i]:
                    damage = 0
                    for other_ci, sign in var_to_clauses[v]:
                        if other_ci != i and clause_sat_count[other_ci] > 0:
                            damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                sign = None
                for j, v in enumerate(clause_vars[i]):
                    if v == best_v:
                        sign = clause_signs[i][j]
                        break
                if assignment_arr[best_v] != sign:
                    flip_var(best_v, assignment_arr, clause_sat_count)
                    changed = True
        return True

    def pure_literal(assignment_arr, clause_sat_count):
        """Find and assign pure literals."""
        pos_count = np.zeros(n, dtype=np.int32)
        neg_count = np.zeros(n, dtype=np.int32)
        for i in range(m):
            for j, v in enumerate(clause_vars[i]):
                if clause_signs[i][j]:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1

        changed = False
        pure_pos = (pos_count > 0) & (neg_count == 0)
        for v in np.where(pure_pos)[0]:
            if not assignment_arr[v]:
                flip_var(int(v), assignment_arr, clause_sat_count)
                changed = True
        pure_neg = (neg_count > 0) & (pos_count == 0)
        for v in np.where(pure_neg)[0]:
            if assignment_arr[v]:
                flip_var(int(v), assignment_arr, clause_sat_count)
                changed = True
        return changed

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    def soft_min_select(vs, ci, assignment_arr, clause_sat_count, temp):
        """Use log-sum-exp soft-min to select variable from unsat clause ci.
        Smooth surrogate for hard min with epigraph-style penalty."""
        scores = []
        for v in vs:
            new_val = not assignment_arr[v]
            cidx = var_clause_idx[v]
            csign = var_clause_sign[v]

            if len(cidx) == 0:
                score = 0.0
            else:
                matched = (csign == new_val)
                current_sat = clause_sat_count[cidx] > 0

                newly_unsat = np.sum(current_sat & ~matched)
                newly_sat = np.sum(~current_sat & matched)

                # Epigraph-style penalty: add small penalty for clauses that are
                # "close" to becoming unsat (sat count == 1) to smooth the constraint
                close_to_unsat = np.sum((clause_sat_count[cidx] == 1) & ~matched)
                penalty = 0.1 * close_to_unsat

                score = newly_unsat - newly_sat + penalty

            scores.append(score)

        scores = np.array(scores, dtype=np.float64)

        # Soft-min via log-sum-exp: use softmax over -scores
        neg_scores = -scores
        max_neg = np.max(neg_scores)
        exp_scores = np.exp((neg_scores - max_neg) / temp)
        probs = exp_scores / np.sum(exp_scores)

        v_idx = int(np.argmax(probs))
        return vs[v_idx]

    while time.time() - start_time < timeout:
        # Random restart
        assignment_arr = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        clause_sat_count = init_state(assignment_arr)

        if np.all(clause_sat_count > 0):
            return assignment_arr.tolist()

        # Pure literal elimination
        pure_literal(assignment_arr, clause_sat_count)

        if np.all(clause_sat_count > 0):
            return assignment_arr.tolist()

        # Unit propagation
        unit_propagate(assignment_arr, clause_sat_count)

        if np.all(clause_sat_count > 0):
            return assignment_arr.tolist()

        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment_arr.copy()

        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 5.0:
            break

        # Adaptive parameters based on unsat count
        if current_unsat <= 3:
            max_flips = 15000
            noise = 0.15
            temp = 0.3  # tighter selection when close
        elif current_unsat <= 10:
            max_flips = 10000
            noise = 0.25
            temp = 0.4
        elif current_unsat <= 30:
            max_flips = 7000
            noise = 0.35
            temp = 0.5
        else:
            max_flips = 5000
            noise = 0.45
            temp = 0.6  # more exploration when far

        for flip_num in range(max_flips):
            if time.time() - start_time > timeout - 3.0:
                break

            # Find unsatisfied clauses
            unsat_mask = (clause_sat_count == 0)
            unsat_count = int(np.sum(unsat_mask))
            if unsat_count == 0:
                return assignment_arr.tolist()

            # Pick a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])

            # Get the 3 variables in this clause
            vs = clause_vars[ci]

            if rng.random() < noise:
                # Random flip
                v = vs[rng.randrange(3)]
            else:
                # Soft-min selection with epigraph penalty
                v = soft_min_select(vs, ci, assignment_arr, clause_sat_count, temp)

            flip_var(v, assignment_arr, clause_sat_count)

            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment_arr.tolist()

            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment_arr.copy()

            # Final polish pass: when very close to solution, run targeted high-precision search
            if current_unsat <= 5:
                polish_flips = 2000
                for pf in range(polish_flips):
                    if time.time() - start_time > timeout - 2.0:
                        break
                    unsat_mask = (clause_sat_count == 0)
                    unsat_count = int(np.sum(unsat_mask))
                    if unsat_count == 0:
                        return assignment_arr.tolist()

                    # In polish mode, use very low noise and tight soft-min
                    unsat_indices = np.where(unsat_mask)[0]
                    ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
                    vs = clause_vars[ci]

                    if rng.random() < 0.05:
                        v = vs[rng.randrange(3)]
                    else:
                        v = soft_min_select(vs, ci, assignment_arr, clause_sat_count, 0.2)

                    flip_var(v, assignment_arr, clause_sat_count)

                    current_unsat = int(np.sum(clause_sat_count == 0))
                    if current_unsat == 0:
                        return assignment_arr.tolist()

                    if current_unsat < best_unsat_count:
                        best_unsat_count = current_unsat
                        best_assignment = assignment_arr.copy()

                # After polish, if still not solved, continue outer loop
                if time.time() - start_time > timeout - 3.0:
                    break

    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
