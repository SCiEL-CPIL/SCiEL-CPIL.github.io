import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    # 330s hard kill; keep 15s safety margin
    deadline = time.time() + 315

    if m == 0:
        return [False] * n

    # Precompute variable occurrences as two parallel lists for speed
    var_occ_clauses = [[] for _ in range(n)]
    var_occ_pos = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occ_clauses[var].append(ci)
            var_occ_pos[var].append(lit > 0)

    # Build NumPy arrays for vectorized WalkSAT
    # clause_vars: (m, 3) array of variable indices (0-based)
    # clause_signs: (m, 3) array of +1/-1
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            clause_vars[ci, li] = abs(lit) - 1
            clause_signs[ci, li] = 1 if lit > 0 else -1

    # Build sparse incidence: for each variable, which clauses and signs
    # var_clause_idx[var] = np.array of clause indices
    # var_clause_sign[var] = np.array of signs (+1/-1)
    var_clause_idx = [None] * n
    var_clause_sign = [None] * n
    for v in range(n):
        idxs = var_occ_clauses[v]
        signs = [1 if p else -1 for p in var_occ_pos[v]]
        var_clause_idx[v] = np.array(idxs, dtype=np.int32) if idxs else np.array([], dtype=np.int32)
        var_clause_sign[v] = np.array(signs, dtype=np.int8) if signs else np.array([], dtype=np.int8)

    def count_unsatisfied(assignment):
        """Count unsatisfied clauses for a boolean list assignment."""
        sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            for li in range(3):
                v = cv[li]
                s = cs[li]
                if s == 1:
                    if assignment[v]:
                        sat_count[ci] += 1
                else:
                    if not assignment[v]:
                        sat_count[ci] += 1
        return int(np.sum(sat_count == 0))

    def count_unsatisfied_np(assignment_np):
        """Vectorized count of unsatisfied clauses. assignment_np: bool array of length n."""
        # For each clause, check if any literal is satisfied
        # satisfied[ci, li] = (assignment_np[clause_vars[ci,li]] == (clause_signs[ci,li] > 0))
        # Use broadcasting: gather assignment values
        vals = assignment_np[clause_vars]  # shape (m, 3)
        # clause_signs > 0 means positive literal
        pos_mask = clause_signs > 0  # (m, 3) bool
        # A literal is satisfied if: (positive and var is True) or (negative and var is False)
        lit_sat = np.where(pos_mask, vals, ~vals)  # (m, 3)
        clause_sat = np.any(lit_sat, axis=1)  # (m,)
        return int(np.sum(~clause_sat))

    def dpll_with_unit_prop(seed, time_budget=10):
        """DPLL with unit propagation and randomized restarts."""
        rng = random.Random(seed)
        start_time = time.time()

        def propagate(clauses_list, assignment):
            """Unit propagation. Returns (new_assignment, new_clauses_list, contradiction)"""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses_list:
                    sat = False
                    unit_lit = None
                    open_lits = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if var in assignment:
                            val = assignment[var]
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                        else:
                            open_lits.append(lit)

                    if sat:
                        continue
                    elif len(open_lits) == 0:
                        return assignment, clauses_list, True
                    elif len(open_lits) == 1:
                        unit_lit = open_lits[0]
                        var = abs(unit_lit) - 1
                        val = unit_lit > 0
                        if var in assignment:
                            if assignment[var] != val:
                                return assignment, clauses_list, True
                        else:
                            assignment[var] = val
                            changed = True
                        continue
                    else:
                        new_clauses.append(open_lits)

            return assignment, new_clauses, False

        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break

            assignment = {}
            num_initial = min(n // 5, 30)
            for var in rng.sample(range(n), num_initial):
                assignment[var] = rng.random() < 0.5

            clauses_copy = [list(c) for c in clauses]
            assignment, clauses_copy, contradiction = propagate(clauses_copy, assignment)

            if contradiction:
                continue

            if not clauses_copy:
                result = [False] * n
                for var in range(n):
                    result[var] = assignment.get(var, rng.random() < 0.5)
                return result

            result = [assignment.get(var, rng.random() < 0.5) for var in range(n)]
            unsat_count = count_unsatisfied(result)
            if unsat_count == 0:
                return result
            return result

        return [False] * n

    def vectorized_walksat(seed, init_assignment=None, noise=0.3, max_steps=2000000):
        """
        Vectorized WalkSAT with score-based clause selection.
        Uses NumPy to compute delta-satisfaction for all 3 variables in the chosen clause simultaneously.
        Score-based: clause score = number of unsatisfied literals (all 3 for a 3-SAT unsat clause).
        """
        rng = random.Random(seed)

        if init_assignment is not None:
            assignment = np.array(init_assignment, dtype=bool)
        else:
            assignment = np.array(rng.random(n) < 0.5, dtype=bool)

        # Compute initial satisfaction state
        # sat_count[ci] = number of satisfied literals in clause ci
        vals = assignment[clause_vars]  # (m, 3)
        pos_mask = clause_signs > 0
        lit_sat = np.where(pos_mask, vals, ~vals)
        sat_count = lit_sat.sum(axis=1)  # (m,)
        unsat_mask = sat_count == 0  # (m,) bool

        if not unsat_mask.any():
            return assignment.tolist()

        start_time = time.time()
        best_unsat = int(unsat_mask.sum())
        best_assignment = assignment.copy()
        stall_counter = 0
        prev_unsat = best_unsat

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if not unsat_mask.any():
                break

            current_unsat = int(unsat_mask.sum())
            if current_unsat >= prev_unsat:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_unsat = current_unsat

            # Adaptive noise
            adaptive_noise = noise + min(0.4, stall_counter * 0.005)

            # Score-based clause selection: pick unsat clause with max "pressure"
            # For 3-SAT, all unsat clauses have 0 satisfied literals, so use random
            # But we can weight by how "hard" the clause is (e.g., variable frequency)
            # Simple: random unsat clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = unsat_indices[rng.randrange(len(unsat_indices))]

            # Get the 3 variables and their target values for this clause
            cv = clause_vars[ci]  # (3,)
            cs = clause_signs[ci]  # (3,)
            # want_val for each var: True if positive literal
            want_vals = cs > 0  # (3,) bool

            # Compute gain/loss for each of the 3 variable flips using vectorized ops
            # For variable v in this clause, flipping to want_val:
            #   gain = number of OTHER clauses where this var's occurrence is satisfied by want_val AND currently unsat
            #   loss = number of OTHER clauses where this var's occurrence is NOT satisfied by want_val AND currently sat_count == 1
            # We compute this for all 3 candidates at once.

            scores = np.zeros(3, dtype=np.int32)
            for k in range(3):
                v = cv[k]
                want = want_vals[k]
                # Get all clauses where v appears
                v_clauses = var_clause_idx[v]
                v_signs = var_clause_sign[v]
                if len(v_clauses) == 0:
                    continue
                # For each occurrence, does setting v to want satisfy that literal?
                # want=True means we want var to be True; literal is satisfied if sign==1 (positive)
                # want=False means we want var to be False; literal is satisfied if sign==-1 (negative)
                lit_would_be_sat = (v_signs == (1 if want else -1))  # (num_occ,) bool

                # gain: clauses where lit_would_be_sat AND currently unsat (sat_count==0)
                # loss: clauses where NOT lit_would_be_sat AND currently sat_count==1
                # Exclude the current clause ci
                mask_not_ci = v_clauses != ci
                gain = int(np.sum(lit_would_be_sat & mask_not_ci & unsat_mask[v_clauses]))
                loss = int(np.sum(~lit_would_be_sat & mask_not_ci & (sat_count[v_clauses] == 1)))
                scores[k] = gain - loss

            # Choose which variable to flip
            if rng.random() < adaptive_noise:
                k = rng.randrange(3)
            else:
                k = int(np.argmax(scores))

            var = cv[k]
            want_val = bool(want_vals[k])

            if assignment[var] == want_val:
                continue

            # Perform the flip
            old_val = assignment[var]
            assignment[var] = want_val

            # Update sat_count using precomputed arrays
            v_clauses = var_clause_idx[var]
            v_signs = var_clause_sign[var]
            for idx in range(len(v_clauses)):
                oc = v_clauses[idx]
                s = v_signs[idx]
                # After flip, does this literal satisfy the clause?
                if (s == 1 and want_val) or (s == -1 and not want_val):
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_mask[oc] = False
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_mask[oc] = True

            if not unsat_mask.any():
                return assignment.tolist()

            # Track best
            current_unsat = int(unsat_mask.sum())
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()

        return best_assignment.tolist()

    def local_dpll_refine(assignment, time_budget=5.0):
        """Local DPLL refinement."""
        rng = random.Random(12345)
        start_time = time.time()

        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_indices = [i for i in range(m) if sat_count[i] == 0]

        if not unsat_indices:
            return assignment

        best_assignment = assignment[:]
        best_unsat = len(unsat_indices)

        max_rounds = 50
        for rnd in range(max_rounds):
            if time.time() - start_time > time_budget:
                break
            if not unsat_indices:
                break

            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clauses[ci]

            found_solution = False
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0

                if assignment[var] == want_val:
                    continue

                old_val = assignment[var]
                assignment[var] = want_val

                prop_assignment = assignment.copy()
                prop_sat_count = [0] * m
                for pi, pclause in enumerate(clauses):
                    for plit in pclause:
                        pvar = abs(plit) - 1
                        if prop_assignment[pvar] == (plit > 0):
                            prop_sat_count[pi] += 1

                contradiction = False
                changed = True
                while changed and not contradiction:
                    changed = False
                    for pi, pclause in enumerate(clauses):
                        if prop_sat_count[pi] > 0:
                            continue
                        open_lits = []
                        for plit in pclause:
                            pvar = abs(plit) - 1
                            if pvar not in prop_assignment:
                                open_lits.append(plit)
                        if len(open_lits) == 0:
                            contradiction = True
                            break
                        elif len(open_lits) == 1:
                            ulit = open_lits[0]
                            uvar = abs(ulit) - 1
                            uval = ulit > 0
                            if uvar in prop_assignment:
                                if prop_assignment[uvar] != uval:
                                    contradiction = True
                                    break
                            else:
                                prop_assignment[uvar] = uval
                                changed = True
                                for oi, oclause in enumerate(clauses):
                                    for olit in oclause:
                                        ovar = abs(olit) - 1
                                        if ovar == uvar:
                                            if (olit > 0 and uval) or (olit < 0 and not uval):
                                                prop_sat_count[oi] += 1
                                            else:
                                                prop_sat_count[oi] -= 1

                if contradiction:
                    assignment[var] = old_val
                    continue

                new_unsat = sum(1 for c in prop_sat_count if c == 0)
                if new_unsat < best_unsat:
                    best_unsat = new_unsat
                    for v in range(n):
                        if v not in prop_assignment:
                            prop_assignment[v] = rng.random() < 0.5
                    best_assignment = prop_assignment
                    found_solution = True
                    break
                else:
                    assignment[var] = old_val

            if not found_solution:
                continue

        return best_assignment

    def targeted_repair(assignment, max_rounds=1000):
        """Greedy min-conflict repair."""
        rng = random.Random(42)
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        if not unsat_set:
            return assignment

        best_unsat = len(unsat_set)
        best_assignment = assignment[:]

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if not unsat_set:
                break

            unsat_list = list(unsat_set)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[ci]

            best_score = -999
            best_var = -1
            best_val = False

            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0

                occ_clauses = var_occ_clauses[var]
                occ_pos = var_occ_pos[var]
                for idx in range(len(occ_clauses)):
                    oc = occ_clauses[idx]
                    if oc == ci:
                        continue
                    lit_val = occ_pos[idx]
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        if sat_count[oc] == 1:
                            loss += 1

                score = gain - loss
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_val = want_val

            if best_var < 0:
                break

            old_val = assignment[best_var]
            if old_val == best_val:
                continue

            assignment[best_var] = best_val

            occ_clauses = var_occ_clauses[best_var]
            occ_pos = var_occ_pos[best_var]
            for idx in range(len(occ_clauses)):
                oc = occ_clauses[idx]
                lit_val = occ_pos[idx]
                if lit_val == best_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                return assignment

            if len(unsat_set) < best_unsat:
                best_unsat = len(unsat_set)
                best_assignment = assignment[:]

        return best_assignment

    # Phase 1: DPLL
    result = dpll_with_unit_prop(0, time_budget=25)
    unsat_count = count_unsatisfied(result)
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 2: Vectorized WalkSAT from DPLL result
    if time.time() < deadline - 20:
        result = vectorized_walksat(0, init_assignment=best_assignment, noise=0.3, max_steps=2000000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2b: Local DPLL refinement
    if time.time() < deadline - 15 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=5.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2c: Targeted repair
    if time.time() < deadline - 12 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=5000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 3: More DPLL restarts
    for seed in range(1, 15):
        if time.time() > deadline - 10:
            break
        result = dpll_with_unit_prop(seed, time_budget=4)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            if time.time() < deadline - 8 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=3.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 4: Multiple Vectorized WalkSAT restarts
    for seed in range(1, 40):
        if time.time() > deadline - 5:
            break
        if seed % 3 == 0:
            result = vectorized_walksat(seed, init_assignment=best_assignment, noise=0.2, max_steps=1000000)
        else:
            result = vectorized_walksat(seed, init_assignment=None, noise=0.3, max_steps=1000000)

        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            if time.time() < deadline - 4 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=2.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 5: Final targeted repair
    if time.time() < deadline - 3 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=10000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5b: Final local DPLL
    if time.time() < deadline - 2 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=1.5)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 6: Final vectorized WalkSAT polish
    if time.time() < deadline - 1 and best_unsat > 0:
        result = vectorized_walksat(999, init_assignment=best_assignment, noise=0.1, max_steps=500000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
