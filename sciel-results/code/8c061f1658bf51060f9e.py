"""Improved planted 3-SAT solver with efficient WalkSAT + restarts + DLS + variable elimination."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: 330s wall-clock limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0

    # Preprocess: convert to 0-based variable indices and sign (+1/-1)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)

    # Build variable -> list of (clause_idx, sign) for fast updates
    var_occurrences = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for pi, (v, s) in enumerate(cv):
            var_occurrences[v].append((ci, s))

    # Clause -> variables as flat arrays for faster access
    clause_var_ids = []
    clause_signs = []
    for cv in clause_vars:
        clause_var_ids.append([x[0] for x in cv])
        clause_signs.append([x[1] for x in cv])

    rng = random.Random(42)

    # Variable elimination preprocessing: fix variables with strong bias
    # For each variable, count how many clauses would be satisfied if set True vs False
    # Fix variables where one assignment satisfies significantly more clauses
    fixed_assignments = {}
    remaining_vars = set(range(n))
    
    # Compute initial bias for each variable
    def compute_variable_bias(var_id):
        """Count how many clauses are satisfied by True vs False for this variable."""
        true_count = 0
        false_count = 0
        for (aci, s) in var_occurrences[var_id]:
            # If var is True, literal is satisfied if s == 1
            # If var is False, literal is satisfied if s == -1
            if s == 1:
                true_count += 1
            else:
                false_count += 1
        return true_count, false_count

    # Iteratively fix variables with strong bias
    max_elim_rounds = 20
    for round_num in range(max_elim_rounds):
        if time.time() > deadline - 300.0:
            break
        if not remaining_vars:
            break
        
        best_var = None
        best_bias_ratio = 0
        best_val = None
        
        for v in remaining_vars:
            tc, fc = compute_variable_bias(v)
            total = tc + fc
            if total == 0:
                continue
            # Bias ratio: how much more one value satisfies
            if tc > fc:
                ratio = (tc - fc) / total
                if ratio > best_bias_ratio:
                    best_bias_ratio = ratio
                    best_var = v
                    best_val = True
            elif fc > tc:
                ratio = (fc - tc) / total
                if ratio > best_bias_ratio:
                    best_bias_ratio = ratio
                    best_var = v
                    best_val = False
        
        # Only fix if bias is strong enough (>60% of clauses prefer one value)
        if best_var is not None and best_bias_ratio > 0.3:
            fixed_assignments[best_var] = best_val
            remaining_vars.discard(best_var)
        else:
            break

    # Apply fixed assignments and simplify
    if fixed_assignments:
        # Filter out satisfied clauses and simplify unsatisfied ones
        new_clause_var_ids = []
        new_clause_signs = []
        new_var_occurrences = [[] for _ in range(n)]
        
        for ci, (vlist, slist) in enumerate(zip(clause_var_ids, clause_signs)):
            clause_satisfied = False
            new_vlist = []
            new_slist = []
            for i in range(len(vlist)):
                v = vlist[i]
                s = slist[i]
                if v in fixed_assignments:
                    # Check if this literal is satisfied
                    if fixed_assignments[v] and s == 1:
                        clause_satisfied = True
                        break
                    elif (not fixed_assignments[v]) and s == -1:
                        clause_satisfied = True
                        break
                    else:
                        # Literal is false, skip it
                        continue
                else:
                    new_vlist.append(v)
                    new_slist.append(s)
            
            if clause_satisfied:
                continue  # Clause is satisfied, remove it
            
            if len(new_vlist) == 0:
                # Clause became empty (all literals false) - unsatisfiable
                # This shouldn't happen for planted SAT instances
                # But if it does, we need to handle it
                pass
            
            if len(new_vlist) > 0:
                new_clause_var_ids.append(new_vlist)
                new_clause_signs.append(new_slist)
                for pi, (v, s) in enumerate(zip(new_vlist, new_slist)):
                    new_var_occurrences[v].append((len(new_clause_var_ids) - 1, s))
        
        if len(new_clause_var_ids) < m:
            # Update to simplified instance
            m = len(new_clause_var_ids)
            clause_var_ids = new_clause_var_ids
            clause_signs = new_clause_signs
            var_occurrences = new_var_occurrences

    def full_check(assignment):
        """Check if all clauses are satisfied."""
        for ci in range(m):
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            sat = False
            for i in range(len(vlist)):
                v = vlist[i]
                s = slist[i]
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    sat = True
                    break
            if not sat:
                return False, ci
        return True, -1

    def init_clause_sat_count(assignment):
        """Initialize clause_sat_count."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            cnt = 0
            for i in range(len(vlist)):
                v = vlist[i]
                s = slist[i]
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            clause_sat_count[ci] = cnt
        return clause_sat_count

    def make_full_assignment(partial_assignment, clause_sat_count):
        """Create a full assignment from partial, applying fixed assignments."""
        full_assignment = np.array([False] * n, dtype=bool)
        # Apply fixed assignments
        for v, val in fixed_assignments.items():
            full_assignment[v] = val
        # Apply partial assignment for remaining variables
        for v in remaining_vars:
            if v in partial_assignment:
                full_assignment[v] = partial_assignment[v]
            else:
                full_assignment[v] = rng.random() < 0.5
        return full_assignment

    def pick_unsat_clause(clause_sat_count, rng):
        """Pick a random unsatisfied clause efficiently."""
        unsat_indices = np.where(clause_sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return -1
        return int(unsat_indices[rng.randint(0, len(unsat_indices) - 1)])

    def compute_delta_unsat(v, assignment, clause_sat_count):
        """Compute change in unsat count if we flip variable v.
        Flipping v toggles the truth value of each literal containing v.
        For each clause containing v:
          - If literal was TRUE and becomes FALSE: if clause had exactly 1 true literal, it becomes unsat (+1)
          - If literal was FALSE and becomes TRUE: if clause had 0 true literals, it becomes sat (-1)
        """
        delta = 0
        v_val = assignment[v]
        for (aci, s) in var_occurrences[v]:
            # Current truth of this literal
            lit_true = (v_val and s == 1) or ((not v_val) and s == -1)
            # After flip, truth is inverted
            if lit_true:
                # Was true, will be false
                if clause_sat_count[aci] == 1:
                    delta += 1
            else:
                # Was false, will be true
                if clause_sat_count[aci] == 0:
                    delta -= 1
        return delta

    def do_flip(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count."""
        old_val = assignment[v]
        assignment[v] = not old_val
        for (aci, s) in var_occurrences[v]:
            # Before flip
            was_sat = (old_val and s == 1) or ((not old_val) and s == -1)
            # After flip
            is_sat = (assignment[v] and s == 1) or ((not assignment[v]) and s == -1)
            if was_sat and not is_sat:
                clause_sat_count[aci] -= 1
            elif (not was_sat) and is_sat:
                clause_sat_count[aci] += 1

    # DPLL with unit propagation for preprocessing
    def dpll_preprocess():
        """Try to simplify via unit propagation. Returns (fixed_assignments, simplified_clauses) or None if conflict."""
        # This is a lightweight pre-pass; skip for speed if it gets complex
        return None

    # Main solver loop with restarts
    best_assignment = None
    best_unsat_count = m + 1

    # Adaptive parameters
    base_steps = 5000
    max_restarts = 5000
    p_good = 0.6  # probability to pick a literal that would satisfy the clause

    # Final polish: try to fix remaining unsat clauses by flipping variables
    def polish(assignment, clause_sat_count, time_limit):
        """Final polish pass: try to satisfy remaining clauses via targeted flips."""
        unsat_count = int(np.sum(clause_sat_count == 0))
        if unsat_count == 0:
            return assignment, clause_sat_count, 0
        t_start = time.time()
        stagnant = 0
        for _ in range(100000):
            if time.time() > time_limit:
                break
            if unsat_count == 0:
                break
            # Find unsat clauses
            unsat_idx = np.where(clause_sat_count == 0)[0]
            if len(unsat_idx) == 0:
                break
            ci = int(unsat_idx[rng.randint(0, len(unsat_idx) - 1)])
            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            # Try each variable in this clause, pick the one with best delta
            best_v = -1
            best_delta = 999
            for v in vlist:
                if v in fixed_assignments:
                    continue
                d = compute_delta_unsat(v, assignment, clause_sat_count)
                if d < best_delta:
                    best_delta = d
                    best_v = v
            if best_v >= 0 and best_delta <= 0:
                do_flip(best_v, assignment, clause_sat_count)
                old_unsat = unsat_count
                unsat_count += best_delta
                if unsat_count < old_unsat:
                    stagnant = 0
                else:
                    stagnant += 1
            elif best_v >= 0 and rng.random() < 0.2:
                do_flip(best_v, assignment, clause_sat_count)
                unsat_count += best_delta
                stagnant += 1
            else:
                stagnant += 1
                if stagnant > 50:
                    break
        return assignment, clause_sat_count, unsat_count

    # Variable activity for DLS-like heuristic
    var_activity = np.zeros(n, dtype=np.float64)
    activity_decay = 0.995

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Initial assignment: use fixed assignments + random for rest
        assignment = np.array([False] * n, dtype=bool)
        for v, val in fixed_assignments.items():
            assignment[v] = val
        for v in remaining_vars:
            assignment[v] = rng.random() < 0.5
        
        clause_sat_count = init_clause_sat_count(assignment)
        unsat_count = int(np.sum(clause_sat_count == 0))

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Adaptive step count: more steps when close to solution
        if unsat_count <= 5:
            steps = base_steps * 3
        elif unsat_count <= 20:
            steps = base_steps * 2
        else:
            steps = base_steps

        # WalkSAT loop
        stagnant = 0
        for step in range(steps):
            if time.time() > deadline - 1.0:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Pick an unsatisfied clause
            ci = pick_unsat_clause(clause_sat_count, rng)
            if ci < 0:
                return assignment.tolist()

            vlist = clause_var_ids[ci]
            slist = clause_signs[ci]
            clen = len(vlist)

            # Filter out fixed variables from candidates
            flist = [v for v in vlist if v not in fixed_assignments]
            if not flist:
                # All variables in this clause are fixed - should not happen for SAT
                break

            # Choose variable: with prob p_good, pick one that would satisfy this clause
            if rng.random() < p_good:
                # Find literals in this clause that are currently unsatisfied
                candidates = []
                for v in flist:
                    s = slist[vlist.index(v)]
                    is_lit_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
                    if not is_lit_sat:
                        candidates.append(v)
                if candidates:
                    # Weight by activity
                    if len(candidates) > 1:
                        weights = [var_activity[v] + 1.0 for v in candidates]
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cum = 0
                        v = candidates[-1]
                        for j, cv in enumerate(candidates):
                            cum += weights[j]
                            if r <= cum:
                                v = cv
                                break
                    else:
                        v = candidates[0]
                else:
                    # All non-fixed literals satisfied? Shouldn't happen for unsat clause
                    v = flist[rng.randint(0, len(flist) - 1)]
            else:
                # Pick based on activity-weighted random from clause
                if len(flist) > 1:
                    weights = [var_activity[v] + 1.0 for v in flist]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    v = flist[-1]
                    for i, fv in enumerate(flist):
                        cum += weights[i]
                        if r <= cum:
                            v = fv
                            break
                else:
                    v = flist[0]

            # Compute delta
            delta = compute_delta_unsat(v, assignment, clause_sat_count)

            # Acceptance: always accept improving, sometimes accept worsening
            if delta <= 0 or (delta > 0 and rng.random() < 0.25):
                do_flip(v, assignment, clause_sat_count)
                new_unsat = unsat_count + delta
                if new_unsat < unsat_count:
                    stagnant = 0
                else:
                    stagnant += 1
                unsat_count = new_unsat

                # Update activity
                for i in range(clen):
                    var_activity[vlist[i]] *= activity_decay
                    var_activity[vlist[i]] += 1.0

                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment.copy()
            else:
                stagnant += 1

            # Early restart if very stagnant
            if stagnant > 300 and unsat_count > 10:
                break

        # Decay activity between restarts
        var_activity *= 0.95

        if time.time() > deadline - 1.0:
            break

    # If we have a best assignment, try to polish it
    if best_assignment is not None and time.time() < deadline - 5.0:
        # Reconstruct clause_sat_count for best_assignment
        best_clause_sat = init_clause_sat_count(best_assignment)
        best_unsat = int(np.sum(best_clause_sat == 0))
        if best_unsat > 0 and best_unsat <= 30:
            polished, _, _ = polish(best_assignment, best_clause_sat, deadline - 2.0)
            final_unsat = int(np.sum(best_clause_sat == 0))
            if final_unsat == 0:
                return best_assignment.tolist()
            # Try polish on current best too
            if final_unsat < best_unsat:
                best_assignment = polished
                best_unsat = final_unsat

    # Final high-precision polish: try multiple random restarts of local search
    if best_assignment is not None and time.time() < deadline - 3.0:
        for attempt in range(10):
            if time.time() > deadline - 1.0:
                break
            # Start from best_assignment with small random perturbations
            perturbed = best_assignment.copy()
            # Flip 1-3 random non-fixed variables
            num_flips = rng.randint(1, 3)
            for _ in range(num_flips):
                v = rng.choice(list(remaining_vars))
                perturbed[v] = not perturbed[v]
            
            psat = init_clause_sat_count(perturbed)
            punsat = int(np.sum(psat == 0))
            if punsat < best_unsat:
                best_assignment = perturbed
                best_unsat = punsat
            
            if punsat <= 5:
                polished, _, _ = polish(perturbed, psat, deadline - 1.0)
                final_unsat = int(np.sum(psat == 0))
                if final_unsat == 0:
                    return perturbed.tolist()
                if final_unsat < best_unsat:
                    best_assignment = polished
                    best_unsat = final_unsat

    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
# EVOLVE-BLOCK-END
