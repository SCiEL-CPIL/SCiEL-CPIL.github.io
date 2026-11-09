"""Improved planted 3-SAT solver with efficient WalkSAT + restarts + DLS."""
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

    def pick_unsat_clause(clause_sat_count, rng):
        """Pick a random unsatisfied clause efficiently."""
        unsat_indices = np.where(clause_sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return -1
        return int(unsat_indices[rng.randint(0, len(unsat_indices) - 1)])

    def compute_delta_unsat(v, assignment, clause_sat_count):
        """Compute change in unsat count if we flip variable v."""
        delta = 0
        for (aci, s) in var_occurrences[v]:
            old_val = assignment[v]
            was_sat = (old_val and s == 1) or (not old_val and s == -1)
            is_sat = (not old_val and s == 1) or (old_val and s == -1)
            if was_sat and not is_sat:
                if clause_sat_count[aci] == 1:
                    delta += 1
            elif not was_sat and is_sat:
                if clause_sat_count[aci] == 0:
                    delta -= 1
        return delta

    def do_flip(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count."""
        assignment[v] = not assignment[v]
        for (aci, s) in var_occurrences[v]:
            old_val = not assignment[v]
            was_sat = (old_val and s == 1) or (not old_val and s == -1)
            is_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
            if was_sat and not is_sat:
                clause_sat_count[aci] -= 1
            elif not was_sat and is_sat:
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
    base_steps = 3000
    max_restarts = 10000
    p_good = 0.6  # probability to pick a literal that would satisfy the clause

    # Variable activity for DLS-like heuristic
    var_activity = np.zeros(n, dtype=np.float64)
    activity_decay = 0.995

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
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

            # Choose variable: with prob p_good, pick one that would satisfy this clause
            if rng.random() < p_good:
                # Find literals in this clause that are currently unsatisfied
                candidates = []
                for i in range(clen):
                    v = vlist[i]
                    s = slist[i]
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
                    # All literals satisfied? Shouldn't happen for unsat clause
                    v = vlist[rng.randint(0, clen - 1)]
            else:
                # Pick based on activity-weighted random from clause
                if clen > 1:
                    weights = [var_activity[vlist[i]] + 1.0 for i in range(clen)]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    v = vlist[-1]
                    for i in range(clen):
                        cum += weights[i]
                        if r <= cum:
                            v = vlist[i]
                            break
                else:
                    v = vlist[0]

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
            if stagnant > 200 and unsat_count > 10:
                break

        # Decay activity between restarts
        var_activity *= 0.95

        if time.time() > deadline - 1.0:
            break

    # Return best found
    if best_assignment is not None:
        ok, _ = full_check(best_assignment)
        if ok:
            return best_assignment.tolist()
        return best_assignment.tolist()
    else:
        return [False] * n
# EVOLVE-BLOCK-END
