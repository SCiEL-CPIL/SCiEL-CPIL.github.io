"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0

    # Pre-compute clause structure for fast updates
    # For each clause, store (var_idx, is_positive) pairs
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))

    # State: assignment array, clause satisfaction tracking
    # For each clause, track:
    #   sat_count[ci] = number of literals satisfied in clause ci
    #   false_vars[ci] = set of var indices whose literal is False in clause ci

    def init_state(assignment):
        sat_count = [0] * m
        false_vars = [set() for _ in range(m)]
        for ci, cv in enumerate(clause_vars):
            for vi, pos in cv:
                if assignment[vi] == pos:
                    sat_count[ci] += 1
                else:
                    false_vars[ci].add(vi)
        return sat_count, false_vars

    def count_unsat(sat_count):
        cnt = 0
        for sc in sat_count:
            if sc == 0:
                cnt += 1
        return cnt

    def find_unsat_clause(sat_count):
        for ci in range(m):
            if sat_count[ci] == 0:
                return ci
        return -1

    def flip_variable(vi, assignment, sat_count, false_vars):
        """Flip variable vi and update state. Returns number of newly unsatisfied clauses."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val

        delta = 0  # net change in unsat count
        for (ci, pos) in var_clauses[vi]:
            if old_val == pos:
                # Was satisfied, now unsatisfied
                sat_count[ci] -= 1
                false_vars[ci].add(vi)
                if sat_count[ci] == 0:
                    delta += 1
            else:
                # Was unsatisfied, now satisfied
                sat_count[ci] += 1
                false_vars[ci].discard(vi)
                if sat_count[ci] == 1:
                    delta -= 1
        return delta

    def unit_propagate(assignment, sat_count, false_vars):
        """Apply unit propagation. Returns (success, new_unsat_count)."""
        changed = True
        while changed:
            changed = False
            for ci in range(m):
                if sat_count[ci] == 1:
                    # Exactly one literal is false; set it to true
                    fv = list(false_vars[ci])
                    if len(fv) == 1:
                        vi = fv[0]
                        pos = None
                        for (v, p) in clause_vars[ci]:
                            if v == vi:
                                pos = p
                                break
                        if assignment[vi] != pos:
                            # Flip to satisfy
                            assignment[vi] = pos
                            for (ci2, pos2) in var_clauses[vi]:
                                if assignment[vi] == pos2:
                                    sat_count[ci2] += 1
                                    false_vars[ci2].discard(vi)
                                else:
                                    sat_count[ci2] -= 1
                                    false_vars[ci2].add(vi)
                            changed = True
                            break
                elif sat_count[ci] == 0:
                    # Clause is unsatisfied - check if all literals are false
                    # This means we need to set one of them to true
                    # But in unit propagation context, this is a conflict
                    pass
        return True, count_unsat(sat_count)

    def pure_literal_eliminate(assignment, sat_count, false_vars):
        """Identify and set pure literals. Returns (success, new_unsat_count)."""
        # Count positive and negative occurrences for each variable
        pos_count = [0] * n
        neg_count = [0] * n
        for ci in range(m):
            if sat_count[ci] > 0:
                continue  # Skip satisfied clauses
            for vi, pos in clause_vars[ci]:
                if pos:
                    pos_count[vi] += 1
                else:
                    neg_count[vi] += 1
        
        changed = True
        while changed:
            changed = False
            for vi in range(n):
                if pos_count[vi] > 0 and neg_count[vi] == 0:
                    # Variable only appears positively in unsat clauses
                    if not assignment[vi]:
                        assignment[vi] = True
                        for (ci, pos) in var_clauses[vi]:
                            if assignment[vi] == pos:
                                sat_count[ci] += 1
                                false_vars[ci].discard(vi)
                            else:
                                sat_count[ci] -= 1
                                false_vars[ci].add(vi)
                        # Update counts
                        for (ci, pos) in var_clauses[vi]:
                            if pos:
                                pos_count[vi] -= 1
                            else:
                                neg_count[vi] -= 1
                        changed = True
                elif neg_count[vi] > 0 and pos_count[vi] == 0:
                    # Variable only appears negatively in unsat clauses
                    if assignment[vi]:
                        assignment[vi] = False
                        for (ci, pos) in var_clauses[vi]:
                            if assignment[vi] == pos:
                                sat_count[ci] += 1
                                false_vars[ci].discard(vi)
                            else:
                                sat_count[ci] -= 1
                                false_vars[ci].add(vi)
                        # Update counts
                        for (ci, pos) in var_clauses[vi]:
                            if pos:
                                pos_count[vi] -= 1
                            else:
                                neg_count[vi] -= 1
                        changed = True
        return True, count_unsat(sat_count)

    # WalkSAT with multiple restarts and adaptive strategy
    rng = random.Random(0)

    # Weight parameter: probability of flipping a literal in the chosen unsat clause
    # vs. flipping a random variable that appears in unsat clauses
    w = 0.4
    w_min = 0.0
    w_max = 1.0
    w_decay = 0.98

    best_unsat = m + 1
    best_assignment = [False] * n

    max_restarts = 200
    base_steps = 5000

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Initialize with random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count, false_vars = init_state(assignment)
        cur_unsat = count_unsat(sat_count)

        if cur_unsat == 0:
            return assignment

        # Apply preprocessing
        success, cur_unsat = pure_literal_eliminate(assignment, sat_count, false_vars)
        if cur_unsat == 0:
            return assignment

        success, cur_unsat = unit_propagate(assignment, sat_count, false_vars)
        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Adaptive step budget: more steps if we're making progress
        # Scale steps based on restart number and current difficulty
        steps = base_steps + restart * 2000

        for step in range(steps):
            if time.time() > deadline - 2.0:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Find an unsatisfied clause
            ci = find_unsat_clause(sat_count)
            if ci == -1:
                return assignment

            # Choose which variable to flip using weighted selection
            # Prefer variables that appear in more unsat clauses
            cv = clause_vars[ci]
            weights = []
            for vi, pos in cv:
                # Count how many unsat clauses this variable appears in
                weight = 1
                for (ci2, pos2) in var_clauses[vi]:
                    if sat_count[ci2] == 0:
                        weight += 1
                weights.append(weight)
            
            # Weighted random selection
            total_weight = sum(weights)
            r = rng.random() * total_weight
            cumulative = 0
            vi = cv[0][0]
            for i, (w_val, (v, p)) in enumerate(zip(weights, cv)):
                cumulative += w_val
                if r <= cumulative:
                    vi = v
                    break

            # Try the flip
            old_cur_unsat = cur_unsat
            new_delta = flip_variable(vi, assignment, sat_count, false_vars)
            cur_unsat += new_delta

            # WalkSAT-style: if we increased unsat, consider reverting
            # but with some probability keep it to escape local minima
            if new_delta > 0:
                # With some probability, revert
                if rng.random() > 0.5:
                    flip_variable(vi, assignment, sat_count, false_vars)
                    cur_unsat = old_cur_unsat

            # Periodically apply unit propagation
            if step % 100 == 0:
                success, cur_unsat = unit_propagate(assignment, sat_count, false_vars)
                if cur_unsat == 0:
                    return assignment

        # Decay w to explore more broadly as we continue
        w = max(w_min, min(w_max, w * w_decay))

    # Return best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
