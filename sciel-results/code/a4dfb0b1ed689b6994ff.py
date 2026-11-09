"""Synthetic planted 3-SAT solver (50 variables) with robust local search."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: finish with a safety margin before 300s (evaluator kills at 330s)
    deadline = time.time() + 280.0

    # Preprocess: remove tautologies and unit clauses, propagate forced assignments
    # Represent literals as signed ints (1-based)
    # assignment: list of bools, index 0..n-1
    # forced: dict var_index -> bool
    forced = {}
    remaining_clauses = []
    unit_vars = set()

    # First pass: handle tautologies and collect unit clauses
    for c in clauses:
        # Check for tautology (both x and -x in clause)
        pos = set()
        neg = set()
        for l in c:
            if l > 0:
                pos.add(l)
            else:
                neg.add(-l)
        if pos & neg:
            continue  # tautology, skip
        # Unit clause
        if len(c) == 1:
            v = abs(c[0]) - 1
            val = c[0] > 0
            if v in forced:
                if forced[v] != val:
                    # Contradiction: instance is unsatisfiable (shouldn't happen for planted)
                    # Return arbitrary assignment
                    return [False] * n
            else:
                forced[v] = val
                unit_vars.add(v)
        else:
            remaining_clauses.append(c)

    # Propagate forced assignments
    def is_sat_clause(clause, assignment, forced_map):
        for l in clause:
            v = abs(l) - 1
            if v in forced_map:
                if forced_map[v] == (l > 0):
                    return True
            elif v in assignment:
                if assignment[v] == (l > 0):
                    return True
        return False

    # Iteratively simplify
    changed = True
    while changed:
        changed = False
        new_remaining = []
        for c in remaining_clauses:
            # Remove literals that are already satisfied by forced
            new_c = []
            satisfied = False
            for l in c:
                v = abs(l) - 1
                if v in forced:
                    if forced[v] == (l > 0):
                        satisfied = True
                        break
                    # else: literal is false, remove it
                else:
                    new_c.append(l)
            if satisfied:
                changed = True
                continue
            if not new_c:
                # Clause became empty -> unsatisfiable
                return [False] * n
            if len(new_c) == 1:
                v = abs(new_c[0]) - 1
                val = new_c[0] > 0
                if v in forced:
                    if forced[v] != val:
                        return [False] * n
                else:
                    forced[v] = val
                    changed = True
            else:
                new_remaining.append(new_c)
        remaining_clauses = new_remaining

    # Build initial assignment from forced + random for rest
    assignment = [None] * n
    for v in range(n):
        if v in forced:
            assignment[v] = forced[v]
        else:
            assignment[v] = False  # will be randomized in search

    # Count remaining clauses for fast satisfaction checking
    # Use a vectorized approach with numpy for speed
    import numpy as np

    # Build literal index arrays for remaining clauses
    # For each clause, store (var_index, sign) where sign is +1 or -1
    clause_vars = []
    clause_signs = []
    for c in remaining_clauses:
        vars_list = []
        signs_list = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            vars_list.append(v)
            signs_list.append(s)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    # If no remaining clauses, we're done
    if not remaining_clauses:
        return [bool(a) for a in assignment]

    # WalkSAT with restarts and adaptive parameters
    best_assignment = None
    best_unsat_count = m + 1

    def get_unsat_count(assign):
        """Count unsatisfied clauses."""
        count = 0
        for i, c in enumerate(remaining_clauses):
            sat = False
            for j, l in enumerate(c):
                v = abs(l) - 1
                if assign[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Use numpy for faster evaluation
    def np_unsat_count(assign_np):
        """assign_np: numpy array of bools, index 0..n-1"""
        total = 0
        for c in remaining_clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assign_np[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                total += 1
        return total

    # Precompute for numpy: array of (var, sign) for each clause
    # Actually, let's use a different approach: for each clause, precompute the literal indices
    # and signs, then check satisfaction in a loop (fast enough for 50 vars)

    def evaluate(assign):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for i, c in enumerate(remaining_clauses):
            sat = False
            for l in c:
                v = abs(l) - 1
                if assign[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat

    # Multiple restarts with different strategies
    max_restarts = 50
    steps_per_restart = 5000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        if restart == 0:
            # Use forced assignments, randomize the rest
            for v in range(n):
                if v not in forced:
                    assignment[v] = (restart * 7 + v) % 3 == 0  # deterministic but varied
        else:
            # Random restart with bias toward previous best
            for v in range(n):
                if v in forced:
                    assignment[v] = forced[v]
                else:
                    # 70% random, 30% keep previous best
                    if best_assignment is not None and random.random() < 0.3:
                        assignment[v] = best_assignment[v]
                    else:
                        assignment[v] = random.random() < 0.5

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            unsat = evaluate(assignment)
            if not unsat:
                return [bool(a) for a in assignment]

            # Track best
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            ci = random.choice(unsat)
            clause = remaining_clauses[ci]

            # Choose literal to flip
            # Weighted by how many other unsat clauses would be satisfied
            best_var = None
            best_score = -1
            best_min_unsat_after = m + 1

            for l in clause:
                v = abs(l) - 1
                # If this literal is currently false, flipping might help
                currently_true = assignment[v] == (l > 0)
                if currently_true:
                    # Flipping would make this clause worse (lose a true literal)
                    continue

                # Count how many unsat clauses contain this variable
                # and would be affected by flipping
                gain = 0
                loss = 0
                for ui in unsat:
                    uc = remaining_clauses[ui]
                    for ul in uc:
                        if abs(ul) - 1 == v:
                            if ul > 0:
                                # This literal is currently false (since clause is unsat)
                                # Flipping v to true would satisfy this clause
                                gain += 1
                            else:
                                # Literal -v is currently false means v is true
                                # But we said currently_true is false, so v is false
                                # So -v is true, which contradicts clause being unsat
                                # This case shouldn't happen if logic is correct
                                pass
                # Simpler: just pick randomly with some weighting
                # Use a simple heuristic: prefer variables in more unsat clauses
                var_unsat_count = sum(
                    1 for ui in unsat
                    if any(abs(ul) - 1 == v for ul in remaining_clauses[ui])
                )
                if var_unsat_count > best_score:
                    best_score = var_unsat_count
                    best_var = v

            if best_var is None:
                # All literals in clause are true? Shouldn't happen for unsat clause
                # Pick random literal
                l = random.choice(clause)
                best_var = abs(l) - 1

            # Flip
            assignment[best_var] = not assignment[best_var]

        # End of restart
        if time.time() > deadline:
            break

    # Return best found or current
    if best_assignment is not None:
        return [bool(a) for a in best_assignment]
    return [bool(a) for a in assignment]
# EVOLVE-BLOCK-END
