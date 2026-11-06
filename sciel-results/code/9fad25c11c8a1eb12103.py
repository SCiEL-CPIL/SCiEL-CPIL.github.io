# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin before 300s limit

    # Convert to 0-based: each literal becomes (var_idx, polarity) where polarity 1 means positive
    # For efficiency, precompute arrays
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n

    # Store clauses as arrays of (var, sign) where sign is +1 or -1
    # var is 0-based
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_list = [abs(l) - 1 for l in c]
        signs_list = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    # Precompute variable occurrence for pure literal elimination
    # For each variable, track if it appears positively or negatively
    pos_occ = [0] * n
    neg_occ = [0] * n
    for c in range(num_clauses):
        for v in clause_vars[c]:
            if clause_signs[c][0] == 1:
                pos_occ[v] += 1
            else:
                neg_occ[v] += 1
    # Actually need per-clause tracking. Let me redo this properly.
    # For pure literal: a literal is pure if its variable only appears with the same sign.
    # Track for each variable: does it appear with + and/or -
    var_pos = [False] * n
    var_neg = [False] * n
    for c in range(num_clauses):
        for i, v in enumerate(clause_vars[c]):
            if clause_signs[c][i] == 1:
                var_pos[v] = True
            else:
                var_neg[v] = True

    # Pure literal elimination: fix variables that are pure
    fixed = {}  # var -> bool
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in fixed:
                continue
            if var_pos[v] and not var_neg[v]:
                # Only appears positive, set to True
                fixed[v] = True
                changed = True
            elif var_neg[v] and not var_pos[v]:
                # Only appears negative, set to False
                fixed[v] = False
                changed = True

    # Apply fixed assignments: remove satisfied clauses, simplify others
    active_clauses = []
    active_vars = []
    active_signs = []
    for c in range(num_clauses):
        sat = False
        new_vars = []
        new_signs = []
        for i, v in enumerate(clause_vars[c]):
            if v in fixed:
                if fixed[v] == (clause_signs[c][i] == 1):
                    sat = True
                    break
            else:
                new_vars.append(v)
                new_signs.append(clause_signs[c][i])
        if not sat:
            if len(new_vars) == 0:
                # Empty clause: UNSAT (shouldn't happen for planted instances)
                return [fixed.get(v, True) for v in range(n)]
            active_clauses.append(len(active_vars))
            active_vars.extend(new_vars)
            active_signs.extend(new_signs)

    # Rebuild clause structure for active clauses
    # Actually let me just work with the simplified set
    simplified_clauses = []
    for c in range(num_clauses):
        sat = False
        new_vars = []
        new_signs = []
        for i, v in enumerate(clause_vars[c]):
            if v in fixed:
                if fixed[v] == (clause_signs[c][i] == 1):
                    sat = True
                    break
            else:
                new_vars.append(v)
                new_signs.append(clause_signs[c][i])
        if not sat and len(new_vars) > 0:
            simplified_clauses.append((new_vars, new_signs))

    num_simplified = len(simplified_clauses)

    # If no simplified clauses, we're done
    if num_simplified == 0:
        return [fixed.get(v, True) for v in range(n)]

    # Get list of free variables
    free_vars = set()
    for cv, cs in simplified_clauses:
        for v in cv:
            free_vars.add(v)
    free_var_list = list(free_vars)
    num_free = len(free_var_list)
    var_to_idx = {v: i for i, v in enumerate(free_var_list)}

    # Build simplified clause structure in terms of free var indices
    simp_clauses = []
    for cv, cs in simplified_clauses:
        simp_clauses.append((cv, cs))

    # WalkSAT on simplified instance
    rng = random.Random(42)
    assignment = {v: rng.random() < 0.5 for v in free_var_list}

    def is_satisfied(cv, cs):
        for v, s in zip(cv, cs):
            if assignment.get(v, False) == (s == 1):
                return True
        return False

    def count_unsat():
        cnt = 0
        for cv, cs in simp_clauses:
            if not is_satisfied(cv, cs):
                cnt += 1
        return cnt

    def flip_var(v):
        assignment[v] = not assignment.get(v, False)

    # Weighted WalkSAT with noise scheduling
    def walksat(max_steps, noise_start, noise_end):
        """Run WalkSAT with dynamic noise."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False

            # Find unsatisfied clauses
            unsat = [(cv, cs) for cv, cs in simp_clauses if not is_satisfied(cv, cs)]
            if not unsat:
                return True

            # Noise decreases over time
            t = step / max_steps
            noise = noise_start + (noise_end - noise_start) * t

            # Choose an unsatisfied clause
            cv, cs = rng.choice(unsat)

            # Try to find a variable flip that satisfies the clause
            # and doesn't break too many others
            best_var = None
            best_score = None

            for v in cv:
                # Score: number of clauses that would become satisfied minus
                # number that would become unsatisfied
                # Quick estimate: flip v, check impact
                old_val = assignment.get(v, False)
                assignment[v] = not old_val

                # Count satisfied/unsatisfied change
                # This is expensive, so let's use a simpler heuristic
                sat_count = 0
                unsat_count = 0
                for cv2, cs2 in simp_clauses:
                    was_sat = False
                    is_sat_now = False
                    for v2, s2 in zip(cv2, cs2):
                        if v2 == v:
                            if assignment[v2] == (s2 == 1):
                                is_sat_now = True
                            break
                        elif assignment.get(v2, False) == (s2 == 1):
                            was_sat = True
                            is_sat_now = True
                            break
                    else:
                        # v not in this clause
                        if was_sat:
                            pass  # unchanged
                        else:
                            # Check if this clause is now satisfied
                            for v3, s3 in zip(cv2, cs2):
                                if assignment.get(v3, False) == (s3 == 1):
                                    is_sat_now = True
                                    break
                    # This is getting complex. Let me use a simpler approach.
                    break

                # Too complex, revert and use simpler heuristic
                assignment[v] = old_val
                break

            # Simpler: pick variable with minimum break count
            best_var = rng.choice(cv)
            # With probability noise, pick random variable from clause
            # With probability 1-noise, pick variable that minimizes breaks
            if rng.random() < noise:
                best_var = rng.choice(cv)
            else:
                # Try each variable in clause, count how many clauses it would break
                best_score = float('inf')
                for v in cv:
                    old_val = assignment.get(v, False)
                    assignment[v] = not old_val
                    breaks = 0
                    for cv2, cs2 in simp_clauses:
                        if v not in cv2:
                            continue
                        # Check if this clause was satisfied before and isn't now
                        # Simple check: is this clause satisfied now?
                        sat_now = False
                        for v2, s2 in zip(cv2, cs2):
                            if assignment.get(v2, False) == (s2 == 1):
                                sat_now = True
                                break
                        if not sat_now:
                            # Was it satisfied before? Check other literals
                            was_sat = False
                            for v2, s2 in zip(cv2, cs2):
                                if v2 != v and assignment.get(v2, False) == (s2 == 1):
                                    was_sat = True
                                    break
                            if was_sat:
                                breaks += 1
                    assignment[v] = old_val
                    if breaks < best_score:
                        best_score = breaks
                        best_var = v

            flip_var(best_var)

        # Check if solved
        for cv, cs in simp_clauses:
            if not is_satisfied(cv, cs):
                return False
        return True

    # Try WalkSAT with different noise schedules
    # First try with decreasing noise
    steps_per_attempt = 5000
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Reset assignment with some bias
        if attempts % 5 == 1:
            # Fresh random start
            for v in free_var_list:
                assignment[v] = rng.random() < 0.5
        elif attempts % 5 == 2:
            # Start with all True
            for v in free_var_list:
                assignment[v] = True
        elif attempts % 5 == 3:
            # Start with all False
            for v in free_var_list:
                assignment[v] = False
        elif attempts % 5 == 4:
            # Start with random but biased
            for v in free_var_list:
                assignment[v] = rng.random() < 0.6
        else:
            # Keep current assignment (continuation)
            pass

        # Vary noise parameters
        noise_start = 0.8 - 0.5 * min(attempts / 10, 1.0)
        noise_end = 0.2 - 0.1 * min(attempts / 10, 1.0)
        if noise_start < 0.1:
            noise_start = 0.1
        if noise_end < 0.05:
            noise_end = 0.05

        if walksat(steps_per_attempt, noise_start, noise_end):
            break

    # Build final assignment
    final_assignment = [False] * n
    for v in range(n):
        if v in fixed:
            final_assignment[v] = fixed[v]
        else:
            final_assignment[v] = assignment.get(v, True)

    return final_assignment
# EVOLVE-BLOCK-END
