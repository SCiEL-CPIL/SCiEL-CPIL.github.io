# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s

    # Precompute variable occurrences for incremental updates
    # var_occ[v] = list of (clause_idx, sign) where sign is +1 if literal is var, -1 if -var
    var_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    m = len(clauses)

    def init_counts(assignment):
        """Compute satisfied counts for each clause."""
        counts = [0] * m
        for ci, clause in enumerate(clauses):
            c = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    c += 1
            counts[ci] = c
        return counts

    def flip_var(var_idx, assignment, counts):
        """Flip variable var_idx, update counts incrementally. Return number of clauses that became unsat/sat."""
        assignment[var_idx] = not assignment[var_idx]
        new_val = assignment[var_idx]
        for ci, sign in var_occ[var_idx]:
            # After flip, the literal (sign * var) is satisfied if new_val == (sign == 1)
            # i.e., if sign == 1: satisfied if new_val is True
            # if sign == -1: satisfied if new_val is False
            if sign == 1:
                is_sat = new_val
            else:
                is_sat = not new_val
            if is_sat:
                counts[ci] += 1
            else:
                counts[ci] -= 1

    def is_satisfied(counts):
        # Check if any clause has count 0
        for c in counts:
            if c == 0:
                return False
        return True

    # WalkSAT with break-count minimization
    def walk_sat(seed, max_steps):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        counts = init_counts(assignment)

        if is_satisfied(counts):
            return assignment

        # Parameters
        noise_prob = 0.25  # probability of random flip vs. best flip
        max_steps_local = max_steps

        for step in range(max_steps_local):
            if time.time() > deadline:
                return None

            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return assignment

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            clause = clauses[ci]

            # Compute break count for each variable in the clause
            # break = number of clauses that become unsat if we flip this var
            # gain = number of clauses that become sat if we flip this var
            best_var = -1
            best_score = float('inf')

            for lit in clause:
                v = abs(lit) - 1
                # Compute break and gain
                break_cnt = 0
                gain_cnt = 0
                for ci2, sign in var_occ[v]:
                    if ci2 == ci:
                        continue
                    # Current state of this literal in clause ci2
                    if sign == 1:
                        currently_sat = assignment[v]
                    else:
                        currently_sat = not assignment[v]

                    if currently_sat:
                        # Flipping v will make this literal unsat
                        if counts[ci2] == 1:
                            # This was the only satisfied literal -> clause becomes unsat
                            break_cnt += 1
                        else:
                            # Clause remains sat
                            pass
                    else:
                        # Flipping v will make this literal sat
                        gain_cnt += 1

                score = break_cnt - gain_cnt
                if score < best_score:
                    best_score = score
                    best_var = v

            # Decide: with probability noise_prob, flip a random var in clause
            if rng.random() < noise_prob:
                lit = rng.choice(clause)
                v = abs(lit) - 1
            else:
                v = best_var

            flip_var(v, assignment, counts)

        return None

    # Try multiple restarts with different seeds
    best_result = None
    seed_base = 42
    step_budget = 50000

    attempt = 0
    while time.time() < deadline:
        seed = seed_base + attempt
        result = walk_sat(seed, step_budget)
        if result is not None:
            # Verify
            for clause in clauses:
                sat = False
                for lit in clause:
                    v = abs(lit) - 1
                    if result[v] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    break
            else:
                return result
        attempt += 1

    # If we get here without finding a solution, return the last assignment
    # Do one more attempt with a longer budget if we have time
    if time.time() < deadline:
        result = walk_sat(seed_base + attempt, 200000)
        if result is not None:
            for clause in clauses:
                sat = False
                for lit in clause:
                    v = abs(lit) - 1
                    if result[v] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    break
            else:
                return result

    # Fallback: return a random assignment (shouldn't happen for planted SAT)
    return [False] * n
# EVOLVE-BLOCK-END
