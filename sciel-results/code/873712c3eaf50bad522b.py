"""Planted 3-SAT solver with conflict-driven local search and restarts."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 250  # leave margin under 300s

    # Precompute clause lengths and literal sets for speed
    clause_lens = [len(c) for c in clauses]
    num_clauses = len(clauses)

    def make_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def is_sat(assignment):
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        cnt = 0
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # Strategy 1: WalkSAT with max-unsat variable selection and restarts
    def walksat_max_unsat(seed, max_flips):
        rng = random.Random(seed)
        assignment = make_assignment(rng)
        phase = [None] * n  # phase saving
        for i in range(n):
            phase[i] = assignment[i]

        for flip in range(max_flips):
            if time.time() - start_time > time_limit:
                return None
            # Find unsatisfied clauses
            unsat = []
            for c in clauses:
                sat = False
                for l in c:
                    if assignment[abs(l) - 1] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat.append(c)
            if not unsat:
                return assignment

            # Pick a random unsat clause
            clause = unsat[rng.randrange(len(unsat))]

            # Count how many times each variable appears in unsat clauses
            # Weight: prefer variables in more unsat clauses
            var_score = {}
            for c in unsat:
                for l in c:
                    v = abs(l) - 1
                    var_score[v] = var_score.get(v, 0) + 1

            # Pick variable with max score, break ties randomly
            max_score = max(var_score.values())
            candidates = [v for v, s in var_score.items() if s == max_score]
            v = candidates[rng.randrange(len(candidates))]

            # Decide flip direction: prefer the polarity that satisfies this clause
            # and also use phase saving
            # Count how many unsat clauses would be satisfied if we flip v
            # to True vs False
            score_true = 0
            score_false = 0
            for c in unsat:
                has_v_pos = False
                has_v_neg = False
                for l in c:
                    if abs(l) - 1 == v:
                        if l > 0:
                            has_v_pos = True
                        else:
                            has_v_neg = True
                if has_v_pos:
                    score_true += 1
                if has_v_neg:
                    score_false += 1

            if score_true > score_false:
                new_val = True
            elif score_false > score_true:
                new_val = False
            else:
                # Use phase saving with some randomness
                if phase[v] is not None and rng.random() < 0.7:
                    new_val = phase[v]
                else:
                    new_val = rng.random() < 0.5

            assignment[v] = new_val
            phase[v] = new_val

        return None

    # Strategy 2: Random restarts with simple random flip
    def walksat_random(seed, max_flips):
        rng = random.Random(seed)
        assignment = make_assignment(rng)
        for flip in range(max_flips):
            if time.time() - start_time > time_limit:
                return None
            unsat = []
            for c in clauses:
                sat = False
                for l in c:
                    if assignment[abs(l) - 1] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat.append(c)
            if not unsat:
                return assignment
            clause = unsat[rng.randrange(len(unsat))]
            literal = clause[rng.randrange(len(clause))]
            v = abs(literal) - 1
            assignment[v] = not assignment[v]
        return None

    # Try multiple seeds with increasing flip budgets
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
             7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]

    # First pass: try max-unsat strategy with moderate flips
    for i, seed in enumerate(seeds):
        if time.time() - start_time > time_limit:
            break
        max_flips = 5000 + i * 2000
        result = walksat_max_unsat(seed, max_flips)
        if result is not None:
            return result

    # Second pass: try random flip strategy
    for i, seed in enumerate(seeds):
        if time.time() - start_time > time_limit:
            break
        max_flips = 5000 + i * 2000
        result = walksat_random(seed, max_flips)
        if result is not None:
            return result

    # Third pass: longer runs
    for i, seed in enumerate(range(100, 200)):
        if time.time() - start_time > time_limit:
            break
        max_flips = 20000 + i * 5000
        result = walksat_max_unsat(seed, max_flips)
        if result is not None:
            return result

    # Final fallback: return best effort (last assignment from last run)
    # Just return a random assignment
    rng = random.Random(0)
    return make_assignment(rng)
