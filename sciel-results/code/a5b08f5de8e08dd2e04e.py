"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # 290s safety margin under 330s limit

    # Precompute clause structure
    clause_vars = []
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    def random_assignment():
        return [random.random() < 0.5 for _ in range(n)]

    def compute_clause_counts(assignment):
        """Compute for each clause how many literals are satisfied."""
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        return counts

    def flip_variable(assignment, vi, counts):
        """Flip variable vi and update clause counts incrementally."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        for ci, sign in var_occurrences[vi]:
            was_sat = old_val == (sign == 1)
            now_sat = new_val == (sign == 1)
            if was_sat and not now_sat:
                counts[ci] -= 1
            elif not was_sat and now_sat:
                counts[ci] += 1

    def is_satisfied(counts):
        return all(c >= 1 for c in counts)

    def count_unsat(counts):
        return sum(1 for c in counts if c == 0)

    def walk_sat(assignment, counts, max_steps, rng, p_random=0.3):
        """WalkSAT with incremental tracking and tunable randomness.
        Returns (success, assignment, counts)."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            lits_in_clause = clause_vars[ci]

            # With probability p_random, flip a random variable in the clause
            if rng.random() < p_random:
                vi = rng.choice(lits_in_clause)[0]
                flip_variable(assignment, vi, counts)
                continue

            # Otherwise, use min-conflicts heuristic
            best_vi = None
            best_score = None
            for vi, sign in lits_in_clause:
                break_count = 0
                for oci, osign in var_occurrences[vi]:
                    if oci == ci:
                        continue
                    cur_sat = assignment[vi] == (osign == 1)
                    if cur_sat and counts[oci] == 1:
                        break_count += 1
                score = break_count
                if best_score is None or score < best_score or (score == best_score and rng.random() < 0.5):
                    best_score = score
                    best_vi = vi

            if best_vi is not None:
                flip_variable(assignment, best_vi, counts)

        return False, assignment, counts

    def local_search(assignment, counts, max_steps, rng):
        """Local search: pick variable that satisfies most unsat clauses."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            var_score = [0] * n
            for ci in unsat:
                for vi, sign in clause_vars[ci]:
                    var_score[vi] += 1

            max_score = max(var_score) if var_score else 0
            if max_score == 0:
                vi = rng.randrange(n)
            else:
                candidates = [vi for vi in range(n) if var_score[vi] == max_score]
                vi = rng.choice(candidates)

            flip_variable(assignment, vi, counts)

        return False, assignment, counts

    def weighted_walk(assignment, counts, max_steps, rng):
        """WalkSAT with weighted literal selection based on unsat clause frequency."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            ci = rng.choice(unsat)
            clause = clause_vars[ci]

            # Count how many unsat clauses each variable in this clause appears in
            var_unsat_freq = {}
            for uc in unsat:
                for v, s in clause_vars[uc]:
                    var_unsat_freq[v] = var_unsat_freq.get(v, 0) + 1

            weights = []
            for vi, sign in clause:
                w = var_unsat_freq.get(vi, 0) + 1
                weights.append(w)

            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            vi = clause[chosen_idx][0]
            flip_variable(assignment, vi, counts)

        return False, assignment, counts

    # Main loop with restarts
    best_assignment = None
    best_unsat_count = m + 1

    # Use different strategies and parameters for diversity
    strategies = [
        ("walk", 0.3),    # WalkSAT with p_random=0.3
        ("walk", 0.5),    # WalkSAT with p_random=0.5
        ("local", None),  # Local search
        ("weighted", None),  # Weighted walk
    ]

    for attempt in range(500):
        if time.time() > deadline:
            break

        rng = random.Random(attempt * 7919 + 13)
        assignment = random_assignment()
        counts = compute_clause_counts(assignment)

        if is_satisfied(counts):
            return assignment

        cur_unsat = count_unsat(counts)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        # Select strategy based on attempt number
        strat_idx = attempt % len(strategies)
        strategy_name, param = strategies[strat_idx]

        # Adjust steps based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Use more steps when we have more time
        base_steps = 5000
        if remaining > 100:
            max_steps = base_steps
        elif remaining > 30:
            max_steps = base_steps // 2
        else:
            max_steps = base_steps // 4

        if strategy_name == "walk":
            success, assignment, counts = walk_sat(assignment, counts, max_steps, rng, p_random=param)
        elif strategy_name == "local":
            success, assignment, counts = local_search(assignment, counts, max_steps, rng)
        elif strategy_name == "weighted":
            success, assignment, counts = weighted_walk(assignment, counts, max_steps, rng)
        else:
            success, assignment, counts = walk_sat(assignment, counts, max_steps, rng, p_random=0.4)

        if success:
            return assignment

        cur_unsat = count_unsat(counts)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
