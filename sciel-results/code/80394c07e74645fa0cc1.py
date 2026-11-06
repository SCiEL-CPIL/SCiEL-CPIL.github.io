import random
import time
import sys


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)

    start_time = time.time()
    time_limit = 285.0  # safety margin before 300s kill

    # Convert clauses to 0-based with sign: each literal becomes (var_idx, is_positive)
    # For speed, precompute arrays
    # clauses_flat: list of (v, sign) where sign is +1 or -1
    # Also build variable -> list of (clause_idx, position, sign) for impact computation

    # Preprocess: remove tautologies and duplicates
    seen_clauses = set()
    processed_clauses = []
    for clause in clauses:
        # Check for tautology
        pos_vars = set()
        neg_vars = set()
        for l in clause:
            v = abs(l)
            if l > 0:
                pos_vars.add(v)
            else:
                neg_vars.add(v)
        if pos_vars & neg_vars:
            continue  # tautology, always satisfied
        key = tuple(sorted(clause))
        if key not in seen_clauses:
            seen_clauses.add(key)
            processed_clauses.append(clause)

    clauses = processed_clauses
    num_clauses = len(clauses)

    if num_clauses == 0:
        return [False] * n

    # Build data structures for fast evaluation
    # For each clause, store (var_idx, sign) pairs where sign=+1 means positive literal
    clause_data = []
    for clause in clauses:
        data = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            data.append((v, sign))
        clause_data.append(data)

    # For each variable, list of (clause_idx, sign) for fast impact calculation
    var_to_clauses = [[] for _ in range(n)]
    for ci, data in enumerate(clause_data):
        for (v, sign) in data:
            var_to_clauses[v].append((ci, sign))

    # State arrays
    # assignment: list of bools (True = variable is True)
    # clause_unsat: bool per clause
    # clause_sat_count: number of satisfied literals per clause (0, 1, 2, or 3)

    def make_initial_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = [0] * num_clauses
        unsat_clauses = set()
        for ci, data in enumerate(clause_data):
            cnt = 0
            for (v, sign) in data:
                if assignment[v] == (sign == 1):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat_clauses.add(ci)
        return assignment, clause_sat_count, unsat_clauses

    def flip_var(v, assignment, clause_sat_count, unsat_clauses):
        """Flip variable v and update clause states. Returns delta in unsat count."""
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val

        delta_unsat = 0
        for (ci, sign) in var_to_clauses[v]:
            # Determine if this literal was satisfied before
            was_sat = old_val == (sign == 1)
            now_sat = new_val == (sign == 1)

            old_cnt = clause_sat_count[ci]
            if was_sat and not now_sat:
                # Literal went from sat to unsat
                clause_sat_count[ci] = old_cnt - 1
                if old_cnt == 1:
                    # Clause just became unsatisfied
                    unsat_clauses.add(ci)
                    delta_unsat += 1
            elif not was_sat and now_sat:
                # Literal went from unsat to sat
                clause_sat_count[ci] = old_cnt + 1
                if old_cnt == 0:
                    # Clause just became satisfied
                    unsat_clauses.discard(ci)
                    delta_unsat -= 1
            # else: no change in sat status for this literal

        return delta_unsat

    best_assignment = None
    best_unsat_count = num_clauses

    rng = random.Random(42)

    max_restarts = 500
    restart = 0

    while restart < max_restarts:
        if time.time() - start_time > time_limit - 3.0:
            break

        # Adaptive parameters
        # Noise parameter p: probability of random flip vs greedy
        # Start with moderate noise, adjust based on progress
        p_noise = 0.3 + 0.3 * (restart % 10) / 10.0  # cycles 0.3 to 0.6
        if restart > 50:
            p_noise = 0.4 + 0.2 * rng.random()

        # Flip budget increases with restart
        flip_budget = 10000 + restart * 2000
        if restart > 100:
            flip_budget = 20000 + (restart - 100) * 5000

        assignment, clause_sat_count, unsat_clauses = make_initial_state(rng)
        curr_unsat = len(unsat_clauses)

        if curr_unsat == 0:
            return assignment

        if curr_unsat < best_unsat_count:
            best_unsat_count = curr_unsat
            best_assignment = assignment[:]

        flips = 0
        stuck_count = 0
        max_stuck = 500

        while flips < flip_budget:
            if time.time() - start_time > time_limit - 1.0:
                break

            if not unsat_clauses:
                return assignment

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            data = clause_data[ci]

            # Compute flip candidates: for each literal in this clause,
            # compute the impact (delta in unsat count) if we flip that variable
            best_delta = float('inf')
            best_var = -1
            deltas = []

            for (v, sign) in data:
                # Compute delta: how many clauses become unsat/sat if we flip v
                delta = 0
                for (ci2, sign2) in var_to_clauses[v]:
                    old_val = assignment[v]
                    new_val = not old_val
                    was_sat = old_val == (sign2 == 1)
                    now_sat = new_val == (sign2 == 1)
                    if was_sat and not now_sat:
                        if clause_sat_count[ci2] == 1:
                            delta += 1
                    elif not was_sat and now_sat:
                        if clause_sat_count[ci2] == 0:
                            delta -= 1
                deltas.append((delta, v))

                if delta < best_delta:
                    best_delta = delta
                    best_var = v

            # Choose flip: with probability p_noise, pick random; else pick best
            if rng.random() < p_noise:
                # Random flip among literals in the clause
                v = rng.choice([x[0] for x in data])
            else:
                # Greedy: pick variable with minimum delta
                # Break ties randomly
                min_delta = min(d[0] for d in deltas)
                candidates = [v for (d, v) in deltas if d == min_delta]
                v = rng.choice(candidates)

            # Perform the flip
            delta = flip_var(v, assignment, clause_sat_count, unsat_clauses)
            curr_unsat += delta
            flips += 1

            if curr_unsat == 0:
                return assignment

            if curr_unsat < best_unsat_count:
                best_unsat_count = curr_unsat
                best_assignment = assignment[:]

            # Stuck detection
            if delta >= 0:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count > max_stuck:
                break

        # End of restart
        restart += 1

    if best_assignment is not None:
        return best_assignment
    return assignment
