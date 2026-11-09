"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # WalkSAT with adaptive flip heuristic and a hard wall-clock deadline.
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)

    # Pre-process: store clauses as lists of (var_index, polarity_bool)
    # var_index is 0-based, polarity_bool is True if literal is positive
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            var_idx = abs(lit) - 1
            pos = (lit > 0)
            pc.append((var_idx, pos))
        processed_clauses.append(pc)

    m = len(processed_clauses)

    # Track which clauses each variable appears in, for fast unsat tracking
    # var_occurrences[v] = list of (clause_idx, polarity)
    var_occurrences = [[] for _ in range(n)]
    for ci, pc in enumerate(processed_clauses):
        for (vi, pol) in pc:
            var_occurrences[vi].append((ci, pol))

    # Count how many literals in each clause are satisfied
    # sat_count[ci] = number of satisfied literals in clause ci
    # A clause is unsat iff sat_count[ci] == 0

    def make_initial_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_sat_count(assignment):
        sc = [0] * m
        for ci, pc in enumerate(processed_clauses):
            cnt = 0
            for (vi, pol) in pc:
                if assignment[vi] == pol:
                    cnt += 1
            sc[ci] = cnt
        return sc

    def try_flip(assignment, sat_count, var_idx):
        """Try flipping var_idx; return (improvement, new_sat_count_delta) or None."""
        # Count how many clauses would become satisfied vs unsatisfied
        gain = 0
        loss = 0
        for (ci, pol) in var_occurrences[var_idx]:
            if assignment[var_idx] == pol:
                # Currently satisfied; flipping makes it unsatisfied
                # Clause stays sat if it has other satisfied literals
                if sat_count[ci] > 1:
                    gain += 0  # still satisfied, just count drops
                else:
                    loss += 1  # clause becomes unsatisfied
            else:
                # Currently unsatisfied; flipping makes it satisfied
                if sat_count[ci] == 0:
                    gain += 1  # clause becomes satisfied
                else:
                    gain += 0  # already satisfied
        return gain - loss

    # Parameters
    p = 0.55  # probability to follow the heuristic (choose literal that maximizes gain)
    # We'll use a simpler approach: pick random unsat clause, then pick literal
    # with weighted probability based on how many other literals in that clause
    # would be satisfied if we flip it.

    def flip_heuristic(assignment, sat_count, clause_idx):
        """Given an unsat clause, pick a literal to flip using WalkSAT heuristic."""
        pc = processed_clauses[clause_idx]
        # For each literal in the clause, compute the number of other unsat clauses
        # that would be satisfied by flipping that variable (excluding this clause)
        # Simplified: count how many literals in this clause are currently false
        # and how many other clauses contain that variable
        weights = []
        for (vi, pol) in pc:
            # If we flip vi, this clause becomes satisfied (since it's unsat and
            # we're flipping a literal in it)
            # Count how many other unsat clauses contain vi and would become satisfied
            # Actually, standard WalkSAT: for each literal in the clause, count the
            # number of other unsatisfied clauses that contain the negation of that
            # literal (i.e., clauses that would be satisfied by the flip)
            cnt = 0
            for (oci, opol) in var_occurrences[vi]:
                if oci == clause_idx:
                    continue
                if sat_count[oci] == 0:
                    # This clause is unsat. Would flipping vi satisfy it?
                    # vi with polarity opol is in clause oci.
                    # If we set assignment[vi] = pol (to satisfy this clause),
                    # then in clause oci, the literal (vi, opol) is satisfied iff
                    # assignment[vi] == opol, i.e., pol == opol.
                    if pol == opol:
                        cnt += 1
            weights.append(cnt)
        # Choose literal with probability proportional to weight + 1
        total = sum(w + 1 for w in weights)
        r = rng.random() * total
        cum = 0
        for i, w in enumerate(weights):
            cum += w + 1
            if r < cum:
                return i
        return len(weights) - 1

    # Main search with deadline
    deadline = time.time() + 280  # 280s safety margin out of 330s

    best_assignment = None
    best_unsat_count = float('inf')

    for restart in range(200):
        if time.time() >= deadline:
            break

        assignment = make_initial_assignment()
        sat_count = compute_sat_count(assignment)
        unsat_count = sum(1 for s in sat_count if s == 0)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        steps = 0
        max_steps = 50000
        stuck_threshold = 5000
        stuck_count = 0
        last_unsat = unsat_count

        while steps < max_steps:
            if time.time() >= deadline:
                break

            # Find an unsat clause
            # For efficiency, we can pick a random clause and check if it's unsat
            # But for correctness, let's iterate through a few random clauses
            clause_idx = None
            for _ in range(min(20, m)):
                ci = rng.randrange(m)
                if sat_count[ci] == 0:
                    clause_idx = ci
                    break
            if clause_idx is None:
                # All clauses are satisfied
                return assignment

            # Choose a literal to flip
            lit_idx = flip_heuristic(assignment, sat_count, clause_idx)
            vi, pol = processed_clauses[clause_idx][lit_idx]

            # Apply the flip
            old_val = assignment[vi]
            new_val = not old_val
            assignment[vi] = new_val

            # Update sat_count for all clauses containing vi
            for (ci, cpol) in var_occurrences[vi]:
                if old_val == cpol:
                    # Was satisfied, now not
                    sat_count[ci] -= 1
                else:
                    # Was not satisfied, now is
                    sat_count[ci] += 1

            # Recompute unsat count (cheap since we track sat_count)
            # Actually, we need to know if any clause went from 0 to 1 or 1 to 0
            # Let's just recompute unsat_count periodically
            steps += 1

            if steps % 100 == 0:
                unsat_count = sum(1 for s in sat_count if s == 0)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]

                if unsat_count < last_unsat:
                    stuck_count = 0
                    last_unsat = unsat_count
                else:
                    stuck_count += 1

                if stuck_count > stuck_threshold:
                    break

        # End of this restart
        if time.time() >= deadline:
            break

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return make_initial_assignment()
# EVOLVE-BLOCK-END
