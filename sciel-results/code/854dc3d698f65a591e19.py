"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Efficient WalkSAT with false-count bookkeeping, weighted min-conflicts flip,
    # random restarts with phase-biased init, and an explicit wall-clock deadline guard.
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295  # safety margin before 330s kill

    # Pre-index clauses as lists of (var_index, sign) for speed
    # var_index is 0-based, sign is +1 or -1
    clause_idx = []
    for c in clauses:
        idx = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            idx.append((v, s))
        clause_idx.append(idx)

    # For each variable, list of (clause_index, sign) where it appears
    var_occ = [[] for _ in range(n)]
    for ci, c in enumerate(clause_idx):
        for v, s in c:
            var_occ[v].append((ci, s))

    def satisfied_clause(assignment, ci):
        for v, s in clause_idx[ci]:
            if assignment[v] == (s == 1):
                return True
        return False

    def all_satisfied(assignment):
        for ci in range(m):
            if not satisfied_clause(assignment, ci):
                return False
        return True

    def build_false_counts(assignment):
        # false_counts[ci] = number of literals in clause ci that are False
        fc = [0] * m
        for ci, c in enumerate(clause_idx):
            cnt = 0
            for v, s in c:
                if assignment[v] != (s == 1):
                    cnt += 1
            fc[ci] = cnt
        return fc

    def flip_var(assignment, var, fc, var_occ_list):
        """Flip variable `var`, update fc incrementally. Returns delta in unsat count."""
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        delta_unsat = 0
        for ci, s in var_occ_list:
            was_false = (old_val != (s == 1))
            now_false = (new_val != (s == 1))
            if was_false and not now_false:
                fc[ci] -= 1
                if fc[ci] == 0:
                    delta_unsat -= 1  # clause became satisfied
            elif not was_false and now_false:
                fc[ci] += 1
                if fc[ci] == len(clause_idx[ci]):
                    delta_unsat += 1  # clause became unsatisfied
        return delta_unsat

    def solve_one_restart(rng, deadline, max_steps):
        # Phase-biased initialization: start with all False, then flip some randomly
        assignment = [False] * n
        # Randomize a fraction to break symmetry
        for i in range(n):
            if rng.random() < 0.5:
                assignment[i] = True

        fc = build_false_counts(assignment)
        unsat_count = sum(1 for x in fc if x == len(clause_idx[x]))

        if unsat_count == 0:
            return assignment

        # Weighted min-conflicts: prefer flipping variables in most unsat clauses
        # Use a simple score: for each var, sum of (3 - fc[ci]) for clauses it's in
        # Actually, for min-conflicts we want to flip a var that appears in many unsat clauses
        # and whose flip would reduce unsat count.

        for step in range(max_steps):
            if time.time() >= deadline:
                return assignment
            if unsat_count == 0:
                return assignment

            # Find unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if fc[ci] == len(clause_idx[ci])]
            if not unsat_clauses:
                return assignment

            # Pick a random unsat clause
            ci = rng.choice(unsat_clauses)
            c = clause_idx[ci]

            # Compute score for each literal in this clause:
            # score = number of other clauses that would become satisfied if we flip this var
            # Simpler: pick the variable whose flip reduces the most false literals
            best_var = None
            best_score = -1
            for v, s in c:
                # If we flip v, how many clauses does it affect?
                # Count clauses where this literal is currently False and would become True
                score = 0
                for cj, sj in var_occ[v]:
                    if assignment[v] != (sj == 1):
                        # Currently false, flipping makes it true
                        score += 1
                    # Also penalize if flipping makes a currently true literal false
                    elif assignment[v] == (sj == 1):
                        score -= 1
                if score > best_score:
                    best_score = score
                    best_var = v

            # With some probability, pick a random variable from the clause (WalkSAT noise)
            if rng.random() < 0.2:
                best_var = rng.choice(c)[0]

            delta = flip_var(assignment, best_var, fc, var_occ[best_var])
            unsat_count += delta

        return assignment

    # Main loop with restarts
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')

    max_steps_per_restart = 5000
    restart_count = 0

    while time.time() < deadline:
        assignment = solve_one_restart(rng, deadline, max_steps_per_restart)
        if time.time() >= deadline:
            break

        # Evaluate
        fc = build_false_counts(assignment)
        unsat_count = sum(1 for x in fc if x == len(clause_idx[x]))

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        restart_count += 1
        # Increase steps slightly for later restarts
        if restart_count % 10 == 0:
            max_steps_per_restart = min(max_steps_per_restart + 500, 20000)

    if best_assignment is not None:
        return best_assignment

    # Fallback: return all False
    return [False] * n
# EVOLVE-BLOCK-END
