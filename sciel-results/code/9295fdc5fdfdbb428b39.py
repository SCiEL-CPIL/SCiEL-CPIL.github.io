"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Efficient WalkSAT with incremental bookkeeping, clause weighting, and restarts
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)

    # Preprocess: convert clauses to 0-based indices
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    m = len(clause_vars)

    # Build variable-to-clause index for efficient damage computation
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((ci, is_pos))

    # Time limit
    start_time = time.time()
    time_limit = 290.0

    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def is_satisfied(ci, assignment):
        for var_idx, is_pos in clause_vars[ci]:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def count_unsat(assignment):
        return sum(1 for i in range(m) if not is_satisfied(i, assignment))

    def walksat_once(assignment, max_steps, noise_param):
        """Run one WalkSAT search with incremental bookkeeping."""
        # Initialize unsat set
        unsat_set = set()
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)

        if not unsat_set:
            return assignment, True

        # Clause weights for weighted clause selection
        clause_weights = [1.0] * m
        weight_decay = 0.999

        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a clause with probability proportional to weight
            unsat_list = list(unsat_set)
            total_weight = sum(clause_weights[i] for i in unsat_list)
            r = rng.random() * total_weight
            cumulative = 0.0
            chosen_clause = unsat_list[0]
            for i in unsat_list:
                cumulative += clause_weights[i]
                if cumulative >= r:
                    chosen_clause = i
                    break

            clause = clause_vars[chosen_clause]

            if noise_param >= 1.0 or rng.random() < noise_param:
                # Random flip
                var_idx, is_pos = rng.choice(clause)
                new_val = is_pos
            else:
                # Damage-minimizing flip using var_to_clauses index
                best_var = -1
                best_damage = float('inf')
                best_flip = False

                for var_idx, is_pos in clause:
                    new_val = is_pos
                    if assignment[var_idx] == new_val:
                        # Flipping this would make it worse for this clause
                        # But we're picking from unsat clause, so this var is currently
                        # set to the opposite. Flipping to is_pos satisfies this clause.
                        pass

                    # Compute damage: number of currently satisfied clauses
                    # that contain the opposite literal of (var_idx, new_val)
                    damage = 0
                    opposite = not new_val
                    for (ci, pos_in_c) in var_to_clauses[var_idx]:
                        if ci == chosen_clause:
                            continue
                        # Check if this clause contains the opposite literal
                        # pos_in_c is whether the literal in this clause is positive
                        if pos_in_c == opposite:
                            # This clause has the opposite literal
                            # Check if it's currently satisfied
                            if ci not in unsat_set:
                                damage += 1

                    if damage < best_damage:
                        best_damage = damage
                        best_var = var_idx
                        best_flip = new_val

                if best_var >= 0:
                    var_idx = best_var
                    new_val = best_flip
                else:
                    var_idx, is_pos = rng.choice(clause)
                    new_val = is_pos

            # Apply flip
            old_val = assignment[var_idx]
            if old_val == new_val:
                continue
            assignment[var_idx] = new_val

            # Update unsat_set incrementally
            for (ci, pos_in_c) in var_to_clauses[var_idx]:
                if ci in unsat_set:
                    if is_satisfied(ci, assignment):
                        unsat_set.discard(ci)
                else:
                    if not is_satisfied(ci, assignment):
                        unsat_set.add(ci)

            # Update clause weights
            for i in range(m):
                clause_weights[i] *= weight_decay
            for i in unsat_set:
                clause_weights[i] += 1.0

        return assignment, len(unsat_set) == 0

    # Main search loop with restarts
    best_assignment = random_assignment()
    best_unsat = count_unsat(best_assignment)

    steps_per_restart = 50000
    noise = 0.25

    while True:
        if time.time() - start_time > time_limit:
            break

        # Choose starting assignment
        if rng.random() < 0.2:
            assignment = best_assignment[:]
        else:
            assignment = random_assignment()

        assignment, solved = walksat_once(assignment, steps_per_restart, noise)

        if solved:
            return assignment

        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
