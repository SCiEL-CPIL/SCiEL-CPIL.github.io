"""Planted 3-SAT solver using weighted WalkSAT with restarts and phase saving."""
# EVOLVE-BLOCK-START
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)

    # Pre-compute for each variable the list of (clause_index, literal_sign) it appears in
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occurrences[v].append((ci, sign))

    # Phase saving
    best_assignment = [rng.random() < 0.5 for _ in range(n)]
    best_unsat_count = len(clauses)  # worst case

    max_restarts = 100
    max_steps_per_restart = 30000
    weight_increment = 1.0
    flip_prob = 0.5

    for restart in range(max_restarts):
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            assignment = best_assignment[:]

        clause_weights = [1.0] * len(clauses)

        # Initialize sat_count
        sat_count = [0] * len(clauses)
        for ci, clause in enumerate(clauses):
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    sat_count[ci] += 1
                    break

        unsat_set = set(ci for ci in range(len(clauses)) if sat_count[ci] == 0)
        current_unsat = len(unsat_set)

        for step in range(max_steps_per_restart):
            if current_unsat == 0:
                return assignment

            # Track best during this restart
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            # Score each literal in the clause
            best_score = -1
            best_lit = None
            for lit in clause:
                v = abs(lit) - 1
                current_val = assignment[v]
                gain = 0
                loss = 0
                for (cj, sign) in var_occurrences[v]:
                    if cj == ci:
                        continue
                    v_satisfies_cj = (assignment[v] == (sign > 0))
                    if not v_satisfies_cj:
                        if sat_count[cj] == 0:
                            gain += clause_weights[cj]
                    else:
                        if sat_count[cj] == 1:
                            loss += clause_weights[cj]
                score = gain - loss
                if score > best_score:
                    best_score = score
                    best_lit = lit

            if rng.random() < flip_prob:
                lit = best_lit
            else:
                lit = rng.choice(clause)

            v = abs(lit) - 1

            # Update sat_counts for all clauses containing v
            for (cj, sign) in var_occurrences[v]:
                v_satisfies_cj = (assignment[v] == (sign > 0))
                if v_satisfies_cj:
                    sat_count[cj] -= 1
                    if sat_count[cj] == 0:
                        unsat_set.add(cj)
                        current_unsat += 1
                else:
                    sat_count[cj] += 1
                    if sat_count[cj] == 1:
                        unsat_set.discard(cj)
                        current_unsat -= 1

            assignment[v] = not assignment[v]

            # Update clause weights for unsatisfied clauses
            for cj in unsat_set:
                clause_weights[cj] += weight_increment

    return best_assignment
# EVOLVE-BLOCK-END
