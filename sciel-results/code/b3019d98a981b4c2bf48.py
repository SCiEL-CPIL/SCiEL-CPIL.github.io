"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s (evaluator kills at 330s)

    m = len(clauses)
    if m == 0:
        return [True] * n

    # Precompute clause literal arrays (0-based variable indices) and signs
    # lit_vars: (m, 3) int array of variable indices (0-based)
    # lit_signs: (m, 3) int array of +1 or -1
    lit_vars = np.empty((m, 3), dtype=np.int32)
    lit_signs = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lit_vars[i, j] = abs(lit) - 1
            lit_signs[i, j] = 1 if lit > 0 else -1

    rng = random.Random(0)
    np_rng = np.random.default_rng(0)

    def check_solution(assignment):
        """Check if assignment satisfies all clauses. Returns True if satisfied."""
        # For each clause, check if any literal is satisfied
        # assignment is a numpy bool array (n,)
        # A literal (var, sign) is satisfied if assignment[var] == (sign == 1)
        for i in range(m):
            sat = False
            for j in range(3):
                v = lit_vars[i, j]
                s = lit_signs[i, j]
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def walk_sat(assignment, max_steps):
        """Run WalkSAT with impact-based literal selection. Returns (assignment, solved)."""
        # Initialize clause satisfaction status
        # clause_sat_count[i] = number of literals in clause i that are satisfied
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j in range(3):
                v = lit_vars[i, j]
                s = lit_signs[i, j]
                if assignment[v] == (s == 1):
                    clause_sat_count[i] += 1

        unsat_mask = clause_sat_count == 0
        num_unsat = int(np.sum(unsat_mask))

        if num_unsat == 0:
            return assignment, True

        # Build variable-to-clauses index for impact calculation
        # var_to_clauses[v] = list of (clause_idx, literal_idx) where variable v appears
        var_to_clauses = [[] for _ in range(n)]
        for i in range(m):
            for j in range(3):
                v = lit_vars[i, j]
                var_to_clauses[v].append((i, j))

        # Precompute for each variable and each clause it appears in, whether flipping
        # that variable would satisfy/unsatisfy the clause
        # Actually, let's compute impact on-the-fly but vectorized per step

        # For efficiency, precompute:
        # For each clause i and literal position j, the variable v = lit_vars[i,j]
        # and sign s = lit_signs[i,j]
        # Flipping v changes the satisfaction of literal (i,j):
        #   if currently satisfied -> becomes unsatisfied (sat_count decreases by 1)
        #   if currently unsatisfied -> becomes satisfied (sat_count increases by 1)

        # We'll maintain clause_sat_count and update incrementally

        # Impact-based selection:
        # For a given unsatisfied clause, we pick a literal to flip.
        # The impact of flipping variable v in clause c is:
        #   - For clause c: if the literal is unsatisfied, flipping makes it satisfied (+1)
        #   - For other clauses containing v: flipping may satisfy or unsatisfy them
        # Net impact = (satisfies unsatisfied clauses) - (unsatisfies satisfied clauses)

        # We use a hybrid: with probability p, pick best impact literal; otherwise random
        p_greedy = 0.8

        # Precompute for each variable, the list of (clause_idx, lit_pos) it appears in
        # Already have var_to_clauses

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if num_unsat == 0:
                return assignment, True

            # Pick an unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])

            # For this clause, compute impact of flipping each of its 3 literals
            # and also consider random tie-breaking
            best_impact = -999
            best_var = -1
            impacts = []
            for j in range(3):
                v = int(lit_vars[ci, j])
                # Compute impact of flipping variable v
                # For clause ci, literal j: currently unsatisfied (since clause is unsat, 
                # but this specific literal might be satisfied? No - if clause is unsat, 
                # all its literals are unsatisfied)
                # Wait: clause is unsatisfied means NO literal is satisfied. So all 3 are unsat.
                # Flipping v makes literal (ci, j) satisfied: +1 for clause ci
                impact = 1  # satisfies clause ci

                # Check other clauses where v appears
                for (ci2, j2) in var_to_clauses[v]:
                    if ci2 == ci and j2 == j:
                        continue
                    # Check if literal (ci2, j2) is currently satisfied
                    s2 = lit_signs[ci2, j2]
                    if assignment[v] == (s2 == 1):
                        # Currently satisfied, flipping will unsatisfy it
                        impact -= 1
                    else:
                        # Currently unsatisfied, flipping will satisfy it
                        impact += 1

                impacts.append(impact)

            # Choose literal: greedy with random tie-breaking
            if rng.random() < p_greedy:
                max_impact = max(impacts)
                candidates = [j for j in range(3) if impacts[j] == max_impact]
                j = candidates[rng.randrange(len(candidates))]
            else:
                j = rng.randrange(3)

            v = int(lit_vars[ci, j])

            # Update clause_sat_count for all clauses containing v
            for (ci2, j2) in var_to_clauses[v]:
                s2 = lit_signs[ci2, j2]
                was_satisfied = assignment[v] == (s2 == 1)
                if was_satisfied:
                    clause_sat_count[ci2] -= 1
                else:
                    clause_sat_count[ci2] += 1

            # Update assignment
            assignment[v] = not assignment[v]

            # Update unsat_mask and num_unsat
            # Recompute unsat_mask efficiently
            # Only clauses containing v changed
            for (ci2, j2) in var_to_clauses[v]:
                was_unsat = unsat_mask[ci2]
                now_unsat = (clause_sat_count[ci2] == 0)
                if was_unsat and not now_unsat:
                    num_unsat -= 1
                elif not was_unsat and now_unsat:
                    num_unsat += 1
                unsat_mask[ci2] = now_unsat

        # Final check
        if num_unsat == 0:
            return assignment, True
        return assignment, False

    # Multi-restart WalkSAT with increasing budgets
    best_assignment = None
    best_unsat_count = float('inf')

    # Restart schedule
    restarts = [
        (5000, 0.7),
        (10000, 0.8),
        (20000, 0.85),
        (50000, 0.9),
        (100000, 0.95),
    ]

    for step_budget, p_greedy in restarts:
        if time.time() > deadline:
            break

        # Try multiple restarts with this budget
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initial assignment
            assignment = np.random.random(n) < 0.5

            # Quick check if already satisfied
            if check_solution(assignment):
                return assignment.tolist()

            # Run WalkSAT
            result, solved = walk_sat(assignment, step_budget)

            if solved:
                return result.tolist()

            # Track best
            # Count unsatisfied clauses
            unsat_count = 0
            for i in range(m):
                sat = False
                for j in range(3):
                    v = lit_vars[i, j]
                    s = lit_signs[i, j]
                    if result[v] == (s == 1):
                        sat = True
                        break
                if not sat:
                    unsat_count += 1

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = result.copy()

    # If we have a best assignment, return it
    if best_assignment is not None:
        return best_assignment.tolist()

    # Fallback: random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
