"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    # Precompute clause literal info: for each clause, store (var_index, polarity)
    # polarity: +1 means literal is positive (x=True satisfies), -1 means negative (x=False satisfies)
    clause_info = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_info.append(lits)

    # Count literal occurrences for initial assignment heuristic
    pos_count = [0] * n
    neg_count = [0] * n
    for c in clauses:
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                pos_count[idx] += 1
            else:
                neg_count[idx] += 1

    def initial_assignment(seed):
        rng = random.Random(seed)
        assignment = [False] * n
        for i in range(n):
            # Bias toward the more frequent polarity, with some randomness
            if pos_count[i] > neg_count[i]:
                assignment[i] = rng.random() < 0.7
            elif neg_count[i] > pos_count[i]:
                assignment[i] = rng.random() < 0.3
            else:
                assignment[i] = rng.random() < 0.5
        return assignment

    def count_unsat(assignment):
        """Return list of unsatisfied clause indices and count."""
        unsat = []
        for ci, lits in enumerate(clause_info):
            sat = False
            for var_idx, pol in lits:
                if assignment[var_idx] == (pol == 1):
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat

    # Try multiple restarts with increasing budgets
    max_total_flips = 50000
    num_restarts = 5
    base_budget = 8000

    for restart in range(num_restarts):
        seed = 42 + restart * 137
        assignment = initial_assignment(seed)
        rng = random.Random(seed)

        # Variable activity scores
        activity = [1.0] * n

        budget = base_budget + restart * 4000
        if budget > max_total_flips:
            budget = max_total_flips

        for step in range(budget):
            unsat = count_unsat(assignment)
            if not unsat:
                return assignment

            # Pick unsat clause: prefer one with fewer false literals (closer to satisfied)
            # or highest total activity among its literals
            best_ci = None
            best_score = -1
            # Sample a few candidates for speed
            candidates = rng.sample(unsat, min(5, len(unsat)))
            for ci in candidates:
                lits = clause_info[ci]
                # Score: sum of activities of literals that are currently false (i.e., flipping them would help)
                score = 0.0
                for var_idx, pol in lits:
                    if assignment[var_idx] != (pol == 1):
                        score += activity[var_idx]
                if score > best_score:
                    best_score = score
                    best_ci = ci

            clause = clause_info[best_ci]

            # Find the literal to flip: prefer the one with highest activity among false literals
            # (flipping a false literal in an unsat clause makes it true)
            false_lits = [(var_idx, pol) for var_idx, pol in clause if assignment[var_idx] != (pol == 1)]
            if not false_lits:
                # Shouldn't happen for unsat clause, but fallback
                var_idx, pol = rng.choice(clause)
            else:
                # Weighted random choice by activity
                weights = [activity[vi] for vi, _ in false_lits]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0.0
                chosen = false_lits[0]
                for i, (vi, pol) in enumerate(false_lits):
                    cum += weights[i]
                    if r <= cum:
                        chosen = false_lits[i]
                        break
                var_idx, pol = chosen

            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

            # Update activity: increase activity of variables in the clause
            for var_idx2, _ in clause:
                activity[var_idx2] *= 1.01

        # After budget exhausted, check one more time
        unsat = count_unsat(assignment)
        if not unsat:
            return assignment

    # Final fallback: return last assignment
    return assignment

# EVOLVE-BLOCK-END
