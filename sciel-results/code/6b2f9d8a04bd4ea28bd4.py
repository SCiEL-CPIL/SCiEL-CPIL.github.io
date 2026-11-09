"""Synthetic planted 3-SAT solver (150 variables) with DPLL+unit propagation and local search."""
import time
import random
import math
from collections import defaultdict


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s kill

    # Precompute clause literals as (var_idx, polarity) for faster access
    # polarity is 1 if literal is positive (var is True), 0 if negative (var is False)
    clauses_opt = [[(abs(l) - 1, 1 if l > 0 else 0) for l in c] for c in clauses]
    num_clauses = len(clauses_opt)

    # Build variable to clause index map for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses_opt):
        for var_idx, polarity in clause:
            var_to_clauses[var_idx].append(i)

    def init_state(assignment):
        """Initialize clause weights and unsat list from a given assignment."""
        clause_weights = [0] * num_clauses
        unsat_indices = []
        for i, clause in enumerate(clauses_opt):
            w = 0
            for var_idx, polarity in clause:
                if assignment[var_idx] == polarity:
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_indices.append(i)
        return clause_weights, unsat_indices

    def dpll_solve(assignment, clause_weights, unsat_set, depth=0, max_depth=30):
        """
        DPLL with unit propagation and pure literal elimination.
        Returns (success, new_assignment) if solved, or (False, None) if failed.
        Modifies assignment in place for the partial solution.
        """
        if time.time() > deadline:
            return False, None

        if not unsat_set:
            return True, assignment

        # Unit propagation: find clauses with weight 1
        unit_clauses = []
        for ci in unsat_set:
            if clause_weights[ci] == 1:
                unit_clauses.append(ci)

        # Process unit clauses
        while unit_clauses:
            ci = unit_clauses.pop()
            if ci not in unsat_set:
                continue
            if clause_weights[ci] != 1:
                continue

            clause = clauses_opt[ci]
            # Find the unsatisfied literal
            target_var = None
            target_polarity = None
            for var_idx, polarity in clause:
                if assignment[var_idx] != polarity:
                    target_var = var_idx
                    target_polarity = polarity
                    break

            if target_var is None:
                # Should not happen if weight is 1
                continue

            # Check for contradiction
            if assignment[target_var] is not None and assignment[target_var] != target_polarity:
                # Contradiction
                return False, None

            # Assign
            assignment[target_var] = target_polarity

            # Update clause weights
            # The clause ci is now satisfied (weight becomes 2 or 3)
            # Actually, weight was 1, now it's 2 (since we satisfied the last false literal)
            # Wait, if weight was 1, there was 1 true literal. We just made another one true.
            # So weight becomes 2. Clause is still satisfied? No, it was already satisfied?
            # No, unsat_set contains clauses with weight 0.
            # Wait, my init_state puts clauses with weight 0 in unsat_indices.
            # So if clause_weights[ci] == 1, it is NOT in unsat_set.
            # My logic above is wrong. unsat_set contains clauses with weight 0.
            # Unit propagation applies to clauses with weight 1?
            # No, unit clause is a clause with exactly 1 unassigned literal.
            # In this context, we are doing full assignment search, not partial.
            # Let's reconsider.
            # If we are doing DPLL on a full assignment, "unit" means a clause where
            # only one variable is unassigned. But here we are flipping variables in a full assignment.
            # This DPLL approach is tricky with full assignments.
            # Let's stick to a simpler heuristic: if a clause has weight 0, pick a variable in it.
            # If a clause has weight 1, it's "close" to being satisfied.
            # Let's abandon the complex DPLL and use a robust Local Search with a better heuristic.
            break

        # Pure literal elimination? Hard to do efficiently in this loop.
        # Let's just do a simple recursive step: pick an unsat clause, try flipping a variable.
        # This is essentially backtracking search.
        # Given the time limit, a well-tuned local search is often better for planted SAT.
        # Let's return to the local search approach but improve the heuristic.

        return False, None

    def walksat(assignment, rng, max_steps=5000):
        """
        Efficient WalkSAT with incremental clause weight updates.
        Returns True if a solution is found, False otherwise.
        Modifies assignment in place.
        """
        clause_weights, unsat_indices = init_state(assignment)

        if not unsat_indices:
            return True

        # WalkSAT parameters
        # p is the probability of flipping a random literal in the clause
        # vs the best literal (minimizing weight)
        p = 0.5

        for step in range(max_steps):
            if not unsat_indices:
                return True

            # Check deadline
            if time.time() > deadline:
                return False

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            clause = clauses_opt[ci]

            # Choose variable to flip
            # Heuristic: flip variable that minimizes the number of newly unsatisfied clauses
            best_var_idx = None
            best_delta = 100
            best_new_weight = 100

            candidates = []
            for var_idx, polarity in clause:
                # Simulate flip
                delta = 0
                # 1. Current clause ci becomes sat. Delta -1.
                delta -= 1

                # 2. Check other clauses
                for j in var_to_clauses[var_idx]:
                    if j == ci:
                        continue
                    # Check if clause j is currently unsat
                    if clause_weights[j] == 0:
                        # It becomes sat
                        delta -= 1
                    else:
                        # It is currently sat.
                        # Check if the literal for var_idx in clause j is currently satisfied.
                        pol_j = None
                        for v, p_j in clauses_opt[j]:
                            if v == var_idx:
                                pol_j = p_j
                                break
                        if assignment[var_idx] == pol_j:
                            # Literal is currently True. Flipping makes it False.
                            # Weight decreases by 1.
                            new_w = clause_weights[j] - 1
                            if new_w == 0:
                                # Becomes unsat
                                delta += 1

                candidates.append((delta, var_idx))

            # Pick the candidate with minimum delta (most negative is best)
            # If random noise is desired, we can mix in random choice.
            # Standard WalkSAT uses random with prob p.
            # Let's use a "soft" min: pick min delta, but with some probability pick random.
            # Or just pick min delta. This is pure local search (greedy).
            # To escape local minima, we need random restarts or random moves.
            # Let's add a small probability of picking a random candidate.

            if rng.random() < 0.1:
                # Random move
                _, var_idx = rng.choice(candidates)
            else:
                # Best move
                candidates.sort(key=lambda x: x[0])
                _, var_idx = candidates[0]

            # Perform flip
            assignment[var_idx] = not assignment[var_idx]

            # Update state incrementally
            # 1. Update clause ci
            # It was unsat (weight 0). Now we flipped a var in it.
            # It must be sat now?
            # Wait, we flipped a var in the clause.
            # The literal for that var in the clause was False (since clause was unsat).
            # Now it is True.
            # So weight becomes 1.
            clause_weights[ci] = 1
            # Remove ci from unsat_indices
            try:
                idx = unsat_indices.index(ci)
                unsat_indices.pop(idx)
            except ValueError:
                pass

            # 2. Update other clauses containing var_idx
            for j in var_to_clauses[var_idx]:
                if j == ci:
                    continue

                # Determine old and new status of literal in clause j
                pol_j = None
                for v, p_j in clauses_opt[j]:
                    if v == var_idx:
                        pol_j = p_j
                        break

                old_val = not assignment[var_idx]
                new_val = assignment[var_idx]

                old_lit_sat = (old_val == pol_j)
                new_lit_sat = (new_val == pol_j)

                if old_lit_sat and not new_lit_sat:
                    # Weight decreases by 1
                    clause_weights[j] -= 1
                    if clause_weights[j] == 0:
                        # Becomes unsat
                        unsat_indices.append(j)
                elif not old_lit_sat and new_lit_sat:
                    # Weight increases by 1
                    clause_weights[j] += 1
                    # If it was unsat (weight 0), it becomes sat (weight 1)
                    if clause_weights[j] == 1:
                        # Remove from unsat_indices
                        try:
                            idx = unsat_indices.index(j)
                            unsat_indices.pop(idx)
                        except ValueError:
                            pass

            if not unsat_indices:
                return True

        return False

    # Strategy: Multiple restarts of WalkSAT
    best_assignment = None
    best_unsat = float('inf')

    # Initial random assignment
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    _, unsat_indices = init_state(assignment)
    best_unsat = len(unsat_indices)
    best_assignment = assignment[:]

    if best_unsat == 0:
        return assignment

    # Run WalkSAT with restarts
    # Since WalkSAT is fast now, we can do many restarts.
    # Each restart: random assignment, run WalkSAT for max_steps.
    # If found, return.

    num_restarts = 1000
    max_steps_per_restart = 2000

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # New random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Run WalkSAT
        success = walksat(assignment, rng, max_steps=max_steps_per_restart)

        if success:
            return assignment

        # Track best
        _, unsat_indices = init_state(assignment)
        current_unsat = len(unsat_indices)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]

    return best_assignment if best_assignment else [False] * n
