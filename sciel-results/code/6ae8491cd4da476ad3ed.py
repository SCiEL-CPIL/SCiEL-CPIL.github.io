"""Synthetic planted 3-SAT solver (150 variables) with optimized WalkSAT and local search."""
import time
import random
import math


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
    # Precompute polarity: var_polarity[var_idx][clause_idx] = polarity (0 or 1)
    var_polarity = [{} for _ in range(n)]
    for i, clause in enumerate(clauses_opt):
        for var_idx, polarity in clause:
            var_to_clauses[var_idx].append(i)
            var_polarity[var_idx][i] = polarity

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

    def walksat(assignment, rng, max_steps=5000):
        """
        Efficient WalkSAT with incremental clause weight updates.
        Returns True if a solution is found, False otherwise.
        Modifies assignment in place.
        """
        clause_weights, unsat_indices = init_state(assignment)

        if not unsat_indices:
            return True

        # Use a set for O(1) removal from unsat set
        unsat_set = set(unsat_indices)

        for step in range(max_steps):
            if not unsat_set:
                return True

            # Check deadline
            if time.time() > deadline:
                return False

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses_opt[ci]

            # Choose variable to flip using precomputed polarity
            candidates = []
            for var_idx, polarity in clause:
                # Simulate flip
                delta = -1  # Current clause ci becomes sat

                # Check other clauses using precomputed polarity
                for j in var_to_clauses[var_idx]:
                    if j == ci:
                        continue
                    pol_j = var_polarity[var_idx][j]
                    if clause_weights[j] == 0:
                        # Unsat clause: flipping any var in it makes weight 1, so delta -= 1
                        delta -= 1
                    else:
                        # Sat clause: check if literal is currently satisfied
                        if assignment[var_idx] == pol_j:
                            # Literal is True. Flipping makes it False.
                            new_w = clause_weights[j] - 1
                            if new_w == 0:
                                delta += 1

                candidates.append((delta, var_idx))

            # Pick best or random candidate
            if rng.random() < 0.1:
                _, var_idx = rng.choice(candidates)
            else:
                candidates.sort(key=lambda x: x[0])
                _, var_idx = candidates[0]

            # Perform flip
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Update state incrementally
            # 1. Update clause ci: was unsat (weight 0), now weight 1
            clause_weights[ci] = 1
            unsat_set.discard(ci)

            # 2. Update other clauses containing var_idx
            for j in var_to_clauses[var_idx]:
                if j == ci:
                    continue

                pol_j = var_polarity[var_idx][j]
                old_lit_sat = (old_val == pol_j)
                new_lit_sat = (new_val == pol_j)

                if old_lit_sat and not new_lit_sat:
                    clause_weights[j] -= 1
                    if clause_weights[j] == 0:
                        unsat_set.add(j)
                elif not old_lit_sat and new_lit_sat:
                    clause_weights[j] += 1
                    if clause_weights[j] == 1:
                        unsat_set.discard(j)

            if not unsat_set:
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
    # Adaptive restart: use more steps for harder instances
    num_restarts = 2000
    max_steps_per_restart = 3000

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
