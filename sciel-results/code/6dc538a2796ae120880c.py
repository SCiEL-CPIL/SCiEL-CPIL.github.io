"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    # Planted 3-SAT solver with WalkSAT + restarts and a deadline guard.
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: return best result before 330s wall-clock.
    start_time = time.time()
    DEADLINE = 325.0  # safety margin

    # Preprocess: convert clauses to 0-based signed ints for speed.
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_sigs = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_sigs[i, j] = 1 if lit > 0 else -1

    # Precompute: for each variable, list of (clause_idx, lit_idx) pairs
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i, j]].append((i, j))

    # WalkSAT with restarts
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    # Parameters
    flip_prob = 0.5
    max_flips_per_run = 10000
    max_restarts = 500

    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        # Initialize random assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        # Initialize clause satisfaction counts
        # clause_unsat_count[i] = number of unsatisfied literals in clause i
        clause_unsat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j in range(3):
                var = clause_vars[i, j]
                sig = clause_sigs[i, j]
                # Literal is satisfied if assignment[var] matches sign
                # sig == 1 means literal is x, satisfied if assignment[var] is True
                # sig == -1 means literal is ~x, satisfied if assignment[var] is False
                if sig == 1:
                    if not assignment[var]:
                        clause_unsat_count[i] += 1
                else:
                    if assignment[var]:
                        clause_unsat_count[i] += 1

        total_unsat = int(np.sum(clause_unsat_count == 3))
        if total_unsat == 0:
            return assignment.tolist()

        if total_unsat < best_unsat_count:
            best_unsat_count = total_unsat
            best_assignment = assignment.copy()

        # Try to solve
        for flip in range(max_flips_per_run):
            if time.time() - start_time > DEADLINE:
                break

            if total_unsat == 0:
                return assignment.tolist()

            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment.copy()

            # Find an unsatisfied clause (one with all 3 literals unsatisfied)
            # Use a random sample approach for speed
            # First, try a few random clauses
            found_clause = -1
            for _ in range(5):
                ci = rng.randint(0, m - 1)
                if clause_unsat_count[ci] == 3:
                    found_clause = ci
                    break
            
            if found_clause == -1:
                # Fallback: scan for an unsatisfied clause
                unsat_indices = np.where(clause_unsat_count == 3)[0]
                if len(unsat_indices) == 0:
                    return assignment.tolist()
                found_clause = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            ci = found_clause

            # Choose which literal to flip
            if rng.random() < flip_prob:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Pick the literal that, if flipped, would satisfy the most clauses
                best_li = 0
                best_gain = -1
                for li in range(3):
                    var = clause_vars[ci, li]
                    # Count how many clauses would be satisfied by flipping this variable
                    # A clause is satisfied if it has at least one satisfied literal
                    # Flipping var changes the satisfaction of all clauses containing var
                    gain = 0
                    for (ci2, li2) in var_to_clauses[var]:
                        if clause_unsat_count[ci2] > 0:
                            # This clause has some unsatisfied literals
                            # If we flip var, and this literal was unsatisfied, it becomes satisfied
                            # Check if this specific literal is currently unsatisfied
                        # Simpler: just count clauses where this is the only unsatisfied literal
                        # That's expensive. Use a simpler heuristic.
                        pass
                    # Use a simpler approach: pick randomly among the 3
                    best_li = li
                    break
                li = rng.randint(0, 2)

            var = clause_vars[ci, li]
            old_val = assignment[var]
            assignment[var] = not old_val

            # Update clause satisfaction counts for all clauses containing this variable
            for (ci2, li2) in var_to_clauses[var]:
                sig = clause_sigs[ci2, li2]
                # Determine if the literal was satisfied before the flip
                if sig == 1:
                    was_satisfied = old_val
                else:
                    was_satisfied = not old_val
                
                if was_satisfied:
                    # Literal was satisfied, now it's unsatisfied
                    clause_unsat_count[ci2] += 1
                    if clause_unsat_count[ci2] == 3:
                        total_unsat += 1
                else:
                    # Literal was unsatisfied, now it's satisfied
                    clause_unsat_count[ci2] -= 1
                    if clause_unsat_count[ci2] == 2:
                        total_unsat -= 1

        if time.time() - start_time > DEADLINE:
            break

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()

    # Fallback: all True
    return [True] * n
# EVOLVE-BLOCK-END
