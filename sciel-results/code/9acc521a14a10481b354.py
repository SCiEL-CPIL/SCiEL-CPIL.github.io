"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 290s safety margin before 330s kill

    # Pre-compute literal indices for speed
    # clauses is list of lists of signed ints (1-based)
    # Convert to 0-based indices with sign
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]

    def satisfied(assignment, clause_idx):
        for idx, pos in clause_idx:
            if assignment[idx] == pos:
                return True
        return False

    def is_satisfied(assignment, clause_idx):
        return satisfied(assignment, clause_idx)

    def count_unsat(assignment, clauses_idx):
        count = 0
        for c in clauses_idx:
            if not satisfied(assignment, c):
                count += 1
        return count

    best_assignment = None
    best_unsat_count = n + len(clauses_idx) + 1

    rng = random.Random(42)

    # WalkSAT with restarts
    max_flips_per_run = 5000
    num_restarts = 50

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = count_unsat(assignment, clauses_idx)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # WalkSAT loop
        for flip in range(max_flips_per_run):
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses_idx:
                if not satisfied(assignment, c):
                    unsat_clauses.append(c)

            if not unsat_clauses:
                return assignment

            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)

            # Count how many literals in clause would be satisfied if flipped
            # Use noise parameter to decide
            noise = 0.3
            # Count satisfied literals in clause (would become unsatisfied)
            sat_count = 0
            unsat_count_in_clause = 0
            for idx, pos in clause:
                if assignment[idx] == pos:
                    sat_count += 1
                else:
                    unsat_count_in_clause += 1

            # With probability (1-noise), flip a literal that is currently unsatisfied (to satisfy clause)
            # With probability noise, flip any literal
            if unsat_count_in_clause > 0 and rng.random() > noise:
                # Flip an unsatisfied literal
                for idx, pos in clause:
                    if not assignment[idx] == pos:
                        assignment[idx] = not assignment[idx]
                        break
            else:
                # Flip a random literal in clause
                idx, pos = rng.choice(clause)
                assignment[idx] = not assignment[idx]

            # Update unsat count incrementally would be better, but for simplicity recompute
            # Actually, let's just check if we solved it
            new_unsat = count_unsat(assignment, clauses_idx)
            if new_unsat == 0:
                return assignment

            if new_unsat < best_unsat_count:
                best_unsat_count = new_unsat
                best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: return random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
