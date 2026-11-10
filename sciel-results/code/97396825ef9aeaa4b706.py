# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """Optimized WalkSAT with unsat-degree heuristic for faster local search."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build index: var_idx -> list of (clause_idx, sign_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))

    def clause_is_satisfied(ci, assignment):
        """Check if clause ci is satisfied under assignment."""
        for var_idx, sign in clause_data[ci]:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def build_unsat_set_and_degrees(assignment):
        """Build set of unsatisfied clause indices and unsat_degree per variable."""
        unsat = set()
        unsat_degree = [0] * n
        for ci in range(num_clauses):
            if not clause_is_satisfied(ci, assignment):
                unsat.add(ci)
                for var_idx, _ in clause_data[ci]:
                    unsat_degree[var_idx] += 1
        return unsat, unsat_degree

    def walk_one(assignment, rng, max_steps):
        """Run WalkSAT with incremental unsat tracking and unsat-degree heuristic."""
        # Build initial unsatisfied clause set and degrees
        unsat_clauses, unsat_degree = build_unsat_set_and_degrees(assignment)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            clause = clause_data[ci]

            # Heuristic: Pick variable with minimum unsat_degree among candidates
            # With probability p, pick randomly to escape local minima
            p = 0.4
            if rng.random() < p:
                chosen_idx = rng.randrange(len(clause))
            else:
                # Find min unsat_degree
                min_deg = float('inf')
                best_candidates = []
                for i, (var_idx, sign) in enumerate(clause):
                    deg = unsat_degree[var_idx]
                    if deg < min_deg:
                        min_deg = deg
                        best_candidates = [i]
                    elif deg == min_deg:
                        best_candidates.append(i)
                chosen_idx = rng.choice(best_candidates)

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Update unsat_clauses and unsat_degree incrementally
            # Only clauses containing var_idx can change status
            for (uci, u_sign) in var_to_clauses[var_idx]:
                was_unsat = uci in unsat_clauses
                is_now_satisfied = clause_is_satisfied(uci, assignment)
                
                if was_unsat and is_now_satisfied:
                    # Clause became satisfied
                    unsat_clauses.discard(uci)
                    for v2, _ in clause_data[uci]:
                        unsat_degree[v2] -= 1
                elif not was_unsat and not is_now_satisfied:
                    # Clause became unsatisfied
                    unsat_clauses.add(uci)
                    for v2, _ in clause_data[uci]:
                        unsat_degree[v2] += 1

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 50000 # Increased due to speedup
    num_restarts = 500  # Max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart)

        if solved:
            return assignment

        # Track best
        uc, _ = build_unsat_set_and_degrees(assignment)
        if len(uc) < best_unsat_count:
            best_unsat_count = len(uc)
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
