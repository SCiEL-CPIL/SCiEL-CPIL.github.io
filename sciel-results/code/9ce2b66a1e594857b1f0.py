"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random


def solve(instance):
    """
    Solve a planted satisfiable 3-SAT instance using WalkSAT with
    the Walk heuristic and a restart strategy.
    """
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)

    # Pre-process: store literals as (var_index, is_positive) for fast access
    # var_index is 0-based
    processed_clauses = []
    for c in clauses:
        processed_clauses.append([(abs(l) - 1, l > 0) for l in c])

    def try_solve(max_flips, walk_p):
        """Run one WalkSAT attempt with given parameters."""
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Build clause satisfaction counts: for each clause, count satisfied literals
        # sat_count[c] = number of literals in clause c that are satisfied
        m = len(processed_clauses)
        sat_count = [0] * m
        for ci, clause in enumerate(processed_clauses):
            cnt = 0
            for var_idx, is_pos in clause:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_count[ci] = cnt

        # Check if already satisfied
        if all(cnt == len(clause) for cnt, clause in zip(sat_count, processed_clauses)):
            return assignment

        for flip_iter in range(max_flips):
            # Find unsatisfied clauses (sat_count < 3 for 3-CNF, or < len(clause))
            unsat_indices = [ci for ci in range(m) if sat_count[ci] < len(processed_clauses[ci])]
            if not unsat_indices:
                return assignment

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            clause = processed_clauses[ci]

            # Compute gain for each literal in the clause
            # gain = number of other clauses this literal would satisfy if flipped
            # For Walk heuristic: find the literal with max gain
            best_gain = -1
            best_literal_idx = -1
            for li, (var_idx, is_pos) in enumerate(clause):
                current_val = assignment[var_idx]
                # If we flip var_idx, this literal becomes satisfied
                # Count how many other clauses have this literal currently unsatisfied
                # (i.e., would become satisfied)
                gain = 0
                for cj in range(m):
                    if cj == ci:
                        continue
                    for vj, posj in processed_clauses[cj]:
                        if vj == var_idx:
                            # Currently: assignment[var_idx] == posj means satisfied
                            # After flip: not assignment[var_idx] == posj means satisfied
                            # So gain if currently unsatisfied: assignment[var_idx] != posj
                            if assignment[var_idx] != posj:
                                gain += 1
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_literal_idx = li

            # Walk heuristic: with probability walk_p, pick best; else random
            if rng.random() < walk_p:
                chosen_li = best_literal_idx
            else:
                chosen_li = rng.randrange(len(clause))

            var_idx, is_pos = clause[chosen_li]

            # Update sat_count for all clauses containing var_idx
            old_val = assignment[var_idx]
            new_val = not old_val
            for cj in range(m):
                for vj, posj in processed_clauses[cj]:
                    if vj == var_idx:
                        if old_val == posj:
                            # Was satisfied, now unsatisfied
                            sat_count[cj] -= 1
                        else:
                            # Was unsatisfied, now satisfied
                            sat_count[cj] += 1
                        break

            assignment[var_idx] = new_val

        # Return final assignment even if not fully satisfied (best effort)
        return assignment

    # Restart strategy: multiple attempts with increasing budgets
    # For planted 3-SAT with 150 vars, typical clause-to-var ratio ~4.2
    # WalkSAT with p=0.5 typically solves these quickly
    restarts = [
        (5000, 0.5),
        (10000, 0.5),
        (20000, 0.4),
        (20000, 0.6),
        (30000, 0.5),
        (40000, 0.5),
    ]

    best_assignment = None
    best_unsat = float('inf')

    for max_flips, walk_p in restarts:
        result = try_solve(max_flips, walk_p)
        # Count unsatisfied clauses
        unsat_count = 0
        for clause in processed_clauses:
            if not any(result[var_idx] == is_pos for var_idx, is_pos in clause):
                unsat_count += 1
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment if best_assignment is not None else [False] * n
# EVOLVE-BLOCK-END
