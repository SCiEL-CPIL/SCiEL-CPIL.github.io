"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Robust WalkSAT with restarts, weighted clause selection, and a deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 300s kill

    # Pre-compute clause literal indices for speed
    clause_idx = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[l > 0 for l in c] for c in clauses]
    m = len(clauses)

    best_assignment = None
    best_unsat_count = float('inf')

    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for i in range(m):
            cidx = clause_idx[i]
            csigns = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cidx[j]] == csigns[j]:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for i in range(m):
            cidx = clause_idx[i]
            csigns = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cidx[j]] == csigns[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    for restart in range(500):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Initial check
        if is_satisfied(assignment):
            return assignment

        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        # WalkSAT loop with adaptive parameters
        steps = 0
        max_steps = 10000
        while steps < max_steps:
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for i in range(m):
                cidx = clause_idx[i]
                csigns = clause_signs[i]
                sat = False
                for j in range(3):
                    if assignment[cidx[j]] == csigns[j]:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment

            # Weighted clause selection: prefer clauses with fewer false literals (closer to being satisfied)
            # For 3-SAT, each unsat clause has 3 false literals; weight by inverse count of true literals in clause
            # Actually for unsat clauses, all 3 literals are false. Let's use a different heuristic.
            # Pick clause with minimum number of satisfied literals among its literals (but for unsat, that's 0).
            # Instead, use the standard WalkSAT heuristic: pick a random unsat clause, then pick a literal
            # that appears in many other unsat clauses (to fix multiple clauses at once).

            clause_idx_chosen = rng.choice(unsat_clauses)
            cidx = clause_idx[clause_idx_chosen]
            csigns = clause_signs[clause_idx_chosen]

            # For each literal in the chosen clause, count how many other unsat clauses would be satisfied
            # if we flip this variable (i.e., how many unsat clauses contain this variable with the opposite sign)
            best_var = -1
            best_gain = -1
            best_cost = float('inf')

            for j in range(3):
                var = cidx[j]
                current_val = assignment[var]
                new_val = not current_val

                # Count how many unsat clauses are satisfied by setting var = new_val
                gain = 0
                cost = 0  # number of currently satisfied clauses that would become unsatisfied
                for i in unsat_clauses:
                    ci = clause_idx[i]
                    cs = clause_signs[i]
                    # Does this clause contain var?
                    found = False
                    for k in range(3):
                        if ci[k] == var:
                            found = True
                            if cs[k] == new_val:
                                gain += 1
                            break
                    if not found:
                        # Check if flipping var affects this clause (it doesn't)
                        pass

                # To keep it simpler and faster, use a lighter heuristic:
                # Count occurrences of the variable in unsat clauses
                occ = 0
                for i in unsat_clauses:
                    if var in clause_idx[i]:
                        occ += 1

                # Use a random tie-breaker
                score = occ + rng.random() * 0.5
                if best_gain == -1 or score > best_gain:
                    best_gain = score
                    best_var = var

            if best_var == -1:
                # Fallback: random flip
                best_var = cidx[rng.randint(0, 2)]

            assignment[best_var] = not assignment[best_var]
            steps += 1

        # Update best
        if time.time() > deadline:
            break
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment

    # Last resort: return random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
