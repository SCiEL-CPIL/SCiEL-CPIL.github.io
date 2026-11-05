"""WalkSAT with clause-weighted heuristic, restarts, and time budget."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline: leave a safety margin (evaluator kills at 330s)
    deadline = time.time() + 280.0

    rng = random.Random(12345)

    # Precompute clause literal indices for speed
    # Each clause is a list of signed 1-based integers
    # Store as (var_idx, polarity) where polarity: True means positive literal
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    def check_all(assignment):
        """Return True if all clauses satisfied."""
        for cv in clause_vars:
            sat = False
            for var_idx, pol in cv:
                if assignment[var_idx] == pol:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        result = []
        for i, cv in enumerate(clause_vars):
            sat = False
            for var_idx, pol in cv:
                if assignment[var_idx] == pol:
                    sat = True
                    break
            if not sat:
                result.append(i)
        return result

    def count_sat_literals(cv, assignment):
        """Count how many literals in clause cv are satisfied."""
        cnt = 0
        for var_idx, pol in cv:
            if assignment[var_idx] == pol:
                cnt += 1
        return cnt

    def count_false_vars(cv, assignment):
        """Count how many variables in cv are set to the wrong value."""
        cnt = 0
        for var_idx, pol in cv:
            if assignment[var_idx] != pol:
                cnt += 1
        return cnt

    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 5000
    max_steps_per_run = 100000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        if check_all(assignment):
            return assignment

        # Track best so far
        uc = unsat_clauses(assignment)
        if len(uc) < best_unsat_count:
            best_unsat_count = len(uc)
            best_assignment = assignment[:]

        # WalkSAT with weighted clause selection
        # Weights: each unsat clause gets weight proportional to 1/(number of false literals)
        # to prefer "critical" clauses
        p_weight = 0.55  # probability to pick a variable from the chosen clause
        p_random = 0.05   # probability to pick a random variable

        for step in range(max_steps_per_run):
            if time.time() > deadline:
                break

            uc = unsat_clauses(assignment)
            if not uc:
                return assignment

            if len(uc) < best_unsat_count:
                best_unsat_count = len(uc)
                best_assignment = assignment[:]

            # Weighted clause selection: prefer clauses with fewer false literals (more critical)
            # Use soft-min-like weighting: weight = 1 / (num_false + 1)
            weights = []
            total_w = 0.0
            for ci in uc:
                nf = count_false_vars(clause_vars[ci], assignment)
                w = 1.0 / (nf + 1.0)
                weights.append(w)
                total_w += w

            # Pick clause
            r = rng.random() * total_w
            cum = 0.0
            chosen_clause_idx = uc[0]
            for j, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_clause_idx = uc[j]
                    break

            cv = clause_vars[chosen_clause_idx]

            # Pick variable to flip
            r2 = rng.random()
            if r2 < p_random:
                # Random variable
                var_idx = rng.randint(0, n - 1)
            else:
                # Weighted pick from clause: prefer variables that are currently false
                # (flipping them satisfies this clause) with some noise
                var_weights = []
                total_vw = 0.0
                for var_idx2, pol in cv:
                    if assignment[var_idx2] != pol:
                        # This variable is false in the clause; flipping helps
                        vw = 3.0
                    else:
                        # This variable is true; flipping hurts
                        vw = 1.0
                    var_weights.append(vw)
                    total_vw += vw
                r3 = rng.random() * total_vw
                cum2 = 0.0
                var_idx = cv[0][0]
                for j, vw in enumerate(var_weights):
                    cum2 += vw
                    if r3 <= cum2:
                        var_idx = cv[j][0]
                        break

            assignment[var_idx] = not assignment[var_idx]

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
