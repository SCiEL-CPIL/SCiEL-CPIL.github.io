# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    # Preprocess: convert to 0-based, build numpy arrays for speed
    # clause_vars: shape (m, 3) with 0-based variable indices
    # clause_signs: shape (m, 3) with +1 or -1
    m = len(clauses)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    def check_all(assign):
        """Return array of booleans: is each clause satisfied?"""
        # assign: bool array of length n
        # For each clause, check if any literal is satisfied
        # lit is satisfied if (assign[var] and sign==1) or (not assign[var] and sign==-1)
        vals = assign[clause_vars]  # shape (m, 3)
        # satisfied if vals == (signs > 0) elementwise
        return np.any(vals == (clause_signs > 0), axis=1)

    def unsat_clauses(assign):
        sat = check_all(assign)
        return np.where(~sat)[0]

    rng = random.Random(42)

    best_assign = None
    best_unsat_count = m + 1

    # WalkSAT with restarts
    max_restarts = 200
    base_steps = 5000
    step_decay = 1.05

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Random initial assignment
        assign = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        # Check if already satisfied
        sat = check_all(assign)
        if sat.all():
            return assign.tolist()

        steps = int(base_steps * (step_decay ** restart))
        # Cap steps to avoid too long single restart
        steps = min(steps, 50000)

        for step in range(steps):
            if time.time() > deadline - 1.0:
                break

            unsat_idx = unsat_clauses(assign)
            unsat_count = len(unsat_idx)
            if unsat_count == 0:
                return assign.tolist()

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assign = assign.copy()

            # Choose an unsatisfied clause (weighted by number of false literals)
            # For 3-SAT, each unsat clause has 0, 1, or 2 true literals (0 true = all false)
            # Prefer clauses with fewer true literals (more "promising" to fix)
            clause_idx = unsat_idx[rng.randint(0, unsat_count - 1)]
            clause = clause_idx

            # Get the three literals in this clause
            c_vars = clause_vars[clause]
            c_signs = clause_signs[clause]
            c_vals = assign[c_vars]

            # A literal is satisfied if c_vals[i] == (c_signs[i] > 0)
            lit_sat = c_vals == (c_signs > 0)

            # Count how many are currently satisfied
            num_sat = np.sum(lit_sat)

            if num_sat == 0:
                # All three are false - flip one randomly
                pos = rng.randint(0, 2)
            elif num_sat == 1:
                # One is true, two are false - prefer flipping a false one (min-conflicts)
                false_positions = np.where(~lit_sat)[0]
                pos = false_positions[rng.randint(0, len(false_positions) - 1)]
            else:
                # Two or more are true but clause still unsat? Shouldn't happen for 3-SAT
                # But if it does, flip a true one
                true_positions = np.where(lit_sat)[0]
                pos = true_positions[rng.randint(0, len(true_positions) - 1)]

            # Flip the chosen variable
            assign[c_vars[pos]] = not assign[c_vars[pos]]

        # End of restart's steps

    # Return best found
    if best_assign is not None:
        return best_assign.tolist()
    return assign.tolist()
# EVOLVE-BLOCK-END
