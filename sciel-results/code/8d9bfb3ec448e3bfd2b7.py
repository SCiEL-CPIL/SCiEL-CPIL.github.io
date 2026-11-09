"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Return a satisfying assignment (list of booleans) for a planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans of length n.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: evaluator kills at 330s wall-clock; keep a safety margin.
    start = time.time()
    deadline = start + 290.0  # 40s safety margin

    # Precompute literal arrays (0-based variable index, sign +1/-1).
    lit_var = np.empty((m, 3), dtype=np.int32)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            lit_var[ci, li] = abs(lit) - 1
            lit_sign[ci, li] = 1 if lit > 0 else -1

    # Per-variable clause index list for fast delta updates.
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        seen = set()
        for li in range(3):
            v = int(lit_var[ci, li])
            if v not in seen:
                var_clauses[v].append(ci)
                seen.add(v)

    rng = random.Random(12345)

    def evaluate(assignment):
        """Vectorized: return (unsat_mask, unsat_count, clause_sat) for a boolean assignment array."""
        # assignment: bool array length n
        # A literal is true iff (assignment[var] == (sign > 0))
        vals = assignment[lit_var]  # shape (m,3)
        want = lit_sign > 0        # shape (m,3)
        lit_true = vals == want    # shape (m,3)
        clause_sat = lit_true.any(axis=1)  # shape (m,)
        unsat_mask = ~clause_sat
        return unsat_mask, int(unsat_mask.sum()), clause_sat

    def flip_var(var, assignment, unsat_mask, clause_sat):
        """Flip variable `var`, update unsat_mask/clause_sat in place. Return (new_unsat_count, old_unsat_count)."""
        old_unsat = int(unsat_mask.sum())
        new_val = not assignment[var]
        # Update the value used by clauses containing this var.
        # For each clause containing var, recompute its satisfaction.
        # We maintain clause_sat; recompute for affected clauses.
        affected = var_clauses[var]
        if not affected:
            assignment[var] = new_val
            return old_unsat, old_unsat
        # Recompute satisfaction for affected clauses using current assignment.
        for ci in affected:
            sat = False
            for li in range(3):
                v = int(lit_var[ci, li])
                s = int(lit_sign[ci, li])
                if (assignment[v] == (s > 0)):
                    sat = True
                    break
            clause_sat[ci] = sat
        assignment[var] = new_val
        unsat_mask[:] = ~clause_sat
        new_unsat = int(unsat_mask.sum())
        return new_unsat, old_unsat

    best_assignment = None
    best_unsat = None

    # Adaptive restarts with growing step budgets.
    restart = 0
    while time.time() < deadline:
        seed = rng.randint(0, 2**31 - 1)
        r2 = random.Random(seed)
        assignment = np.array([r2.random() < 0.5 for _ in range(n)], dtype=bool)

        unsat_mask, unsat_count, clause_sat = evaluate(assignment)
        if unsat_count == 0:
            return assignment.tolist()

        if best_unsat is None or unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # Step budget grows with restart count.
        step_budget = 400 + 120 * restart
        # Cap so a single restart can't blow the whole deadline.
        remaining = deadline - time.time()
        # Heuristic: ~ each step is cheap; allow up to ~ remaining seconds worth.
        max_steps_by_time = int(remaining * 4000)  # generous cap
        step_budget = min(step_budget, max_steps_by_time)

        for _ in range(step_budget):
            if time.time() >= deadline:
                break
            # Pick a random unsatisfied clause.
            unsat_idx = np.nonzero(unsat_mask)[0]
            if unsat_idx.size == 0:
                return assignment.tolist()
            ci = int(unsat_idx[r2.randrange(unsat_idx.size)])

            # Candidate literals in this clause.
            best_var = None
            best_cost = None
            for li in range(3):
                v = int(lit_var[ci, li])
                # Cost: number of currently-unsatisfied clauses this variable appears in.
                cost = int(unsat_mask[var_clauses[v]].sum())
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_var = v
            # Randomized tie-break: with some probability, pick a random literal instead.
            if r2.random() < 0.35:
                li = r2.randrange(3)
                best_var = int(lit_var[ci, li])

            new_unsat, old_unsat = flip_var(best_var, assignment, unsat_mask, clause_sat)
            if new_unsat == 0:
                return assignment.tolist()
            if new_unsat < best_unsat:
                best_unsat = new_unsat
                best_assignment = assignment.copy()

        restart += 1

    # Return best assignment found (may or may not be fully satisfied).
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
