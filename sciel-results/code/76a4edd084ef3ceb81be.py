"""Planted 3-SAT solver: vectorized smooth WalkSAT with deadline guard."""
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    C = len(clauses)
    if C == 0:
        return [True] * n

    # Build literal -> (var, sign) and per-variable clause incidence
    # var_clauses[v] = list of (clause_idx, sign) where sign=+1 if var positive in clause
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v].append((ci, sign))

    # Soft assignment: 1.0 = True, 0.0 = False, 0.5 = unassigned (for soft cost)
    # We work with hard booleans for the actual assignment but use soft cost for guidance.
    # Assignment as int array: 0 = False, 1 = True
    rng = np.random.default_rng(42)
    assign = rng.integers(0, 2, size=n)

    # clause_satisfied_count[c] = number of satisfied literals in clause c (0..3)
    clause_sat = np.zeros(C, dtype=np.float64)

    def compute_clause_sat():
        """Compute satisfied literal count per clause from current assignment."""
        for ci, clause in enumerate(clauses):
            cnt = 0.0
            for lit in clause:
                v = abs(lit) - 1
                is_sat = (assign[v] == 1) if lit > 0 else (assign[v] == 0)
                if is_sat:
                    cnt += 1.0
            clause_sat[ci] = cnt

    compute_clause_sat()

    # Total soft score: sum over clauses of min(clause_sat, 1) -- but we want to maximize
    # satisfied clauses. Use: score = sum(1 if clause_sat>=1 else 0)
    # For smooth guidance, use soft: sum(clause_sat) but cap contribution per clause at 1
    # Actually for flip selection, we want to maximize the number of satisfied clauses.
    # Soft surrogate: for each clause, contribution = min(clause_sat, 1.0)
    # Total = sum of contributions. Flipping a var changes only clauses it's in.

    def total_soft_score():
        return np.minimum(clause_sat, 1.0).sum()

    best_assign = assign.copy()
    best_score = total_soft_score()
    best_unsat = int(C - (clause_sat >= 1.0).sum())

    # Precompute: for each clause, which vars are in it and their signs
    # clause_vars[ci] = list of (var, sign)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
        clause_vars.append(cv)

    step = 0
    max_steps = 5000000  # hard cap, but deadline will trigger first

    while time.time() < deadline and step < max_steps:
        # Check if solved
        unsat_mask = clause_sat < 1.0
        if not unsat_mask.any():
            break

        # Adaptive noise: start low, increase
        noise_prob = 0.15 + 0.25 * min(1.0, step / 200000.0)

        if rng.random() < noise_prob:
            # Random restart: flip a random variable
            v = rng.integers(0, n)
            assign[v] = 1 - assign[v]
            # Update clause_sat for affected clauses
            for ci, sign in var_clauses[v]:
                old = clause_sat[ci]
                # Recompute this clause
                cnt = 0.0
                for (vv, ss) in clause_vars[ci]:
                    is_sat = (assign[vv] == 1) if ss == 1 else (assign[vv] == 0)
                    if is_sat:
                        cnt += 1.0
                clause_sat[ci] = cnt
            step += 1
            continue

        # Choose an unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        ci = unsat_indices[rng.integers(0, len(unsat_indices))]

        # For each variable in this clause, compute the delta in soft score if flipped
        best_var = -1
        best_delta = -np.inf
        for (v, sign) in clause_vars[ci]:
            # Compute delta if we flip v
            # Old contribution of v to clauses it's in:
            # For each clause c' containing v with sign s':
            #   old_sat = 1 if (assign[v]==1 and s'==1) or (assign[v]==0 and s'==-1) else 0
            #   new_sat = 1 - old_sat
            #   old_contrib = min(old_clause_sat_c, 1.0)
            #   new_contrib = min(old_clause_sat_c - old_sat + new_sat, 1.0)
            #   delta_c = new_contrib - old_contrib
            # Total delta = sum over affected clauses

            delta = 0.0
            for (c_idx, s_prime) in var_clauses[v]:
                old_sat_v = 1.0 if ((assign[v] == 1 and s_prime == 1) or (assign[v] == 0 and s_prime == -1)) else 0.0
                new_sat_v = 1.0 - old_sat_v
                old_contrib = min(clause_sat[c_idx], 1.0)
                new_clause_sat = clause_sat[c_idx] - old_sat_v + new_sat_v
                new_contrib = min(new_clause_sat, 1.0)
                delta += new_contrib - old_contrib

            if delta > best_delta:
                best_delta = delta
                best_var = v

        if best_var >= 0:
            v = best_var
            assign[v] = 1 - assign[v]
            # Update clause_sat
            for (c_idx, s_prime) in var_clauses[v]:
                cnt = 0.0
                for (vv, ss) in clause_vars[c_idx]:
                    is_sat = (assign[vv] == 1) if ss == 1 else (assign[vv] == 0)
                    if is_sat:
                        cnt += 1.0
                clause_sat[c_idx] = cnt
            step += 1

        # Track best
        cur_score = total_soft_score()
        cur_unsat = int(C - (clause_sat >= 1.0).sum())
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_score = cur_score
            best_assign = assign.copy()
            if best_unsat == 0:
                break

    # Return best assignment found
    return [bool(x) for x in best_assign]
