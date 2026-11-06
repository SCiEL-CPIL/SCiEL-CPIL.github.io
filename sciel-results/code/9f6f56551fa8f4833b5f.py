"""Stronger planted 3-SAT solver: optimized WalkSAT with precomputed criticality and adaptive restarts."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _build_structures(n, clauses):
    """Precompute literal arrays, var-to-clause mapping, and per-variable clause indices."""
    m = len(clauses)
    if m == 0:
        return None, None, None, None
    lit_arr = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        lit_arr[i, 0] = c[0]
        lit_arr[i, 1] = c[1]
        lit_arr[i, 2] = c[2]
    var_arr = np.abs(lit_arr) - 1  # 0-based variable index

    # Build var_to_clauses: for each variable, list of (clause_idx, position)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for pos in range(3):
            v = var_arr[i, pos]
            var_to_clauses[v].append((i, pos))

    # Precompute for each variable: array of clause indices and positions (for fast iteration)
    var_clause_idx = [np.array([c[0] for c in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_to_clauses]
    var_clause_pos = [np.array([c[1] for c in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_to_clauses]

    return lit_arr, var_arr, var_to_clauses, (var_clause_idx, var_clause_pos)


def _eval_clauses(assignment, lit_arr, var_arr):
    """Return boolean array: is each clause satisfied?"""
    a = assignment[var_arr]  # shape (m, 3), bool
    pos = lit_arr > 0        # shape (m, 3), bool
    sat_literal = a == pos   # shape (m, 3)
    return sat_literal.any(axis=1)  # shape (m,)


def _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v):
    """Incrementally update sat_mask after flipping variable v. Returns new unsat_count."""
    for ci, pos in var_to_clauses[v]:
        # Check if clause is satisfied by any literal
        # Use numpy for speed on the 3-literal check
        clause_vars = var_arr[ci]
        clause_lits = lit_arr[ci]
        clause_assign = assignment[clause_vars]
        clause_pos = clause_lits > 0
        sat_mask[ci] = np.any(clause_assign == clause_pos)
    
    return int((~sat_mask).sum())


def _compute_flip_score(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v):
    """
    Compute exact gain/loss score for flipping variable v.
    Returns: (gain - loss) where gain = # unsat clauses that become sat,
             loss = # sat clauses that become unsat.
    """
    v_is_true = assignment[v]
    new_val = not v_is_true
    
    score = 0.0
    for cj, cpos in var_to_clauses[v]:
        lit_v_cj = lit_arr[cj, cpos]
        v_satisfies_after_flip = (new_val == (lit_v_cj > 0))
        
        if sat_mask[cj]:
            v_satisfies_now = (v_is_true == (lit_v_cj > 0))
            if v_satisfies_now and not v_satisfies_after_flip:
                # Check if any other literal in this clause satisfies it
                other_sat = False
                for p2 in range(3):
                    if p2 == cpos:
                        continue
                    var2 = var_arr[cj, p2]
                    lit2 = lit_arr[cj, p2]
                    if assignment[var2] == (lit2 > 0):
                        other_sat = True
                        break
                if not other_sat:
                    score -= 1.0
        else:
            if v_satisfies_after_flip:
                score += 1.0
    
    return score


def _smart_init(n, clauses, lit_arr, var_arr, var_to_clauses, rng, max_iter=150):
    """
    Smart initialization: start random, then do greedy best-improvement flips.
    """
    m = len(clauses)
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    sat_mask = _eval_clauses(assignment, lit_arr, var_arr)
    unsat_count = int((~sat_mask).sum())
    
    if unsat_count == 0:
        return assignment, sat_mask, unsat_count
    
    for _ in range(max_iter):
        if unsat_count == 0:
            break
        best_var = -1
        best_score = float('-inf')
        
        # Sample random variables that appear in unsat clauses
        unsat_vars = set()
        unsat_indices = np.where(~sat_mask)[0]
        for ci in unsat_indices:
            for pos in range(3):
                unsat_vars.add(int(var_arr[ci, pos]))
        
        if not unsat_vars:
            break
        
        candidates = list(unsat_vars)
        if len(candidates) > 80:
            candidates = rng.sample(candidates, 80)
        
        for v in candidates:
            score = _compute_flip_score(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
            if score > best_score:
                best_score = score
                best_var = v
        
        if best_var == -1 or best_score <= 0:
            break
        
        v = best_var
        assignment[v] = not assignment[v]
        unsat_count = _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
    
    return assignment, sat_mask, unsat_count


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    DEADLINE = 295.0

    lit_arr, var_arr, var_to_clauses, _ = _build_structures(n, clauses)

    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(42)

    restart = 0
    while time.time() - start_time < DEADLINE:
        restart += 1

        # Vary max_steps and p-range per restart for diversity
        base_steps = 30000 + 20000 * (restart % 3)
        max_steps = base_steps
        p_min = 0.1 + 0.05 * (restart % 2)
        p_max = 0.85 + 0.05 * (restart % 2)

        # Smart initial assignment
        assignment, sat_mask, unsat_count = _smart_init(n, clauses, lit_arr, var_arr, var_to_clauses, rng)

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        unsat_indices = np.where(~sat_mask)[0]

        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Adaptive p: ramp up based on step
            progress = step / max_steps
            p = p_min + (p_max - p_min) * progress

            # Pick an unsatisfied clause (weighted toward clauses with fewer satisfied literals for diversity)
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause_vars = var_arr[ci]

            if rng.random() < p:
                pos = rng.randrange(3)
            else:
                # Exact gain/loss scoring for each variable in the clause
                best_pos = 0
                best_score = float('-inf')
                for pos in range(3):
                    v = int(clause_vars[pos])
                    score = _compute_flip_score(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                pos = best_pos

            v = int(clause_vars[pos])
            assignment[v] = not assignment[v]

            # Incremental update
            unsat_count = _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
            unsat_indices = np.where(~sat_mask)[0]

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

        if best_unsat == 0:
            return best_assignment.tolist()

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
