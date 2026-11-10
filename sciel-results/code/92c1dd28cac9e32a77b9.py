"""Planted 3-SAT solver (50 vars). WalkSAT + phase-saving + vectorized tracking + polish."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np


def _init_phase(clauses, n):
    """Initial assignment from literal phase counts (phase-saving heuristic)."""
    pos = np.zeros(n, dtype=np.int32)
    neg = np.zeros(n, dtype=np.int32)
    for cl in clauses:
        for lit in cl:
            if lit > 0:
                pos[lit - 1] += 1
            else:
                neg[-lit - 1] += 1
    return pos >= neg  # True where positive phase dominates


def _build_clauses(clauses, n):
    """Vectorized clause representation: rows = literals, cols = (idx, sign)."""
    total = sum(len(c) for c in clauses)
    rows = np.empty(total, dtype=np.int32)
    signs = np.empty(total, dtype=np.int8)
    r = 0
    for cl in clauses:
        for lit in cl:
            rows[r] = abs(lit) - 1
            signs[r] = 1 if lit > 0 else 0
            r += 1
    return rows, signs


def _clause_state(rows, signs, assignment):
    """Return per-clause (num_true, num_false) as (m,2) int32 array."""
    vals = np.where(signs == 1, assignment, ~assignment).astype(np.int8)
    num_true = np.bincount(vals, minlength=2)
    m = len(num_true)
    num_false = m - num_true
    return np.stack([num_true, num_false], axis=1)


def _satisfied_any(clause, assignment):
    return any(assignment[abs(l) - 1] == (l > 0) for l in clause)


def _satisfied_all(clauses, assignment):
    for c in clauses:
        if not any(assignment[abs(l) - 1] == (l > 0) for l in c):
            return False
    return True


def _walksat_step(unsat_idx, clauses, rows, signs, assignment, rng, weights,
                  num_true, num_false, n):
    """One WalkSAT step. Returns (flipped_var, new_unsat_count, new_num_true, new_num_false)."""
    k = rng.randint(0, len(unsat_idx) - 1)
    cidx = unsat_idx[k]
    # Gather literal rows for this clause
    lit_rows = []
    for lit in clauses[cidx]:
        lit_rows.append(lit)
    # Choose literal: weighted by impact (vars in many unsat clauses) + phase bias
    # weight = number of unsat clauses containing the variable (impact)
    # We precompute per-variable unsat incidence via rows.
    # For simplicity: pick literal by impact of its variable.
    # Compute impact for each literal's variable:
    # count how many unsat clauses contain that variable.
    # We'll compute quickly using rows restricted to unsat clauses.
    # Build a set of (clause) indices; but we need per-variable count.
    # Use a dict: for each unsat clause, for each lit, var = abs(lit)-1.
    var_count = {}
    for ci in unsat_idx:
        for lit in clauses[ci]:
            v = abs(lit) - 1
            var_count[v] = var_count.get(v, 0) + 1
    # Choose literal: weight = var_count[var] * (1 if phase matches else 0.3)
    total_w = 0.0
    weights_local = []
    for lit in lit_rows:
        v = abs(lit) - 1
        w = var_count.get(v, 1)
        # phase-saving: prefer flipping to match the variable's phase
        phase = assignment[v]  # current
        # We don't have global phase here; use weights (precomputed phase)
        weights_local.append(w)
        total_w += w
    # Weighted random pick
    r = rng.random() * total_w
    acc = 0.0
    chosen = 0
    for i, w in enumerate(weights_local):
        acc += w
        if r <= acc:
            chosen = i
            break
    lit = lit_rows[chosen]
    v = abs(lit) - 1
    # Compute new unsat count efficiently
    # Flipping v changes truth value of v.
    # For each clause containing v, the num_true/num_false changes.
    # We need to find which clauses contain v.
    # Precompute variable -> list of (clause_idx, row_idx, sign) for speed.
    # We'll pass in a precomputed structure.
    # For now, do a scan (n=50, small).
    new_num_true = num_true.copy()
    new_num_false = num_false.copy()
    new_unsat_count = 0
    # Find all clauses containing variable v
    # We can use rows to find row indices where rows == v
    # But rows is per-literal; we need clause mapping.
    # Let's precompute var_to_clause_rows: var -> list of (clause_idx, row_idx)
    # We'll pass this in.
    # For efficiency, do a direct scan of clauses.
    for ci, cl in enumerate(clauses):
        found = False
        for lit2 in cl:
            if abs(lit2) - 1 == v:
                found = True
                break
        if not found:
            continue
        # This clause contains v. Update its num_true/num_false.
        # Before flip: clause state was (num_true[ci], num_false[ci]).
        # The literal for v in this clause: determine sign.
        sign_v = 1
        for lit2 in cl:
            if abs(lit2) - 1 == v:
                sign_v = 1 if lit2 > 0 else 0
                break
        # Current value of this literal:
        cur_val = assignment[v] if sign_v == 1 else (not assignment[v])
        # After flip:
        new_val = not cur_val
        if cur_val and not new_val:
            # true -> false: num_true -= 1, num_false += 1
            new_num_true[ci] -= 1
            new_num_false[ci] += 1
        elif (not cur_val) and new_val:
            # false -> true: num_true += 1, num_false -= 1
            new_num_true[ci] += 1
            new_num_false[ci] -= 1
        # Recompute unsat for this clause
        if new_num_true[ci] == 0:
            new_unsat_count += 1
    # Apply flip
    assignment[v] = not assignment[v]
    return v, new_unsat_count, new_num_true, new_num_false


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 270.0  # safety margin before 300s
    rng = random.Random(0)

    # Build vectorized structures
    rows, signs = _build_clauses(clauses, n)

    # Precompute variable -> list of (clause_idx, sign) for fast updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_to_clauses[v].append((ci, sign))

    # Phase-saving initial assignment
    assignment = _init_phase(clauses, n)
    assignment = np.array(assignment, dtype=bool)

    # Compute initial clause states
    num_true, num_false = _clause_state(rows, signs, assignment)
    unsat_mask = num_true == 0
    unsat_count = int(np.sum(unsat_mask))
    best_unsat = unsat_count
    best_assignment = assignment.copy()

    # If already satisfied
    if unsat_count == 0:
        return assignment.tolist()

    # WalkSAT with restarts
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        # Reset or perturb
        if restart == 0:
            pass
        else:
            # Perturb: flip a few random variables
            for _ in range(rng.randint(1, 3)):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
            num_true, num_false = _clause_state(rows, signs, assignment)
            unsat_mask = num_true == 0
            unsat_count = int(np.sum(unsat_mask))

        if unsat_count == 0:
            return assignment.tolist()

        # WalkSAT loop
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return assignment.tolist()
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

            # Get unsat clause indices
            unsat_idx = np.where(unsat_mask)[0]
            if len(unsat_idx) == 0:
                return assignment.tolist()

            # Choose clause
            cidx = unsat_idx[rng.randint(0, len(unsat_idx) - 1)]
            cl = clauses[cidx]

            # Weighted literal selection: weight = number of unsat clauses containing var
            # Compute quickly using var_to_clauses
            var_unsat_count = np.zeros(n, dtype=np.int32)
            for ci in unsat_idx:
                for lit in clauses[ci]:
                    var_unsat_count[abs(lit) - 1] += 1

            # Build weights for literals in chosen clause
            weights = []
            for lit in cl:
                v = abs(lit) - 1
                w = float(var_unsat_count[v]) + 1.0
                weights.append(w)
            total_w = sum(weights)
            r = rng.random() * total_w
            acc = 0.0
            chosen = 0
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    chosen = i
                    break
            lit = cl[chosen]
            v = abs(lit) - 1

            # Compute effect of flipping v
            new_num_true = num_true.copy()
            new_num_false = num_false.copy()
            # Update all clauses containing v
            for ci, sign_v in var_to_clauses[v]:
                cur_val = assignment[v] if sign_v == 1 else (not assignment[v])
                new_val = not cur_val
                if cur_val and not new_val:
                    new_num_true[ci] -= 1
                    new_num_false[ci] += 1
                elif (not cur_val) and new_val:
                    new_num_true[ci] += 1
                    new_num_false[ci] -= 1

            new_unsat_count = int(np.sum(new_num_true == 0))

            # Accept if it doesn't worsen too much (WalkSAT always accepts)
            # Apply flip
            assignment[v] = not assignment[v]
            num_true = new_num_true
            num_false = new_num_false
            unsat_mask = num_true == 0
            unsat_count = new_unsat_count

        # End of WalkSAT loop for this restart

    # Final polish: try to reduce remaining unsat with targeted flips
    if time.time() < deadline - 5.0:
        # High-precision polish: try flipping each variable and keep if it helps
        for polish_iter in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for v in range(n):
                if time.time() > deadline - 5.0:
                    break
                # Try flipping v
                old_val = assignment[v]
                assignment[v] = not old_val
                num_true_new, num_false_new = _clause_state(rows, signs, assignment)
                unsat_new = int(np.sum(num_true_new == 0))
                if unsat_new < unsat_count:
                    num_true = num_true_new
                    num_false = num_false_new
                    unsat_count = unsat_new
                    unsat_mask = num_true == 0
                    improved = True
                    if unsat_count == 0:
                        return assignment.tolist()
                else:
                    assignment[v] = old_val
            if not improved:
                break

    # Return best found
    if best_unsat < unsat_count:
        return best_assignment.tolist()
    return assignment.tolist()
# EVOLVE-BLOCK-END
