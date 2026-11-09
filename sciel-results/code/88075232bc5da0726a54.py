"""Planted 3-SAT solver (50 vars) using DPLL + WalkSAT hybrid with deadline guard."""
import random
import time
import sys

DEADLINE_FRACTION = 0.95  # safety margin


def _walksat(n, clauses, rng, max_steps, deadline_fn):
    """WalkSAT local search. Returns assignment if solved, else None."""
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Precompute clause literal indices for speed
    def is_sat(clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    for _ in range(max_steps):
        if deadline_fn():
            break
        # Find unsatisfied clauses
        unsat = []
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(c)
        if not unsat:
            return assignment

        # Pick a random unsatisfied clause
        clause = rng.choice(unsat)
        # Pick a literal to flip (bias toward making clause true)
        literal = rng.choice(clause)
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]

    # Final check
    for c in clauses:
        sat = False
        for l in c:
            if assignment[abs(l) - 1] == (l > 0):
                sat = True
                break
        if not sat:
            return None
    return assignment


def _dpll(n, clauses, deadline_fn):
    """
    DPLL with unit propagation and pure literal elimination.
    Returns assignment if SAT, None if UNSAT or timeout.
    """
    # assignment: -1 unassigned, 0 False, 1 True
    assignment = [-1] * n
    # active_clauses: list of clauses (each is list of signed literals)
    active_clauses = [list(c) for c in clauses]

    def unit_propagate():
        """Apply unit propagation. Returns False on contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            new_clauses = []
            for clause in active_clauses:
                # Remove satisfied literals
                new_clause = []
                sat = False
                for l in clause:
                    idx = abs(l) - 1
                    val = assignment[idx]
                    if val == 1 and l > 0:
                        sat = True
                        break
                    if val == 0 and l < 0:
                        sat = True
                        break
                    if val == -1:
                        new_clause.append(l)
                if sat:
                    continue
                if len(new_clause) == 0:
                    return False  # Contradiction
                if len(new_clause) == 1:
                    l = new_clause[0]
                    idx = abs(l) - 1
                    assignment[idx] = 1 if l > 0 else 0
                    changed = True
                else:
                    new_clauses.append(new_clause)
            active_clauses = new_clauses
        return True

    def pure_literal_elimination():
        """Apply pure literal elimination. Returns True if any literal was eliminated."""
        # Track which literals appear positively and negatively
        pos = [set() for _ in range(n)]
        neg = [set() for _ in range(n)]
        for clause in active_clauses:
            for l in clause:
                idx = abs(l) - 1
                if l > 0:
                    pos[idx].add(id(clause))
                else:
                    neg[idx].add(id(clause))

        changed = False
        for i in range(n):
            if assignment[i] != -1:
                continue
            if not pos[i] and neg[i]:
                assignment[i] = 0
                changed = True
            elif pos[i] and not neg[i]:
                assignment[i] = 1
                changed = True
        return changed

    # Initial propagation
    if not unit_propagate():
        return None

    # Check if all assigned
    if all(a != -1 for a in assignment):
        if unit_propagate():
            return [a == 1 for a in assignment]
        return None

    # Recursive DPLL
    def _dpll_rec(depth):
        if deadline_fn():
            return None

        # Unit propagation
        if not unit_propagate():
            return None

        # Check if all assigned
        all_assigned = True
        for a in assignment:
            if a == -1:
                all_assigned = False
                break
        if all_assigned:
            return [a == 1 for a in assignment]

        # Pure literal elimination
        if pure_literal_elimination():
            # After pure literal elimination, need to propagate again
            if not unit_propagate():
                return None
            all_assigned = True
            for a in assignment:
                if a == -1:
                    all_assigned = False
                    break
            if all_assigned:
                return [a == 1 for a in assignment]

        # Choose variable: pick one from shortest clause
        var_idx = -1
        min_len = 999
        for clause in active_clauses:
            unassigned = [l for l in clause if assignment[abs(l) - 1] == -1]
            if len(unassigned) < min_len and len(unassigned) > 0:
                min_len = len(unassigned)
                var_idx = abs(unassigned[0]) - 1

        if var_idx == -1:
            return None

        # Try True
        assignment[var_idx] = 1
        result = _dpll_rec(depth + 1)
        if result is not None:
            return result

        # Try False
        assignment[var_idx] = 0
        result = _dpll_rec(depth + 1)
        if result is not None:
            return result

        # Backtrack
        assignment[var_idx] = -1
        return None

    return _dpll_rec(0)


def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance.
    Returns a list of booleans (assignment) satisfying all clauses.
    """
    n = instance["n"]
    clauses = instance["clauses"]

    start_time = time.time()
    deadline = start_time + 330 * DEADLINE_FRACTION

    def deadline_fn():
        return time.time() >= deadline

    # Try DPLL first (exact, deterministic)
    result = _dpll(n, clauses, deadline_fn)
    if result is not None:
        # Verify
        for c in clauses:
            sat = False
            for l in c:
                if result[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                break
        else:
            return result

    # Fallback to WalkSAT with multiple restarts
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    if remaining_time <= 0:
        # Last resort: random assignment
        return [rng.random() < 0.5 for _ in range(n)]

    # WalkSAT with adaptive steps based on remaining time
    max_steps = int(min(500000, remaining_time * 50000))  # ~50k steps per second
    max_steps = max(max_steps, 1000)

    # Multiple restarts
    num_restarts = max(1, int(remaining_time / 5))
    for r in range(num_restarts):
        if deadline_fn():
            break
        steps_per_restart = max(1000, max_steps // num_restarts)
        result = _walksat(n, clauses, rng, steps_per_restart, deadline_fn)
        if result is not None:
            return result

    # If we get here, try one more WalkSAT with all remaining time
    result = _walksat(n, clauses, rng, max_steps, deadline_fn)
    if result is not None:
        return result

    # Final fallback: return last assignment from WalkSAT (may not satisfy)
    # Generate a random assignment as last resort
    return [rng.random() < 0.5 for _ in range(n)]
