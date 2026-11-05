"""Synthetic planted 3-SAT solver (50 variables)."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 280.0

    # Simple DPLL with unit propagation and pure literal elimination
    # Variables are 0-indexed internally, literals are 1-based signed

    def dpll(assignment):
        """assignment: list of None/True/False. Returns None if UNSAT, else assignment."""
        if time.time() > deadline:
            return None

        # Unit propagation
        changed = True
        while changed:
            changed = False
            # Check for conflicts
            for c in clauses:
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] is not None:
                        if assignment[v] == (l > 0):
                            sat = True
                            break
                if not sat:
                    # Check if unit
                    unassigned = []
                    for l in c:
                        v = abs(l) - 1
                        if assignment[v] is None:
                            unassigned.append(l)
                    if len(unassigned) == 0:
                        return None  # Conflict
                    if len(unassigned) == 1:
                        l = unassigned[0]
                        v = abs(l) - 1
                        assignment[v] = (l > 0)
                        changed = True

        # Check if all assigned
        all_assigned = all(a is not None for a in assignment)
        if all_assigned:
            for c in clauses:
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        sat = True
                        break
                if not sat:
                    return None
            return assignment[:]

        # Pure literal elimination
        lit_polarity = {}  # literal -> set of variables where it appears positively/negatively
        for c in clauses:
            for l in c:
                v = abs(l) - 1
                if assignment[v] is None:
                    lit_polarity.setdefault(l, set()).add(v)

        # Find pure literals
        pure = {}
        for l, vars_ in lit_polarity.items():
            v = abs(l) - 1
            # Check if the opposite literal appears
            opposite = -l
            if opposite in lit_polarity:
                # If the variable appears with both polarities, not pure
                # Actually we need to check if the variable appears with the opposite sign
                # in any clause
                pass
            # Simpler: for each unassigned var, check if it appears only with one sign
        for v in range(n):
            if assignment[v] is not None:
                continue
            pos = False
            neg = False
            for c in clauses:
                for l in c:
                    if abs(l) - 1 == v:
                        if l > 0:
                            pos = True
                        else:
                            neg = True
            if pos and not neg:
                assignment[v] = True
            elif neg and not pos:
                assignment[v] = False

        # If still not all assigned, branch
        # Pick variable with highest occurrence in unassigned clauses
        counts = [0] * n
        for c in clauses:
            # Skip satisfied clauses
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] is not None and assignment[v] == (l > 0):
                    sat = True
                    break
            if sat:
                continue
            for l in c:
                v = abs(l) - 1
                if assignment[v] is None:
                    counts[v] += 1

        best_var = -1
        best_count = -1
        for v in range(n):
            if assignment[v] is None and counts[v] > best_count:
                best_count = counts[v]
                best_var = v

        if best_var == -1:
            # All should be assigned, verify
            for c in clauses:
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        sat = True
                        break
                if not sat:
                    return None
            return assignment[:]

        # Branch: try True first
        assignment[best_var] = True
        result = dpll(assignment)
        if result is not None:
            return result

        # Try False
        assignment[best_var] = False
        result = dpll(assignment)
        if result is not None:
            return result

        # Backtrack
        assignment[best_var] = None
        return None

    initial = [None] * n
    result = dpll(initial)
    if result is not None:
        return result

    # Fallback WalkSAT
    rng = random.Random(0)
    wal = [rng.random() < 0.5 for _ in range(n)]

    def satisfied(clause):
        return any(wal[abs(l) - 1] == (l > 0) for l in clause)

    for _ in range(20000):
        if time.time() > deadline:
            break
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        literal = rng.choice(clause)
        wal[abs(literal) - 1] = not wal[abs(literal) - 1]

    return wal

# EVOLVE-BLOCK-END
