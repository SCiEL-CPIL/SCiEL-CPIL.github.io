"""Planted 3-SAT solver (50 vars): DPLL + unit propagation + pure literal + WalkSAT fallback."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # ---------- DPLL with unit propagation and pure literal elimination ----------
    # We represent the formula as a list of clauses (each clause is a list of signed literals).
    # assignment: dict var -> bool (True/False), or None if unassigned.

    def dpll(clauses, assignment, depth=0):
        """
        clauses: list of lists of signed ints (1-based)
        assignment: dict int -> bool
        Returns: dict int -> bool (satisfying assignment for all n vars) or None
        """
        if time.time() > deadline:
            return None

        # Remove satisfied clauses; simplify unit clauses
        new_clauses = []
        unit_clauses = []
        for clause in clauses:
            sat = False
            simplified = []
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    val = assignment[var]
                    is_true = (lit > 0) == val
                    if is_true:
                        sat = True
                        break
                    # else literal is false, drop it
                else:
                    simplified.append(lit)
            if sat:
                continue
            if len(simplified) == 0:
                # Empty clause: contradiction
                return None
            if len(simplified) == 1:
                unit_clauses.append(simplified)
            else:
                new_clauses.append(simplified)

        # Unit propagation
        while unit_clauses:
            unit = unit_clauses.pop()
            lit = unit[0]
            var = abs(lit)
            val = (lit > 0)
            if var in assignment:
                if assignment[var] != val:
                    return None  # conflict
                continue
            assignment[var] = val
            # Re-simplify all clauses
            still_units = []
            new_clauses2 = []
            for clause in new_clauses:
                sat = False
                simplified = []
                for l in clause:
                    v = abs(l)
                    if v in assignment:
                        if (l > 0) == assignment[v]:
                            sat = True
                            break
                        # false, drop
                    else:
                        simplified.append(l)
                if sat:
                    continue
                if len(simplified) == 0:
                    return None
                if len(simplified) == 1:
                    still_units.append(simplified)
                else:
                    new_clauses2.append(simplified)
            new_clauses = new_clauses2
            unit_clauses = still_units

        if not new_clauses:
            # All clauses satisfied; fill in remaining vars arbitrarily
            for v in range(1, n + 1):
                if v not in assignment:
                    assignment[v] = True
            return assignment

        # Pure literal elimination
        # Count occurrences of each literal sign
        pos_count = [0] * (n + 1)
        neg_count = [0] * (n + 1)
        for clause in new_clauses:
            for lit in clause:
                var = abs(lit)
                if lit > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1

        changed = True
        while changed:
            changed = False
            for var in range(1, n + 1):
                if var in assignment:
                    continue
                if pos_count[var] == 0 and neg_count[var] > 0:
                    # Only appears negated -> set to False
                    assignment[var] = False
                    changed = True
                    # Re-simplify
                    new_clauses = _simplify(new_clauses, assignment, n)
                    if new_clauses is None:
                        return None
                    break
                elif neg_count[var] == 0 and pos_count[var] > 0:
                    # Only appears positively -> set to True
                    assignment[var] = True
                    changed = True
                    new_clauses = _simplify(new_clauses, assignment, n)
                    if new_clauses is None:
                        return None
                    break
            if changed:
                # Recompute counts
                pos_count = [0] * (n + 1)
                neg_count = [0] * (n + 1)
                for clause in new_clauses:
                    for lit in clause:
                        var = abs(lit)
                        if lit > 0:
                            pos_count[var] += 1
                        else:
                            neg_count[var] += 1

        if not new_clauses:
            for v in range(1, n + 1):
                if v not in assignment:
                    assignment[v] = True
            return assignment

        # Variable selection: pick the variable in the shortest clause
        # Clause length heuristic
        min_len = len(new_clauses[0])
        best_clause = new_clauses[0]
        for c in new_clauses[1:]:
            if len(c) < min_len:
                min_len = len(c)
                best_clause = c
                if min_len == 1:
                    break

        # Pick a literal from the best clause (deterministic: first one)
        lit = best_clause[0]
        var = abs(lit)

        # Try True first, then False
        for val in [True, False]:
            if var in assignment:
                if assignment[var] == val:
                    continue
                else:
                    break  # already assigned to opposite
            assignment[var] = val
            new_clauses2 = _simplify(new_clauses, assignment, n)
            if new_clauses2 is None:
                del assignment[var]
                continue
            result = dpll(new_clauses2, assignment, depth + 1)
            if result is not None:
                return result
            del assignment[var]

        return None

    def _simplify(clauses, assignment, n):
        """Simplify clauses given assignment. Returns None on contradiction."""
        result = []
        for clause in clauses:
            sat = False
            simplified = []
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    if (lit > 0) == assignment[var]:
                        sat = True
                        break
                else:
                    simplified.append(lit)
            if sat:
                continue
            if len(simplified) == 0:
                return None
            result.append(simplified)
        return result

    # Run DPLL
    assignment_dict = {}
    result = dpll(list(clauses), assignment_dict, 0)

    if result is not None:
        # Convert to list of booleans
        return [result.get(v, False) for v in range(1, n + 1)]

    # ---------- Fallback: WalkSAT with multiple restarts ----------
    rng = random.Random(12345)
    best_assignment = None
    best_unsat_count = float('inf')

    for restart in range(20):
        if time.time() > deadline:
            break
        assignment = [rng.random() < 0.5 for _ in range(n)]
        for step in range(50000):
            if time.time() > deadline:
                break
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
            # Track best
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment[:]
            clause = rng.choice(unsat)
            literal = rng.choice(clause)
            assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]

    if best_assignment is not None:
        return best_assignment

    # Last resort: return random assignment
    return [False] * n

# EVOLVE-BLOCK-END
