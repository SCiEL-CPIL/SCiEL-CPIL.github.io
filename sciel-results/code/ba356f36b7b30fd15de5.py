"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Hybrid: DPLL with unit propagation + WalkSAT fallback, with deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 300s kill
    rng = random.Random(0)

    # Convert to 0-based signed literals for internal use
    def to_internal(clause):
        return [(abs(l) - 1, l > 0) for l in clause]

    internal_clauses = [to_internal(c) for c in clauses]

    # DPLL with unit propagation and pure literal elimination
    def dpll(clauses, assignment, var_order):
        """Returns (success, assignment) or (None, None) if timeout."""
        if time.time() > deadline:
            return None, None

        # Work on a copy of assignment: -1=unset, 0=False, 1=True
        assign = assignment[:]
        clauses = clauses[:]

        def propagate(clauses, assign):
            """Unit propagation. Returns (success, clauses, assign, conflict)."""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    sat = False
                    unassigned = []
                    for var, pos in clause:
                        if assign[var] == 1:
                            if pos:
                                sat = True
                                break
                            # negated literal is false, skip
                        elif assign[var] == 0:
                            if not pos:
                                sat = True
                                break
                        else:
                            unassigned.append((var, pos))
                    if sat:
                        continue
                    if not unassigned:
                        return False, None, None, True  # conflict
                    if len(unassigned) == 1:
                        var, pos = unassigned[0]
                        assign[var] = 1 if pos else 0
                        changed = True
                        # Don't add unit clause to new_clauses, it's satisfied
                    else:
                        new_clauses.append(unassigned)
            return True, new_clauses, assign, False

        def pure_literal_elim(clauses, assign):
            """Pure literal elimination. Returns (clauses, assign)."""
            if not clauses:
                return clauses, assign
            # Count occurrences of each literal
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] != -1:
                        continue
                    if pos:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1

            # Find pure literals
            pure = {}
            for var in range(n):
                if assign[var] != -1:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    pure[var] = 1
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    pure[var] = 0

            if not pure:
                return clauses, assign

            # Apply pure literals
            for var, val in pure.items():
                assign[var] = val

            # Filter clauses
            new_clauses = []
            for clause in clauses:
                sat = False
                remaining = []
                for var, pos in clause:
                    if assign[var] == 1:
                        if pos:
                            sat = True
                            break
                    elif assign[var] == 0:
                        if not pos:
                            sat = True
                            break
                    else:
                        remaining.append((var, pos))
                if sat:
                    continue
                if not remaining:
                    return None, None  # conflict
                new_clauses.append(remaining)
            return new_clauses, assign

        # Main DPLL loop
        while True:
            if time.time() > deadline:
                return None, None

            # Unit propagation
            ok, clauses, assign, conflict = propagate(clauses, assign)
            if conflict:
                return False, None, None, None
            if not clauses:
                return True, assign, None, None

            # Pure literal elimination
            clauses, assign = pure_literal_elim(clauses, assign)
            if clauses is None:
                return False, None, None, None
            if not clauses:
                return True, assign, None, None

            # Find a variable to branch on
            # Choose variable appearing in most clauses
            var_freq = {}
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] == -1:
                        var_freq[var] = var_freq.get(var, 0) + 1

            if not var_freq:
                return True, assign, None, None

            var = max(var_freq, key=lambda v: var_freq[v])

            # Try True first, then False
            for val in [1, 0]:
                new_assign = assign[:]
                new_assign[var] = val
                new_clauses = []
                conflict_found = False
                for clause in clauses:
                    sat = False
                    remaining = []
                    for v, pos in clause:
                        if v == var:
                            if (val == 1 and pos) or (val == 0 and not pos):
                                sat = True
                                break
                        elif new_assign[v] == -1:
                            remaining.append((v, pos))
                        else:
                            # Already assigned
                            if (new_assign[v] == 1 and pos) or (new_assign[v] == 0 and not pos):
                                sat = True
                                break
                    if sat:
                        continue
                    if not remaining:
                        conflict_found = True
                        break
                    new_clauses.append(remaining)

                if not conflict_found:
                    result, new_assign2, _, _ = dpll(new_clauses, new_assign, var_order)
                    if result is not None:
                        return result, new_assign2, None, None
                    if result is False:
                        continue  # This branch failed, try other value
                    # Timeout
                    return None, None, None, None
                # Conflict, try other value

            # Both branches failed
            return False, None, None, None

    # Initial assignment: all unset (-1)
    initial_assign = [-1] * n

    # Try DPLL first
    success, dpll_assign, _, _ = dpll(internal_clauses, initial_assign, list(range(n)))

    if success and dpll_assign is not None:
        # Convert back to boolean list
        result = []
        for v in range(n):
            if dpll_assign[v] == -1:
                result.append(rng.random() < 0.5)
            else:
                result.append(dpll_assign[v] == 1)
        return result

    # If DPLL timed out or failed, fall back to WalkSAT
    # Use a good initial assignment: random
    assignment = [rng.random() < 0.5 for _ in range(n)]

    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)

    def count_unsat(clause):
        return sum(1 for l in clause if not (assignment[abs(l) - 1] == (l > 0)))

    # WalkSAT with restarts and deadline
    max_restarts = 50
    flips_per_restart = 10000
    noise = 0.25

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random restart
        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        for _ in range(flips_per_restart):
            if time.time() > deadline:
                return assignment

            unsat = [c for c in clauses if not satisfied(c)]
            if not unsat:
                return assignment

            # Pick clause with minimum unsatisfied literals (most nearly satisfied)
            clause = min(unsat, key=count_unsat)

            # Score each literal: number of other clauses it would satisfy
            best_lit = None
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                # Count how many clauses this flip would satisfy
                score = 0
                for c in clauses:
                    if satisfied(c):
                        continue
                    # Check if flipping this variable would satisfy clause c
                    would_satisfy = False
                    for ll in c:
                        if abs(ll) - 1 == var:
                            # This clause contains the variable
                            if (ll > 0 and not assignment[var]) or (ll < 0 and assignment[var]):
                                # Flipping would satisfy this literal
                                would_satisfy = True
                            break
                    if would_satisfy:
                        score += 1

                # Add noise
                score += rng.random() * noise * len(clauses)

                if score > best_score:
                    best_score = score
                    best_lit = l

            if best_lit is not None:
                assignment[abs(best_lit) - 1] = not assignment[abs(best_lit) - 1]

    return assignment
# EVOLVE-BLOCK-END
