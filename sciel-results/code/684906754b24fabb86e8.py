"""Improved solver: fast incremental local search for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time


def _clause_satisfied(clause, assignment):
    """Check if a clause is satisfied by the assignment."""
    for lit in clause:
        if lit > 0:
            if assignment[lit - 1]:
                return True
        else:
            if not assignment[-lit - 1]:
                return True
    return False


def _update_clause_sat(ci, assignment, clauses, clause_sat, var_occ):
    """Update clause_sat[ci] after a flip. Returns new satisfied status."""
    clause_sat[ci] = _clause_satisfied(clauses[ci], assignment)
    return clause_sat[ci]


def _flip_var(v, assignment, clauses, clause_sat, var_occ, unsat_count, unsat_set):
    """
    Flip variable v and incrementally update clause satisfaction.
    Returns new unsat_count and unsat_set.
    """
    assignment[v] = not assignment[v]
    # Update all clauses containing variable v
    for ci in var_occ[v]:
        was_sat = clause_sat[ci]
        now_sat = _clause_satisfied(clauses[ci], assignment)
        if was_sat and not now_sat:
            unsat_count += 1
            unsat_set.add(ci)
        elif not was_sat and now_sat:
            unsat_count -= 1
            unsat_set.discard(ci)
        clause_sat[ci] = now_sat
    return unsat_count, unsat_set


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0  # safety margin before 330s kill

    # Build variable occurrence index: var -> list of clause indices
    var_occ = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            var_occ[v].append(ci)

    rng = random.Random(42)
    best_assignment = None
    best_unsat = m + 1

    # Weighted WalkSAT with incremental updates
    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat = [_clause_satisfied(c, assignment) for c in clauses]
        unsat_set = {ci for ci in range(m) if not clause_sat[ci]}
        unsat_count = len(unsat_set)

        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # Adaptive noise: start low (greedy), increase when stuck
        steps = 0
        max_steps = 5000
        stagnation = 0
        prev_unsat = unsat_count
        noise = 0.3

        while steps < max_steps and time.time() < deadline:
            if unsat_count == 0:
                return assignment
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            # Separate literals into satisfying and non-satisfying
            sat_lits = []
            unsat_lits = []
            for lit in clause:
                v = abs(lit) - 1
                desired = (lit > 0)
                if assignment[v] != desired:
                    sat_lits.append(lit)
                else:
                    unsat_lits.append(lit)

            # Choose which literal to flip
            if sat_lits and rng.random() < (1.0 - noise):
                # Greedy: pick a satisfying literal
                lit = rng.choice(sat_lits)
            else:
                # Random: pick any literal from the clause
                lit = rng.choice(clause)

            v = abs(lit) - 1
            unsat_count, unsat_set = _flip_var(v, assignment, clauses, clause_sat, var_occ, unsat_count, unsat_set)

            steps += 1
            if unsat_count < prev_unsat:
                stagnation = 0
                noise = max(0.1, noise * 0.95)  # decrease noise on progress
            else:
                stagnation += 1
                noise = min(0.8, noise + 0.02)  # increase noise when stuck
            prev_unsat = unsat_count

            # Restart this trial if stuck too long
            if stagnation > 200:
                break

        if time.time() >= deadline:
            break

    # Final polish: try to fix remaining unsat clauses
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        clause_sat = [_clause_satisfied(c, assignment) for c in clauses]
        unsat_set = {ci for ci in range(m) if not clause_sat[ci]}
        unsat_count = len(unsat_set)

        for _ in range(1000):
            if time.time() >= deadline:
                break
            if unsat_count == 0:
                return assignment

            # For each unsat clause, try flipping a literal to satisfy it
            improved = False
            unsat_list = list(unsat_set)
            rng.shuffle(unsat_list)
            for ci in unsat_list:
                clause = clauses[ci]
                # Try each literal that would satisfy this clause
                for lit in clause:
                    v = abs(lit) - 1
                    desired = (lit > 0)
                    if assignment[v] != desired:
                        old = assignment[v]
                        assignment[v] = desired
                        # Check net effect
                        new_unsat_count = unsat_count
                        for c2i in var_occ[v]:
                            was_sat = clause_sat[c2i]
                            now_sat = _clause_satisfied(clauses[c2i], assignment)
                            if was_sat and not now_sat:
                                new_unsat_count += 1
                            elif not was_sat and now_sat:
                                new_unsat_count -= 1
                            clause_sat[c2i] = now_sat

                        if new_unsat_count < unsat_count:
                            unsat_count = new_unsat_count
                            # Update unsat_set
                            for c2i in var_occ[v]:
                                if clause_sat[c2i]:
                                    unsat_set.discard(c2i)
                                else:
                                    unsat_set.add(c2i)
                            improved = True
                            break
                        else:
                            # Revert
                            assignment[v] = old
                            for c2i in var_occ[v]:
                                clause_sat[c2i] = _clause_satisfied(clauses[c2i], assignment)

                if improved:
                    break

            if not improved:
                break

        if unsat_count == 0:
            return assignment

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
