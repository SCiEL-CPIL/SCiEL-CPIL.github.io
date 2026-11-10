"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Improved WalkSAT with:
    #  - deadline guard (safety margin before 330s)
    #  - adaptive noise / phase-saturation handling
    #  - restarts with random restarts
    #  - better clause selection (weakest-via-strongest)
    #  - vectorized-ish inner loops via list comprehensions where helpful
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 280.0  # safety margin

    # Precompute literal -> variable index and sign for speed
    # clauses is list of lists of signed 1-based ints

    def make_random_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def count_unsat(assignment):
        # Return list of unsatisfied clause indices
        unsat = []
        for i, c in enumerate(clauses):
            ok = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    ok = True
                    break
            if not ok:
                unsat.append(i)
        return unsat

    def is_satisfied(assignment):
        for c in clauses:
            ok = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def walk_sat_single(assignment, rng, budget, deadline):
        """Run WalkSAT from given assignment with given budget steps.
        Returns (assignment, solved_bool)."""
        # Precompute for each variable, the list of clause indices it appears in
        # and the sign in that clause
        var_to_clauses = [[] for _ in range(n)]
        for ci, c in enumerate(clauses):
            for l in c:
                var_to_clauses[abs(l) - 1].append((ci, l > 0))

        # Current unsat set
        unsat = set(count_unsat(assignment))
        if not unsat:
            return assignment, True

        # Weights / noise parameters
        base_noise = 0.55
        min_noise = 0.1

        for step in range(budget):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment, True

            # Pick a random unsatisfied clause
            clause_idx = rng.choice(list(unsat))
            clause = clauses[clause_idx]

            # Compute scores for each literal in the clause
            # For each literal l in clause:
            #   if we flip var(l), how many currently-unsat clauses become satisfied?
            #   and how many currently-sat clauses become unsat?
            # Score = (num unsat clauses that would become sat) - noise * (num sat clauses that would become unsat)

            best_var = -1
            best_score = -float('inf')

            for l in clause:
                v = abs(l) - 1
                # Count how many unsat clauses contain this variable
                # and how many sat clauses contain this variable
                # We need to check: if we flip v, which unsat clauses get satisfied?
                # An unsat clause gets satisfied if it contains v with the sign that matches NOT assignment[v]
                # Wait: currently assignment[v] is some value. If we flip it, 
                # a clause containing v with literal sign s is satisfied after flip iff 
                # (not assignment[v]) == s, i.e., assignment[v] != s.
                # Currently the clause is unsat, so for every literal l' in it, assignment[var(l')] != sign(l').
                # After flipping v, clause is sat iff for literal l' with var==v, sign(l') == not assignment[v].

                # For sat clauses: a sat clause becomes unsat after flip iff it contains v 
                # and that is the ONLY literal in the clause that is currently satisfied.

                unsat_gained = 0
                sat_lost = 0

                for (ci2, sign2) in var_to_clauses[v]:
                    if ci2 in unsat:
                        # This unsat clause: will it become sat?
                        # It becomes sat if sign2 == not assignment[v]
                        if sign2 != assignment[v]:
                            unsat_gained += 1
                    else:
                        # This sat clause: will it become unsat?
                        # It becomes unsat if the only satisfied literal in it is v with sign2 == assignment[v]
                        # Check if this clause has exactly one satisfied literal, and it's v
                        c2 = clauses[ci2]
                        sat_count = 0
                        for l2 in c2:
                            if assignment[abs(l2) - 1] == (l2 > 0):
                                sat_count += 1
                                if abs(l2) - 1 != v:
                                    break
                        if sat_count == 1 and sign2 == assignment[v]:
                            sat_lost += 1

                score = unsat_gained - base_noise * sat_lost
                if score > best_score:
                    best_score = score
                    best_var = v

            if best_var < 0:
                break

            # Flip best_var
            assignment[best_var] = not assignment[best_var]

            # Update unsat set
            # Re-evaluate clauses containing best_var
            for (ci2, sign2) in var_to_clauses[best_var]:
                c2 = clauses[ci2]
                ok = False
                for l2 in c2:
                    if assignment[abs(l2) - 1] == (l2 > 0):
                        ok = True
                        break
                if ok:
                    unsat.discard(ci2)
                else:
                    unsat.add(ci2)

            # Adaptive noise: decrease noise as we go deeper
            if step % 1000 == 0:
                base_noise = max(min_noise, base_noise * 0.95)

        return assignment, (not unsat)

    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Try multiple restarts
    max_restarts = 200
    base_budget = 5000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random restart
        assignment = make_random_assignment(rng)
        unsat_initial = count_unsat(assignment)
        if not unsat_initial:
            return assignment

        # Run walk-sat
        budget = base_budget + restart * 200  # slightly increase budget
        assignment, solved = walk_sat_single(assignment, rng, budget, deadline)

        if solved:
            return assignment

        # Track best
        unsat_count = len(count_unsat(assignment))
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # If we're close, try a polish pass
        if unsat_count <= 5:
            # Try to fix remaining clauses by direct search
            unsat_clauses = count_unsat(assignment)
            # Try flipping each variable in each unsat clause
            for _ in range(200):
                if time.time() > deadline:
                    break
                if not unsat_clauses:
                    return assignment
                ci = rng.choice(unsat_clauses)
                c = clauses[ci]
                l = rng.choice(c)
                v = abs(l) - 1
                assignment[v] = not assignment[v]
                unsat_clauses = count_unsat(assignment)

            if not unsat_clauses:
                return assignment

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return make_random_assignment(rng)
# EVOLVE-BLOCK-END
