"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280  # safety margin before 300s wall-clock

    # Precompute for each variable: list of (clause_index, literal) where variable appears
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            var_clauses[vi].append((ci, lit))

    m = len(clauses)

    def count_violated(assignment):
        """Count violated clauses using precomputed structure."""
        violated = 0
        for clause in clauses:
            ok = False
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    ok = True
                    break
            if not ok:
                violated += 1
        return violated

    def solve_walksat(assignment, max_steps):
        """WalkSAT with greedy + random flip selection."""
        # Build initial violated set
        violated_set = set()
        for ci, clause in enumerate(clauses):
            ok = False
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    ok = True
                    break
            if not ok:
                violated_set.add(ci)

        if not violated_set:
            return True

        p_greedy = 0.5  # probability of greedy flip

        for step in range(max_steps):
            if time.time() > deadline:
                return False
            if not violated_set:
                return True

            # Pick a random violated clause
            ci = rng.choice(list(violated_set))
            clause = clauses[ci]

            # Compute flip gains for each literal in the clause
            best_gain = -1
            best_var = -1
            best_is_greedy = False
            candidates = []

            for lit in clause:
                vi = abs(lit) - 1
                # Compute gain if we flip variable vi
                gain = 0
                # Check all clauses containing this variable
                for (cidx, vlit) in var_clauses[vi]:
                    if cidx == ci:
                        # This is the clause we picked; flipping will satisfy it if the literal matches
                        # Actually, the literal in the clause for this var is lit, and we're flipping var
                        # If lit > 0, current assignment is False (since clause is violated), flipping makes it True -> satisfies
                        # If lit < 0, current assignment is True (since clause is violated), flipping makes it False -> satisfies
                        # So flipping always satisfies this clause (since it was violated and this is the only way)
                        # Wait, we need to check if this clause would become satisfied
                        # The clause is currently violated. If we flip vi, does this clause become satisfied?
                        # It becomes satisfied if the literal lit is now true after flip.
                        # Before flip: assignment[vi] == (lit > 0) is False (since clause violated, this literal is false)
                        # After flip: assignment[vi] == (lit > 0) is True
                        # So yes, this clause becomes satisfied.
                        gain += 1  # satisfy this clause
                    else:
                        # Check if this other clause is currently violated
                        # Determine if vlit is currently true
                        vlit_true = assignment[abs(vlit) - 1] == (vlit > 0)
                        if vlit_true:
                            # Clause is currently satisfied (at least this literal is true)
                            # After flip, vlit becomes false. Need to check if clause still has another true literal
                            # This is expensive; approximate by checking if clause was in violated_set
                            if cidx in violated_set:
                                # Clause is violated, so no literal is true. But vlit_true says vlit is true?
                                # Contradiction - shouldn't happen. Skip.
                                pass
                            else:
                                # Clause is satisfied. After flip, vlit becomes false.
                                # Check if any other literal in this clause is true
                                still_sat = False
                                for olit in clauses[cidx]:
                                    if abs(olit) - 1 == vi:
                                        continue  # skip this variable
                                    if assignment[abs(olit) - 1] == (olit > 0):
                                        still_sat = True
                                        break
                                if not still_sat:
                                    gain -= 1  # we break this clause
                        else:
                            # vlit is currently false
                            # After flip, vlit becomes true, so clause becomes satisfied
                            if cidx in violated_set:
                                gain += 1  # satisfy this clause
                            # else: clause already satisfied, no change in count

                candidates.append((gain, vi, lit))

            # Select flip
            if rng.random() < p_greedy:
                # Greedy: pick max gain
                candidates.sort(key=lambda x: -x[0])
                best_gain, best_var, best_lit = candidates[0]
            else:
                # Random from clause
                best_lit = rng.choice(clause)
                best_var = abs(best_lit) - 1
                best_gain = 0

            # Perform flip
            assignment[best_var] = not assignment[best_var]

            # Update violated_set
            for (cidx, vlit) in var_clauses[best_var]:
                # Check if clause is now satisfied
                is_sat = False
                for olit in clauses[cidx]:
                    if assignment[abs(olit) - 1] == (olit > 0):
                        is_sat = True
                        break
                if is_sat:
                    violated_set.discard(cidx)
                else:
                    violated_set.add(cidx)

            if not violated_set:
                return True

        return False

    # Try multiple restarts
    best_assignment = None
    best_violated = m + 1

    for restart in range(50):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try WalkSAT with varying step budgets
        steps = 10000 + rng.randint(0, 5000)
        if solve_walksat(assignment, steps):
            return assignment

        # Track best
        v = count_violated(assignment)
        if v < best_violated:
            best_violated = v
            best_assignment = assignment[:]

    if best_assignment is not None and count_violated(best_assignment) == 0:
        return best_assignment

    # Last resort: return best found or random
    if best_assignment is not None:
        return best_assignment

    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
