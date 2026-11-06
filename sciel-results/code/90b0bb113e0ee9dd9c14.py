# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit

    m = len(clauses)

    # Convert clauses to 0-based signed literals
    # For each clause, store (var_idx, sign) where sign is +1 for positive, -1 for negative
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    # Build data structures for efficient clause evaluation
    # For each variable, store list of (clause_idx, sign) where it appears
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # Precompute clause sizes
    clause_sizes = [len(c) for c in clauses_0based]

    # Helper: compute unsatisfied clause count and which clauses are unsat
    def _compute_unsat(assignment):
        clause_unsat = np.zeros(m, dtype=bool)
        for ci, clause in enumerate(clauses_0based):
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat
        return clause_unsat

    # Helper: verify if assignment satisfies all clauses
    def _is_satisfied(assignment):
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Strategy 1: Unit propagation + WalkSAT hybrid
    def _unit_propagation_walksat(rng, max_restarts=100, steps_per_restart=10000):
        """Unit propagation to simplify, then WalkSAT on the simplified problem."""
        best_assignment = None
        best_unsat_count = m + 1

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initial assignment
            assignment = [rng.random() < 0.5 for _ in range(n)]

            # Apply unit propagation-like simplification
            # Find clauses that are unit (only one unassigned literal) and force them
            # Actually, let's do a more direct approach: identify forced assignments
            # For planted SAT, many variables may be determined by the structure

            # Compute initial unsatisfied clause count
            clause_unsat = _compute_unsat(assignment)
            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

            # WalkSAT loop with adaptive noise
            noise = 0.5  # Initial noise level
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_assignment

                if unsat_count == 0:
                    return assignment

                # Pick a random unsatisfied clause
                unsat_indices = np.where(clause_unsat)[0]
                ci = int(rng.choice(unsat_indices))
                clause = clauses_0based[ci]

                # Count how many literals in this clause are satisfied
                # and identify which variable to flip
                # Use a scoring heuristic: prefer flipping a variable that
                # satisfies this clause and breaks few other unsatisfied clauses

                best_var = -1
                best_score = -float('inf')

                for var, sign in clause:
                    # Score: number of other clauses this flip would break minus satisfy
                    score = 0
                    for other_ci, other_sign in var_clauses[var]:
                        if other_ci == ci:
                            continue
                        # Would this flip satisfy or break the other clause?
                        if clause_unsat[other_ci]:
                            # Flipping might satisfy this clause
                            # Check if this flip would make the clause satisfied
                            # The literal in other clause for var has sign other_sign
                            # After flip, assignment[var] = not assignment[var]
                            # Satisfied if not assignment[var] == (other_sign > 0)
                            # Which is assignment[var] != (other_sign > 0)
                            if assignment[var] != (other_sign > 0):
                                score += 1
                        else:
                            # Other clause is satisfied. Would flipping break it?
                            # Check if var is the only satisfying literal
                            only_sat = True
                            for v2, s2 in clauses_0based[other_ci]:
                                if v2 == var:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                score -= 1

                    # Add random noise for exploration
                    score += rng.random() * noise

                    if score > best_score:
                        best_score = score
                        best_var = var

                if best_var == -1:
                    break

                # Flip the variable
                old_val = assignment[best_var]
                assignment[best_var] = not old_val

                # Update clause satisfaction
                for other_ci, other_sign in var_clauses[best_var]:
                    other_clause = clauses_0based[other_ci]
                    # Recompute if this clause is satisfied
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat

                if unsat_count == 0:
                    return assignment

                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]

                # Adaptive noise: reduce noise if we're making progress
                if step % 1000 == 0 and step > 0:
                    noise = max(0.1, noise * 0.9)

        return best_assignment if best_assignment is not None else [False] * n

    # Strategy 2: Spectral initialization + local search
    def _spectral_init_local_search(rng, max_iters=20000):
        """Use spectral methods to get a good initial assignment, then local search."""
        # Build incidence matrix for spectral analysis
        # For each clause, create a row indicating which variables appear
        # Use a simple approach: count variable frequencies and use that to bias initialization

        # Count variable appearances
        var_freq = np.zeros(n)
        for clause in clauses_0based:
            for var, sign in clause:
                var_freq[var] += 1

        # Normalize
        if var_freq.max() > 0:
            var_freq_norm = var_freq / var_freq.max()
        else:
            var_freq_norm = var_freq

        # Initialize assignment biased by frequency
        # Variables that appear more often are more "important" - try both polarities
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try a few different initialization strategies and pick the best
        best_init = assignment[:]
        best_init_unsat = m + 1

        for init_strategy in range(5):
            if init_strategy == 0:
                # Random
                candidate = [rng.random() < 0.5 for _ in range(n)]
            elif init_strategy == 1:
                # All true
                candidate = [True] * n
            elif init_strategy == 2:
                # All false
                candidate = [False] * n
            elif init_strategy == 3:
                # Based on frequency: high frequency -> true
                threshold = np.median(var_freq_norm)
                candidate = [var_freq_norm[v] > threshold for v in range(n)]
            else:
                # Random with bias towards frequent variables being true
                candidate = []
                for v in range(n):
                    p = 0.3 + 0.4 * var_freq_norm[v]
                    candidate.append(rng.random() < p)

            clause_unsat_c = _compute_unsat(candidate)
            unsat_c = int(np.sum(clause_unsat_c))
            if unsat_c < best_init_unsat:
                best_init_unsat = unsat_c
                best_init = candidate[:]

        assignment = best_init[:]
        clause_unsat = _compute_unsat(assignment)
        unsat_count = best_init_unsat

        if unsat_count == 0:
            return assignment

        best_assignment = assignment[:]
        best_unsat = unsat_count

        # Local search with tabu
        tabu = [0] * n
        tabu_len = 30

        for step in range(max_iters):
            if time.time() > deadline:
                return best_assignment

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick random unsatisfied clause
            unsat_indices = np.where(clause_unsat)[0]
            if len(unsat_indices) == 0:
                return assignment
            ci = int(rng.choice(unsat_indices))
            clause = clauses_0based[ci]

            # Find best flip among variables in this clause
            best_var = -1
            best_delta = -float('inf')

            for var, sign in clause:
                if step < tabu[var]:
                    continue

                # Compute delta: change in unsat count if we flip var
                delta = 0
                for other_ci, other_sign in var_clauses[var]:
                    other_clause = clauses_0based[other_ci]
                    if clause_unsat[other_ci]:
                        # Flipping might satisfy this clause
                        if assignment[var] != (other_sign > 0):
                            delta -= 1
                    else:
                        # Clause is sat. Flipping might break it if var is the only sat literal
                        only_sat = True
                        for v2, s2 in other_clause:
                            if v2 == var:
                                continue
                            if assignment[v2] == (s2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            delta += 1

                # Add noise
                delta += rng.random() * 0.3

                if delta < best_delta:
                    best_delta = delta
                    best_var = var

            if best_var == -1:
                # All variables tabu, pick randomly from clause
                var, sign = clause[rng.randrange(len(clause))]
                best_var = var

            # Flip
            assignment[best_var] = not assignment[best_var]
            tabu[best_var] = step + tabu_len

            # Update clause states
            for other_ci, other_sign in var_clauses[best_var]:
                other_clause = clauses_0based[other_ci]
                sat = False
                for v, s in other_clause:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                was_unsat = clause_unsat[other_ci]
                is_unsat = not sat
                if was_unsat and not is_unsat:
                    unsat_count -= 1
                elif not was_unsat and is_unsat:
                    unsat_count += 1
                clause_unsat[other_ci] = is_unsat

        return best_assignment

    # Strategy 3: Max-2-SAT reduction with spectral solution
    def _max2sat_spectral(rng):
        """
        Transform 3-CNF to weighted Max-2-SAT by introducing auxiliary variables.
        For clause (l1 ∨ l2 ∨ l3), introduce aux var a such that:
        (l1 ∨ l2 ∨ a) ∧ (l3 ∨ ¬a) is equivalent to (l1 ∨ l2 ∨ l3)
        Then solve the resulting 2-CNF using spectral methods.
        """
        # For each 3-clause, we can split it into two 2-clauses with an auxiliary variable
        # But this increases the problem size. Instead, let's use a different approach:
        # Use the fact that for planted SAT, the solution has a specific structure.
        # Try to identify the planted solution using frequency analysis.

        # Count how often each literal appears
        lit_count = Counter()
        for clause in clauses_0based:
            for var, sign in clause:
                lit_count[(var, sign)] += 1

        # For each variable, compute the balance between positive and negative occurrences
        var_balance = np.zeros(n)
        for (var, sign), count in lit_count.items():
            if sign > 0:
                var_balance[var] += count
            else:
                var_balance[var] -= count

        # The planted solution likely assigns variables based on their balance
        # If a variable appears more often positively, it's likely true in the solution
        assignment = [var_balance[v] >= 0 for v in range(n)]

        # Refine with local search
        clause_unsat = _compute_unsat(assignment)
        unsat_count = int(np.sum(clause_unsat))

        if unsat_count == 0:
            return assignment

        best_assignment = assignment[:]
        best_unsat = unsat_count

        # Simple local search refinement
        for step in range(10000):
            if time.time() > deadline:
                return best_assignment

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Pick random unsatisfied clause
            unsat_indices = np.where(clause_unsat)[0]
            if len(unsat_indices) == 0:
                return assignment
            ci = int(rng.choice(unsat_indices))
            clause = clauses_0based[ci]

            # Find best flip
            best_var = -1
            best_delta = -float('inf')

            for var, sign in clause:
                delta = 0
                for other_ci, other_sign in var_clauses[var]:
                    other_clause = clauses_0based[other_ci]
                    if clause_unsat[other_ci]:
                        if assignment[var] != (other_sign > 0):
                            delta -= 1
                    else:
                        only_sat = True
                        for v2, s2 in other_clause:
                            if v2 == var:
                                continue
                            if assignment[v2] == (s2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            delta += 1

                delta += rng.random() * 0.2

                if delta < best_delta:
                    best_delta = delta
                    best_var = var

            if best_var == -1:
                var, sign = clause[rng.randrange(len(clause))]
                best_var = var

            assignment[best_var] = not assignment[best_var]

            for other_ci, other_sign in var_clauses[best_var]:
                other_clause = clauses_0based[other_ci]
                sat = False
                for v, s in other_clause:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                was_unsat = clause_unsat[other_ci]
                is_unsat = not sat
                if was_unsat and not is_unsat:
                    unsat_count -= 1
                elif not was_unsat and is_unsat:
                    unsat_count += 1
                clause_unsat[other_ci] = is_unsat

        return best_assignment

    # Strategy 4: DPLL-lite with random restarts
    def _dpll_lite(rng, max_restarts=50):
        """Simple DPLL with random variable ordering and restarts."""
        best_assignment = None
        best_unsat_count = m + 1

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random variable ordering
            var_order = list(range(n))
            rng.shuffle(var_order)

            # Try to assign variables in this order, using unit propagation
            assignment = [None] * n  # None means unassigned

            # Simple backtracking search with depth limit
            def _search(depth, current_unsat):
                nonlocal best_assignment, best_unsat_count

                if time.time() > deadline:
                    return False

                if depth > 50:  # Limit search depth
                    return False

                # Check if all variables are assigned
                all_assigned = all(a is not None for a in assignment)
                if all_assigned:
                    if current_unsat == 0:
                        best_assignment = assignment[:]
                        best_unsat_count = 0
                        return True
                    return False

                # Find a variable to assign
                # Prefer variables that appear in unsatisfied clauses
                var_to_assign = None
                for v in var_order:
                    if assignment[v] is None:
                        # Check if this variable appears in any unsatisfied clause
                        for ci, sign in var_clauses[v]:
                            # Check if clause is unsatisfied
                            clause = clauses_0based[ci]
                            all_false = True
                            for v2, s2 in clause:
                                if assignment[v2] is None:
                                    all_false = False
                                    break
                                if assignment[v2] == (s2 > 0):
                                    all_false = False
                                    break
                            if all_false:
                                var_to_assign = v
                                break
                        if var_to_assign is not None:
                            break

                if var_to_assign is None:
                    # Pick first unassigned variable
                    for v in var_order:
                        if assignment[v] is None:
                            var_to_assign = v
                            break

                if var_to_assign is None:
                    return False

                # Try both values
                for val in [True, False]:
                    assignment[var_to_assign] = val

                    # Compute new unsat count
                    new_unsat = 0
                    for clause in clauses_0based:
                        sat = False
                        all_assigned_in_clause = True
                        for v2, s2 in clause:
                            if assignment[v2] is None:
                                all_assigned_in_clause = False
                                break
                            if assignment[v2] == (s2 > 0):
                                sat = True
                                break
                        if not sat and all_assigned_in_clause:
                            new_unsat += 1

                    if new_unsat == 0:
                        best_assignment = assignment[:]
                        best_unsat_count = 0
                        return True

                    if _search(depth + 1, new_unsat):
                        return True

                    assignment[var_to_assign] = None

                return False

            if _search(0, m):
                if best_unsat_count == 0:
                    return best_assignment

        return best_assignment if best_assignment is not None else [False] * n

    # Try multiple strategies and return the best result
    rng = random.Random(42)

    # Strategy 1: WalkSAT with restarts
    result = _unit_propagation_walksat(rng, max_restarts=50, steps_per_restart=10000)
    if _is_satisfied(result):
        return result

    # Strategy 2: Spectral initialization + local search
    if time.time() < deadline - 20:
        result2 = _spectral_init_local_search(random.Random(123), max_iters=15000)
        if _is_satisfied(result2):
            return result2

    # Strategy 3: Max-2-SAT spectral
    if time.time() < deadline - 15:
        result3 = _max2sat_spectral(random.Random(777))
        if _is_satisfied(result3):
            return result3

    # Strategy 4: DPLL-lite
    if time.time() < deadline - 10:
        result4 = _dpll_lite(random.Random(999), max_restarts=30)
        if _is_satisfied(result4):
            return result4

    # Return the best result found (least unsatisfied clauses)
    best_result = result
    best_unsat = m + 1

    for r in [result, result2, result3, result4]:
        if r is None:
            continue
        unsat = 0
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if r[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                unsat += 1
        if unsat < best_unsat:
            best_unsat = unsat
            best_result = r

    return best_result
# EVOLVE-BLOCK-END
