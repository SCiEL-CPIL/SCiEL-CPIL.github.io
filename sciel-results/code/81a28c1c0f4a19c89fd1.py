"""Planted 3-SAT solver: incremental-count WalkSAT + clause weights + SA escape + repair."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]

    # 0-based clause representation: list of (var_idx, is_pos)
    clause_literals = []
    for c in clauses_raw:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_literals.append(cv)

    m = len(clause_literals)

    # variable -> list of (clause_idx, is_pos_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_literals):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((i, is_pos))

    rng = random.Random(42)
    deadline = time.monotonic() + 290.0

    def init_clause_counts(assignment):
        counts = [0] * m
        for i, cv in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts

    def belief_propagation_init(num_iterations=100):
        var_bias = [0.0] * n
        for it in range(num_iterations):
            new_var_bias = [0.0] * n
            for c_idx, cv in enumerate(clause_literals):
                for j, (var_idx, is_pos) in enumerate(cv):
                    prod_complement = 1.0
                    for k, (v2, p2) in enumerate(cv):
                        if k == j:
                            continue
                        bias = var_bias[v2]
                        if p2:
                            # need P(v2 = False)
                            if bias > 0:
                                p_false = 1.0 / (1.0 + math.exp(bias))
                            else:
                                p_false = math.exp(bias) / (1.0 + math.exp(bias))
                        else:
                            # need P(v2 = True)
                            if bias > 0:
                                p_true = 1.0 / (1.0 + math.exp(-bias))
                            else:
                                p_true = 1.0 / (1.0 + math.exp(bias))
                            p_false = p_true
                        if p_false > 1.0:
                            p_false = 1.0
                        elif p_false < 0.0:
                            p_false = 0.0
                        prod_complement *= p_false
                    p_sat_given_false = 1.0 - prod_complement
                    if p_sat_given_false < 1e-15:
                        p_sat_given_false = 1e-15
                    msg = -math.log(p_sat_given_false)
                    if is_pos:
                        new_var_bias[var_idx] += msg
                    else:
                        new_var_bias[var_idx] -= msg
            var_bias = new_var_bias
        return [bias > 0 for bias in var_bias]

    def weighted_walksat(assignment, clause_weights, max_steps, sa_temp=1.0):
        counts = init_clause_counts(assignment)
        unsat_set = set(i for i in range(m) if counts[i] == 0)

        # Precompute total weight for fast weighted selection
        # We'll maintain running total of unsat weights
        for step in range(max_steps):
            if time.monotonic() > deadline - 2.0:
                return False, assignment, unsat_set, counts

            if not unsat_set:
                return True, assignment, unsat_set, counts

            # Weighted clause selection
            total_w = 0.0
            for ci in unsat_set:
                total_w += clause_weights[ci]
            if total_w <= 0:
                c_idx = next(iter(unsat_set))
            else:
                r = rng.random() * total_w
                acc = 0.0
                c_idx = None
                for ci in unsat_set:
                    acc += clause_weights[ci]
                    if r <= acc:
                        c_idx = ci
                        break
                if c_idx is None:
                    c_idx = next(iter(unsat_set))

            cv = clause_literals[c_idx]

            # Score each variable in the clause: weighted gain - weighted loss
            best_var = None
            best_score = float('-inf')

            for j, (var_idx, is_pos) in enumerate(cv):
                # Flipping var_idx: currently assignment[var_idx] != is_pos (since clause is unsat,
                # none of its literals are satisfied, so all are currently "wrong")
                # After flip, var_idx becomes is_pos, satisfying this clause.
                # Gain: unsat clauses containing var_idx that become satisfied
                # Loss: sat clauses containing var_idx that become unsatisfied
                gain = 0.0
                loss = 0.0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if counts[c_aff] == 0:
                        # Unsat clause: will it become satisfied?
                        # After flip, var_idx = not assignment[var_idx].
                        # The literal in c_aff is satisfied if assignment[var_idx] == is_pos_aff after flip.
                        # Current: assignment[var_idx] != is_pos_aff (since clause is unsat, all literals unsat)
                        # After flip: new_val = not assignment[var_idx]
                        # We need new_val == is_pos_aff
                        new_val = not assignment[var_idx]
                        if new_val == is_pos_aff:
                            gain += clause_weights[c_aff]
                    else:
                        # Satisfied clause: will it become unsatisfied?
                        # Only if the var_idx literal is the ONLY satisfying literal.
                        # Check other literals in c_aff.
                        cv_aff = clause_literals[c_aff]
                        still_sat = False
                        for k2, (v2, p2) in enumerate(cv_aff):
                            if v2 == var_idx:
                                continue
                            if assignment[v2] == p2:
                                still_sat = True
                                break
                        if not still_sat:
                            # This clause would become unsat if we flip var_idx
                            # (the var_idx literal is currently the only satisfying one)
                            loss += clause_weights[c_aff]

                score = gain - 1.0 * loss
                # Add small random tiebreak
                score += rng.random() * 0.3

                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var is None:
                best_var = cv[0][0]

            var_idx = best_var
            old_unsat = len(unsat_set)

            # Flip variable
            assignment[var_idx] = not assignment[var_idx]

            # Update counts incrementally
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)

            new_unsat = len(unsat_set)

            # SA: accept worse moves with decreasing probability
            if new_unsat > old_unsat:
                delta = new_unsat - old_unsat
                T = sa_temp * max(0.01, 1.0 - step / max_steps)
                if rng.random() > math.exp(-delta / max(T, 0.001)):
                    # Reject: flip back
                    assignment[var_idx] = not assignment[var_idx]
                    for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                        if assignment[var_idx] == is_pos_aff:
                            if counts[c_aff] == 0:
                                unsat_set.discard(c_aff)
                            counts[c_aff] += 1
                        else:
                            counts[c_aff] -= 1
                            if counts[c_aff] == 0:
                                unsat_set.add(c_aff)

            # Update clause weights: bump the selected clause if still unsat
            if c_idx in unsat_set:
                clause_weights[c_idx] *= 1.05
            else:
                # Clause got satisfied, decay its weight slightly
                clause_weights[c_idx] *= 0.95

            # Periodic normalization to prevent overflow
            if step % 500 == 0:
                max_w = max(clause_weights) if clause_weights else 1.0
                if max_w > 0:
                    clause_weights = [w / max_w for w in clause_weights]

        return False, assignment, unsat_set, counts

    def sa_escape(assignment, clause_weights, steps, t0):
        counts = init_clause_counts(assignment)
        unsat_set = set(i for i in range(m) if counts[i] == 0)

        for step in range(steps):
            if time.monotonic() > deadline - 2.0:
                return assignment, unsat_set, counts

            if not unsat_set:
                return assignment, unsat_set, counts

            # Pick a random unsat clause and a random variable in it
            c_idx = next(iter(unsat_set))
            cv = clause_literals[c_idx]
            var_idx = cv[rng.randrange(len(cv))][0]

            old_unsat = len(unsat_set)

            # Flip
            assignment[var_idx] = not assignment[var_idx]
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)

            new_unsat = len(unsat_set)
            delta = new_unsat - old_unsat

            if delta > 0:
                if rng.random() >= math.exp(-delta / max(t0, 1e-9)):
                    # Reject
                    assignment[var_idx] = not assignment[var_idx]
                    for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                        if assignment[var_idx] == is_pos_aff:
                            if counts[c_aff] == 0:
                                unsat_set.discard(c_aff)
                            counts[c_aff] += 1
                        else:
                            counts[c_aff] -= 1
                            if counts[c_aff] == 0:
                                unsat_set.add(c_aff)

            t0 *= 0.9995

        return assignment, unsat_set, counts

    def greedy_repair(assignment, max_flips=2000):
        """Try to reduce unsat count to zero by greedily flipping the variable
        that gives the best net improvement in unsat count."""
        counts = init_clause_counts(assignment)
        unsat_set = set(i for i in range(m) if counts[i] == 0)

        for _ in range(max_flips):
            if time.monotonic() > deadline - 1.0:
                break
            if not unsat_set:
                return assignment, unsat_set, counts, True

            # For each variable, compute the net change in unsat count if flipped
            # Only consider variables that appear in at least one unsat clause
            candidate_vars = set()
            for ci in unsat_set:
                for var_idx, is_pos in clause_literals[ci]:
                    candidate_vars.add(var_idx)

            best_var = None
            best_delta = 0  # We want to minimize unsat, so best_delta < 0 is good

            for var_idx in candidate_vars:
                # Simulate flip
                new_val = not assignment[var_idx]
                delta = 0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if new_val == is_pos_aff:
                        # Literal becomes true
                        if counts[c_aff] == 0:
                            delta -= 1  # unsat -> sat
                    else:
                        # Literal becomes false
                        if counts[c_aff] == 1:
                            delta += 1  # sat -> unsat
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx

            if best_var is None:
                break  # No improving flip found

            var_idx = best_var
            assignment[var_idx] = not assignment[var_idx]
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)

        solved = len(unsat_set) == 0
        return assignment, unsat_set, counts, solved

    best_assignment = [False] * n
    best_unsat_count = m + 1

    restart_budget = 2000
    base_steps = 5000

    for restart in range(restart_budget):
        if time.monotonic() > deadline - 5.0:
            break

        # Initialize
        if restart < 5:
            assignment = belief_propagation_init(num_iterations=50 + restart * 10)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        clause_weights = [1.0] * m
        sa_temp = 0.5 + rng.random() * 1.5

        remaining = deadline - time.monotonic()
        if remaining < 10.0:
            break
        elif remaining < 30.0:
            steps = 2000
        elif remaining < 60.0:
            steps = 4000
        else:
            steps = base_steps

        solved, assignment, unsat_set, counts = weighted_walksat(
            assignment, clause_weights, steps, sa_temp)

        if solved:
            return assignment

        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # SA escape from current state
        if time.monotonic() < deadline - 5.0 and current_unsat > 0:
            a2 = assignment[:]
            w2 = clause_weights[:]
            a2, unsat2, counts2 = sa_escape(a2, w2, 1000, 5.0)
            if len(unsat2) == 0:
                return a2
            if len(unsat2) < best_unsat_count:
                best_unsat_count = len(unsat2)
                best_assignment = a2[:]

    # Final: greedy repair from best assignment
    if best_unsat_count < m and time.monotonic() < deadline - 2.0:
        a = best_assignment[:]
        a, unsat_set, counts, solved = greedy_repair(a, max_flips=3000)
        if solved:
            return a
        if len(unsat_set) < best_unsat_count:
            best_unsat_count = len(unsat_set)
            best_assignment = a[:]

    return best_assignment
# EVOLVE-BLOCK-END
