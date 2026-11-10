"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s kill

    # Preprocess: convert clauses to 0-based indices with sign
    # clause_list[i] = list of (var_idx, sign) where sign is +1 or -1
    clause_list = []
    for c in clauses:
        parsed = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            parsed.append((var_idx, sign))
        clause_list.append(parsed)

    m = len(clause_list)

    # Build variable-to-clauses index
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_list):
        for var_idx, sign in c:
            var_clauses[var_idx].append(i)

    best_assignment = None
    best_unsat = m + 1

    # Multiple restarts with WalkSAT variants
    rng = random.Random(42)

    for restart in range(1000):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Compute initial unsat count and per-clause satisfaction status
        clause_sat = [False] * m
        unsat_count = 0
        for i, c in enumerate(clause_list):
            sat = False
            for var_idx, sign in c:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                unsat_count += 1

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # Simulated annealing local search
        max_steps = 5000
        T_start = 3.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps)

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            # Current temperature
            T = T_start * (cooling_rate ** step)

            # Select an unsat clause
            unsat_indices = [i for i in range(m) if not clause_sat[i]]
            if not unsat_indices:
                return assignment

            # Weighted selection: prefer clauses closer to being satisfied
            weights = []
            for i in unsat_indices:
                false_count = 0
                for var_idx, sign in clause_list[i]:
                    if assignment[var_idx] != (sign == 1):
                        false_count += 1
                w = 3 - false_count + 1  # at least 1
                weights.append(w)

            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = unsat_indices[0]
            for j, i in enumerate(unsat_indices):
                cum += weights[j]
                if r <= cum:
                    chosen_idx = i
                    break

            c = clause_list[chosen_idx]

            # Evaluate all possible flips in the clause and pick one via SA
            best_var = None
            best_delta = None
            all_deltas = []  # list of (var_idx, delta)

            for var_idx, sign in c:
                # Compute delta if we flip var_idx
                delta = 0
                for ci in var_clauses[var_idx]:
                    cur_sat = clause_sat[ci]
                    if cur_sat:
                        # Check if any other literal in this clause is true
                        other_true = False
                        for v2, s2 in clause_list[ci]:
                            if v2 != var_idx and assignment[v2] == (s2 == 1):
                                other_true = True
                                break
                        if not other_true:
                            delta += 1  # becomes unsat
                    else:
                        # Currently unsat; after flip, is this literal now true?
                        if (not assignment[var_idx]) == (sign == 1):
                            delta -= 1  # becomes sat
                all_deltas.append((var_idx, delta))
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_var = var_idx

            # Simulated annealing acceptance
            # Find the minimum delta (best move)
            min_delta = min(d for _, d in all_deltas)

            # If the best move is non-worsening (delta <= 0), accept it
            # Otherwise, use SA to decide whether to accept a worsening move
            if min_delta <= 0:
                # Accept the best non-worsening move
                var_idx = best_var
            else:
                # All moves worsen; use SA probability
                # Accept with probability exp(-delta/T) where delta is the worsening amount
                # Use the best (least worsening) move as candidate
                delta_worst = min_delta  # this is positive (worsening)
                accept_prob = math.exp(-delta_worst / max(T, 1e-10))

                if rng.random() < accept_prob:
                    # Accept the least worsening move
                    var_idx = best_var
                else:
                    # Try a random move from the clause (more exploratory at high T)
                    # Or reject and try a different strategy
                    # At high temperatures, we're more likely to accept worsening moves
                    # At low temperatures, we stick to the best available
                    # For now, just pick the best (least bad) option
                    var_idx = best_var

            # Perform the flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Update clause satisfaction
            for ci in var_clauses[var_idx]:
                was_sat = clause_sat[ci]
                now_sat = False
                for v2, s2 in clause_list[ci]:
                    if assignment[v2] == (s2 == 1):
                        now_sat = True
                        break
                if was_sat and not now_sat:
                    unsat_count += 1
                elif not was_sat and now_sat:
                    unsat_count -= 1
                clause_sat[ci] = now_sat

        # After max_steps, if we found a better assignment, keep it
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
