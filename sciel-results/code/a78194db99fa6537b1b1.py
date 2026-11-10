"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """SA + WalkSAT hybrid SAT solver with O(1) incremental tracking and position-based unsat list."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign)
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build index: var_idx -> list of (clause_idx, sign_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))

    rng = random.Random(42)

    # SA Parameters
    T_initial = 8.0
    alpha = 0.995
    min_T = 0.005

    # WalkSAT parameters
    ws_probability = 0.5  # probability of using WalkSAT heuristic vs pure SA
    ws_min_break = True   # if True, prefer min-break; else random among unsat clause vars

    # Multi-restart strategy
    for restart in range(500):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Initialize sat_count: number of true literals in each clause
        sat_count = [0] * num_clauses
        unsat_clauses_list = []
        unsat_count = 0
        # pos_in_unsat[ci] = index in unsat_clauses_list if ci is unsat, else -1
        pos_in_unsat = [-1] * num_clauses

        for ci, clause in enumerate(clause_data):
            count = 0
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    count += 1
            sat_count[ci] = count
            if count == 0:
                unsat_count += 1
                pos_in_unsat[ci] = len(unsat_clauses_list)
                unsat_clauses_list.append(ci)

        if unsat_count == 0:
            return assignment

        T = T_initial
        steps = 0
        max_steps = 80000  # Steps per restart (increased since O(1) removal is faster)

        while steps < max_steps:
            if time.time() > deadline:
                return assignment

            if unsat_count == 0:
                return assignment

            steps += 1

            # 1. Select a move
            if not unsat_clauses_list:
                return assignment

            ci = rng.randrange(unsat_count)
            # ci is an index into unsat_clauses_list
            clause_idx = unsat_clauses_list[ci]

            # Choose variable to flip
            if rng.random() < ws_probability:
                # WalkSAT: pick variable that minimizes break count
                clause = clause_data[clause_idx]
                best_var = -1
                best_break = 100
                best_tie = 0
                for k in range(3):
                    var_idx, sign = clause[k]
                    # Compute break count: how many currently-satisfied clauses would become unsat
                    break_count = 0
                    for (j, s_j) in var_to_clauses[var_idx]:
                        if sat_count[j] == 1:
                            # This clause has exactly 1 true literal
                            # Check if that literal is the one we'd flip
                            lit_is_true = (assignment[var_idx] and s_j == 1) or (not assignment[var_idx] and s_j == -1)
                            if lit_is_true:
                                break_count += 1
                    if break_count < best_break or (break_count == best_break and rng.random() < 0.5):
                        best_break = break_count
                        best_var = var_idx
                var_idx = best_var
            else:
                # Pure SA: random variable from unsat clause
                var_idx, sign = clause_data[clause_idx][rng.randrange(3)]

            # 2. Evaluate the delta of unsat_count if we flip var_idx
            delta_unsat = 0

            for (j, s_j) in var_to_clauses[var_idx]:
                lit_is_true = (assignment[var_idx] and s_j == 1) or (not assignment[var_idx] and s_j == -1)

                if lit_is_true:
                    if sat_count[j] == 1:
                        delta_unsat += 1
                else:
                    if sat_count[j] == 0:
                        delta_unsat -= 1

            # 3. Acceptance Rule
            accept = False
            if delta_unsat < 0:
                accept = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_unsat / T)
                    if rng.random() < prob:
                        accept = True

            if accept:
                # Apply the flip
                assignment[var_idx] = not assignment[var_idx]

                # Update sat_count and unsat_clauses_list with O(1) position tracking
                for (j, s_j) in var_to_clauses[var_idx]:
                    lit_is_true_now = (assignment[var_idx] and s_j == 1) or (not assignment[var_idx] and s_j == -1)

                    if lit_is_true_now:
                        # Literal became True. sat_count increases.
                        if sat_count[j] == 0:
                            # Clause was unsat, now sat.
                            unsat_count -= 1
                            # Remove j from unsat_clauses_list in O(1)
                            pos_j = pos_in_unsat[j]
                            last_idx = unsat_count  # index of last element (after decrement)
                            if pos_j != last_idx:
                                last_clause = unsat_clauses_list[last_idx]
                                unsat_clauses_list[pos_j] = last_clause
                                pos_in_unsat[last_clause] = pos_j
                            unsat_clauses_list.pop()
                            pos_in_unsat[j] = -1
                        sat_count[j] += 1
                    else:
                        # Literal became False. sat_count decreases.
                        if sat_count[j] == 1:
                            # Clause was sat (count 1), now unsat (count 0).
                            unsat_count += 1
                            pos_in_unsat[j] = len(unsat_clauses_list)
                            unsat_clauses_list.append(j)
                        sat_count[j] -= 1

                # Cool down
                T *= alpha
            else:
                # Rejected, no state change
                T *= alpha

    # If no solution found within deadline, return last assignment
    return assignment

# EVOLVE-BLOCK-END
