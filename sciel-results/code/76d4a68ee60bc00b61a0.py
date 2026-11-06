"""Rewritten 3-SAT solver: weighted WalkSAT with adaptive clause weighting and soft-min scoring."""
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 310.0

    # Precompute literal arrays
    lit_idx = np.empty((m, 3), dtype=np.int32)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lit_idx[i, j] = abs(lit) - 1
            lit_sign[i, j] = 1 if lit > 0 else -1

    # Build variable-to-clause mapping
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_idx[i, j]
            var_clauses[v].append((i, j, lit_sign[i, j]))

    # Precompute arrays for each variable
    var_clause_idx = [np.array([c[0] for c in var_clauses[v]], dtype=np.int32) if var_clauses[v] else np.array([], dtype=np.int32) for v in range(n)]
    var_pos = [np.array([c[1] for c in var_clauses[v]], dtype=np.int8) if var_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]
    var_sign = [np.array([c[2] for c in var_clauses[v]], dtype=np.int8) if var_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]

    rng = np.random.default_rng(42)

    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 80

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment with slight bias toward the planted solution (unknown, so random)
        assignment = rng.random(n) < 0.5

        # Compute initial sat_count
        is_true = (assignment[lit_idx] == (lit_sign > 0))
        sat_count = is_true.sum(axis=1)
        unsat_count = int((sat_count == 0).sum())

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Adaptive noise: starts higher, decreases
        base_noise = 0.45 + 0.25 * (restart / max_restarts)
        base_noise = min(base_noise, 0.70)

        # Clause weights: start uniform, boost for clauses that remain unsat
        clause_weight = np.ones(m, dtype=np.float64)
        # Track how long each clause has been unsat
        unsat_duration = np.zeros(m, dtype=np.float64)

        # Flip budget
        time_per_restart = (deadline - time.time()) / max(1, (max_restarts - restart))
        flip_budget = int(min(time_per_restart * 400000, 4000000))
        flip_budget = max(flip_budget, 8000)

        # Stagnation counter
        stagnation = 0
        local_best_unsat = unsat_count

        for step in range(flip_budget):
            if time.time() > deadline:
                break

            if unsat_count == 0:
                return assignment.tolist()

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

            # Detect stagnation
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                stagnation = 0
            else:
                stagnation += 1

            # Find unsatisfied clauses
            unsat_mask = (sat_count == 0)
            if not np.any(unsat_mask):
                break

            # Update unsat duration and clause weights
            unsat_duration[unsat_mask] += 1
            # Boost weights for clauses that have been unsat longer
            # Use a smooth boost: weight increases with duration
            boost_factor = 1.0 + 0.1 * np.log1p(unsat_duration)
            clause_weight[unsat_mask] *= boost_factor
            # Normalize weights to prevent overflow
            if clause_weight.max() > 1e6:
                clause_weight /= clause_weight.max()

            # Select clause: weighted random among unsat clauses
            unsat_clauses = np.where(unsat_mask)[0]
            unsat_weights = clause_weight[unsat_clauses]
            total_w = unsat_weights.sum()
            if total_w <= 0:
                clause_idx = unsat_clauses[rng.integers(len(unsat_clauses))]
            else:
                probs = unsat_weights / total_w
                clause_idx = unsat_clauses[rng.choice(len(unsat_clauses), p=probs)]

            # Get the 3 literals in this clause
            c_idx = lit_idx[clause_idx]
            c_sign = lit_sign[clause_idx]

            # Compute scores for each of the 3 variables
            # Score = number of currently-satisfied clauses (with sat_count==1) that would become unsat
            # Use soft-min via log-sum-exp for a smooth surrogate
            scores = np.zeros(3, dtype=np.float64)
            for j in range(3):
                v = c_idx[j]
                vc_idx = var_clause_idx[v]
                if len(vc_idx) == 0:
                    scores[j] = 0.0
                    continue
                v_s = var_sign[v]
                # For each occurrence of v, the literal is true iff assignment[v] == (v_s > 0)
                lit_true_arr = (assignment[v] == (v_s > 0))
                if np.any(lit_true_arr):
                    # Clauses where v's literal is true AND sat_count == 1
                    relevant = vc_idx[lit_true_arr]
                    sat_relevant = sat_count[relevant]
                    # Count how many would become unsat (sat_count == 1)
                    scores[j] = float((sat_relevant == 1).sum())
                else:
                    scores[j] = 0.0

            # Soft-min: use log-sum-exp to get a smooth minimum
            # soft_min = -log(sum(exp(-scores/tau))) * tau, with tau controlling smoothness
            tau = 0.3
            neg_scores = -scores / tau
            max_neg = neg_scores.max()
            log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
            soft_min_score = -log_sum_exp * tau

            # With probability noise, pick randomly; otherwise pick argmin of scores
            # Use soft-min as a tiebreaker guide: if all scores are close, prefer random
            noise = base_noise + 0.1 * min(stagnation / 50.0, 1.0)
            noise = min(noise, 0.85)

            if rng.random() < noise:
                flip_pos = rng.integers(3)
            else:
                min_score = scores.min()
                candidates = np.where(scores == min_score)[0]
                flip_pos = candidates[rng.integers(len(candidates))]

            v = c_idx[flip_pos]

            # Flip the variable
            assignment[v] = not assignment[v]

            # Update sat_count for all clauses containing v
            vc_idx = var_clause_idx[v]
            if len(vc_idx) > 0:
                vs = var_sign[v]
                old_v = not assignment[v]
                old_lit = (old_v == (vs > 0))
                delta = np.where(old_lit, -1, 1)
                sat_count[vc_idx] += delta
                sat_count = np.clip(sat_count, 0, 3)

            unsat_count = int((sat_count == 0).sum())

        # End of restart
        if unsat_count == 0:
            return assignment.tolist()

        # Decay clause weights between restarts
        clause_weight = np.ones(m, dtype=np.float64)
        unsat_duration = np.zeros(m, dtype=np.float64)

    # Final polish: try to fix remaining unsat clauses with targeted flips
    if best_assignment is not None and best_unsat_count < m:
        if time.time() < deadline - 5.0:
            assignment = best_assignment.copy()
            is_true = (assignment[lit_idx] == (lit_sign > 0))
            sat_count = is_true.sum(axis=1)
            unsat_count = int((sat_count == 0).sum())

            # Try targeted flips on remaining unsat clauses
            polish_budget = int(min((deadline - time.time() - 2.0) * 300000, 1000000))
            for _ in range(polish_budget):
                if time.time() > deadline - 2.0:
                    break
                if unsat_count == 0:
                    return assignment.tolist()

                unsat_clauses = np.where(sat_count == 0)[0]
                if len(unsat_clauses) == 0:
                    break
                clause_idx = unsat_clauses[rng.integers(len(unsat_clauses))]
                c_idx = lit_idx[clause_idx]

                # Try each of the 3 flips, pick the one that reduces unsat most
                best_delta = 0
                best_v = -1
                for j in range(3):
                    v = c_idx[j]
                    # Simulate flip
                    vc_idx = var_clause_idx[v]
                    if len(vc_idx) == 0:
                        continue
                    vs = var_sign[v]
                    old_lit = (assignment[v] == (vs > 0))
                    delta = np.where(old_lit, -1, 1)
                    new_sat = sat_count.copy()
                    new_sat[vc_idx] += delta
                    new_sat = np.clip(new_sat, 0, 3)
                    new_unsat = int((new_sat == 0).sum())
                    delta_unsat = unsat_count - new_unsat
                    if delta_unsat > best_delta:
                        best_delta = delta_unsat
                        best_v = v

                if best_v >= 0:
                    v = best_v
                    assignment[v] = not assignment[v]
                    vc_idx = var_clause_idx[v]
                    if len(vc_idx) > 0:
                        vs = var_sign[v]
                        old_v = not assignment[v]
                        old_lit = (old_v == (vs > 0))
                        delta = np.where(old_lit, -1, 1)
                        sat_count[vc_idx] += delta
                        sat_count = np.clip(sat_count, 0, 3)
                    unsat_count = int((sat_count == 0).sum())
                    if unsat_count < best_unsat_count:
                        best_unsat_count = unsat_count
                        best_assignment = assignment.copy()

            if unsat_count == 0:
                return assignment.tolist()

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
