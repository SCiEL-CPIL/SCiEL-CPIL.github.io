"""Synthetic planted 3-SAT solver: WalkSAT with incremental clause bookkeeping,
flip-cost greedy selection, noise/budget schedules, deadline guard, best-tracking,
and a final high-precision polish pass."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 300.0  # safety margin before the 330s kill

    # Precompute clause structure: for each clause, list of (var_idx, polarity_bool)
    # polarity_bool True means the literal is positive (satisfied when assignment[var] is True)
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        cdata = []
        for lit in clause:
            v = abs(lit) - 1
            pol = (lit > 0)
            cdata.append((v, pol))
            var_to_clauses[v].append(ci)
        clause_data.append(cdata)

    m = len(clauses)
    rng = random.Random(0)

    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def satisfied_count_for_clause(assignment, ci):
        c = clause_data[ci]
        cnt = 0
        for v, pol in c:
            if assignment[v] == pol:
                cnt += 1
        return cnt

    def build_state(assignment):
        """Return (satisfied_count list, violated set) for a given assignment."""
        sat_cnt = [0] * m
        violated = set()
        for ci in range(m):
            c = clause_data[ci]
            cnt = 0
            for v, pol in c:
                if assignment[v] == pol:
                    cnt += 1
            sat_cnt[ci] = cnt
            if cnt == 0:
                violated.add(ci)
        return sat_cnt, violated

    def flip_cost(assignment, var_idx):
        """Change in number of violated clauses if we flip var_idx.
        Uses current sat_cnt to decide if a clause is 'pivotal' (satisfied only by this var).
        Returns (delta, min_cost_for_tiebreak) where delta is the change in violated count.
        """
        delta = 0
        for ci in var_to_clauses[var_idx]:
            c = clause_data[ci]
            # Find position of var_idx in this clause
            pos = -1
            for k, (v, pol) in enumerate(c):
                if v == var_idx:
                    pos = k
                    break
            if pos < 0:
                continue
            pol = c[pos][1]
            current_val = assignment[var_idx]
            new_val = not current_val
            currently_sat_by_this = (current_val == pol)
            after_sat_by_this = (new_val == pol)
            # Count how many OTHER literals in this clause are satisfied
            sat_by_other = 0
            for k, (v2, pol2) in enumerate(c):
                if k == pos:
                    continue
                if assignment[v2] == pol2:
                    sat_by_other += 1
            currently_sat = (currently_sat_by_this or sat_by_other > 0)
            after_sat = (after_sat_by_this or sat_by_other > 0)
            if not currently_sat and after_sat:
                delta -= 1
            elif currently_sat and not after_sat:
                delta += 1
        return delta

    best_assignment = random_assignment()
    best_sat_cnt, best_violated = build_state(best_assignment)
    best_count = len(best_violated)

    if best_count == 0:
        return best_assignment

    # Main search loop with restarts
    restarts = 0
    max_restarts = 400
    base_budget = 8000
    noise = 0.25

    while restarts < max_restarts and time.time() < deadline:
        assignment = random_assignment()
        sat_cnt, violated = build_state(assignment)
        current_count = len(violated)

        if current_count < best_count:
            best_count = current_count
            best_assignment = assignment[:]
            best_sat_cnt, best_violated = sat_cnt, violated

        if current_count == 0:
            return assignment

        # Adaptive noise: decrease over restarts, but bump periodically to escape local optima
        eff_noise = max(0.10, noise - restarts * 0.0015)
        budget = base_budget + restarts * 1500

        no_improve = 0
        last_count = current_count

        for step in range(budget):
            if time.time() > deadline:
                break
            if current_count == 0:
                return assignment
            if not violated:
                return assignment

            # Pick a random violated clause
            ci = rng.choice(list(violated))
            c = clause_data[ci]

            # Decide which variable to flip
            if rng.random() < eff_noise:
                v = rng.choice(c)[0]
            else:
                # Greedy: pick the variable in this clause with the smallest flip cost
                best_v = None
                best_cost = float('inf')
                for v, pol in c:
                    cost = flip_cost(assignment, v)
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v
                v = best_v

            old_val = assignment[v]
            assignment[v] = not assignment[v]

            # Incrementally update sat_cnt and violated for clauses containing v
            for cj in var_to_clauses[v]:
                c2 = clause_data[cj]
                # Find position of v in c2
                pos = -1
                for k, (v2, pol2) in enumerate(c2):
                    if v2 == v:
                        pos = k
                        break
                if pos < 0:
                    continue
                pol2 = c2[pos][1]
                new_val = assignment[v]
                was_sat = (old_val == pol2)
                now_sat = (new_val == pol2)
                if was_sat and not now_sat:
                    sat_cnt[cj] -= 1
                    if sat_cnt[cj] == 0:
                        violated.add(cj)
                elif not was_sat and now_sat:
                    sat_cnt[cj] += 1
                    if cj in violated:
                        violated.discard(cj)

            current_count = len(violated)
            if current_count < best_count:
                best_count = current_count
                best_assignment = assignment[:]
                best_sat_cnt, best_violated = sat_cnt, violated
                no_improve = 0
            else:
                no_improve += 1
                # If stuck for a while, increase noise temporarily
                if no_improve > 2000:
                    eff_noise = min(0.6, eff_noise + 0.05)
                    no_improve = 0

        restarts += 1
        # Periodically bump base noise to escape local optima
        if restarts % 8 == 0:
            noise = min(0.55, noise + 0.02)

    # Final high-precision polish pass on the best assignment found
    if time.time() < deadline - 5.0 and best_count > 0:
        assignment = best_assignment[:]
        sat_cnt, violated = build_state(assignment)
        current_count = len(violated)
        polish_noise = 0.05
        polish_budget = 20000
        for step in range(polish_budget):
            if time.time() > deadline:
                break
            if current_count == 0:
                return assignment
            if not violated:
                return assignment

            ci = rng.choice(list(violated))
            c = clause_data[ci]

            if rng.random() < polish_noise:
                v = rng.choice(c)[0]
            else:
                # Strict greedy with tie-breaking: pick smallest flip cost, break ties by
                # a soft-min surrogate (average of costs) to smooth the objective
                costs = []
                for v, pol in c:
                    cost = flip_cost(assignment, v)
                    costs.append((cost, v))
                if costs:
                    # Soft-min: weight each cost by exp(-cost/T) and pick the variable
                    # with the highest weight; on ties, prefer the one that satisfies
                    # the most other clauses (smallest cost first, then random)
                    min_cost = min(c0 for c0, _ in costs)
                    # Among variables with the minimum cost, pick one that also
                    # helps other clauses: recompute a secondary score
                    candidates = [v0 for c0, v0 in costs if c0 == min_cost]
                    if len(candidates) == 1:
                        v = candidates[0]
                    else:
                        # Secondary: pick the candidate with the smallest total
                        # contribution to other violated clauses (greedy tie-break)
                        best_v = candidates[0]
                        best_sec = float('inf')
                        for v0 in candidates:
                            sec = 0
                            for cj in var_to_clauses[v0]:
                                if cj in violated:
                                    # Count how many other literals in cj are satisfied
                                    c2 = clause_data[cj]
                                    other_sat = 0
                                    for v2, pol2 in c2:
                                        if v2 != v0 and assignment[v2] == pol2:
                                            other_sat += 1
                                    # Flipping v0 satisfies cj if other_sat >= 0 (it does, since cj is violated)
                                    # Secondary score: prefer flipping a var that appears in more violated clauses
                                    sec -= len([cj2 for cj2 in var_to_clauses[v0] if cj2 in violated])
                            if sec < best_sec:
                                best_sec = sec
                                best_v = v0
                        v = best_v
                else:
                    v = rng.choice(c)[0]

            old_val = assignment[v]
            assignment[v] = not assignment[v]

            for cj in var_to_clauses[v]:
                c2 = clause_data[cj]
                pos = -1
                for k, (v2, pol2) in enumerate(c2):
                    if v2 == v:
                        pos = k
                        break
                if pos < 0:
                    continue
                pol2 = c2[pos][1]
                new_val = assignment[v]
                was_sat = (old_val == pol2)
                now_sat = (new_val == pol2)
                if was_sat and not now_sat:
                    sat_cnt[cj] -= 1
                    if sat_cnt[cj] == 0:
                        violated.add(cj)
                elif not was_sat and now_sat:
                    sat_cnt[cj] += 1
                    if cj in violated:
                        violated.discard(cj)

            current_count = len(violated)
            if current_count < best_count:
                best_count = current_count
                best_assignment = assignment[:]
                best_sat_cnt, best_violated = sat_cnt, violated

    return best_assignment
# EVOLVE-BLOCK-END
