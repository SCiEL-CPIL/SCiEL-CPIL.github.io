# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0

    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n

    # Preprocessing: Pure Literal Elimination
    var_pos = [False] * n
    var_neg = [False] * n
    for c in clauses:
        for l in c:
            v = abs(l) - 1
            if l > 0:
                var_pos[v] = True
            else:
                var_neg[v] = True

    fixed = {}
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in fixed:
                continue
            if var_pos[v] and not var_neg[v]:
                fixed[v] = True
                changed = True
            elif var_neg[v] and not var_pos[v]:
                fixed[v] = False
                changed = True

    # Simplify clauses
    simplified_clauses = []
    for c in clauses:
        new_clause = []
        sat = False
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            if v in fixed:
                if fixed[v] == (sign == 1):
                    sat = True
                    break
            else:
                new_clause.append((v, sign))
        if not sat:
            if len(new_clause) == 0:
                return [fixed.get(v, True) for v in range(n)]
            simplified_clauses.append(new_clause)

    if not simplified_clauses:
        return [fixed.get(v, True) for v in range(n)]

    free_vars = set()
    for c in simplified_clauses:
        for v, s in c:
            free_vars.add(v)
    free_var_list = list(free_vars)
    num_free = len(free_var_list)

    if num_free == 0:
        return [fixed.get(v, True) for v in range(n)]

    # Build incidence: var -> list of (clause_idx, sign)
    var_incidence = [[] for _ in range(n)]
    for c_idx, c in enumerate(simplified_clauses):
        for v, s in c:
            var_incidence[v].append((c_idx, s))

    rng = random.Random(42)

    def evaluate_assignment(full_assign):
        clause_sat_count = [0] * len(simplified_clauses)
        for c_idx, c in enumerate(simplified_clauses):
            sat_count = 0
            for v, s in c:
                if full_assign[v] == (s == 1):
                    sat_count += 1
            clause_sat_count[c_idx] = sat_count
        num_unsat = sum(1 for count in clause_sat_count if count == 0)
        return clause_sat_count, num_unsat

    best_assign = None
    best_unsat = float('inf')

    max_steps_per_restart = 5000
    max_restarts = 1000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart % 3 == 0:
            assign = {v: rng.random() < 0.5 for v in free_var_list}
        elif restart % 3 == 1:
            assign = {v: True for v in free_var_list}
        else:
            assign = {v: False for v in free_var_list}

        full_assign = [False] * n
        for v in range(n):
            if v in fixed:
                full_assign[v] = fixed[v]
            else:
                full_assign[v] = assign.get(v, False)

        clause_sat_count, num_unsat = evaluate_assignment(full_assign)

        if num_unsat == 0:
            best_assign = full_assign
            break

        if num_unsat < best_unsat:
            best_unsat = num_unsat
            best_assign = full_assign[:]

        unsat_clauses = [i for i, cnt in enumerate(clause_sat_count) if cnt == 0]

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                best_assign = full_assign
                return best_assign

            c_idx = rng.choice(unsat_clauses)
            clause = simplified_clauses[c_idx]

            best_var = None
            best_score = float('inf')

            for v, s in clause:
                old_val = full_assign[v]
                new_val = not old_val

                delta_unsat = 0
                for c_idx_2, s_2 in var_incidence[v]:
                    v_was_sat = (old_val == (s_2 == 1))
                    v_will_sat = (new_val == (s_2 == 1))

                    if v_was_sat and not v_will_sat:
                        if clause_sat_count[c_idx_2] == 1:
                            delta_unsat += 1
                    elif not v_was_sat and v_will_sat:
                        if clause_sat_count[c_idx_2] == 0:
                            delta_unsat -= 1

                score = delta_unsat + rng.random() * 0.5

                if score < best_score:
                    best_score = score
                    best_var = v

            if best_var is not None:
                old_val = full_assign[best_var]
                new_val = not old_val
                full_assign[best_var] = new_val

                for c_idx_2, s_2 in var_incidence[best_var]:
                    v_was_sat = (old_val == (s_2 == 1))
                    v_will_sat = (new_val == (s_2 == 1))

                    if v_was_sat and not v_will_sat:
                        clause_sat_count[c_idx_2] -= 1
                        if clause_sat_count[c_idx_2] == 0:
                            if c_idx_2 not in unsat_clauses:
                                unsat_clauses.append(c_idx_2)
                    elif not v_was_sat and v_will_sat:
                        clause_sat_count[c_idx_2] += 1
                        if clause_sat_count[c_idx_2] == 1:
                            if c_idx_2 in unsat_clauses:
                                unsat_clauses.remove(c_idx_2)

                num_unsat = len(unsat_clauses)
                if num_unsat < best_unsat:
                    best_unsat = num_unsat
                    best_assign = full_assign[:]

                if num_unsat == 0:
                    return full_assign

    if best_assign is not None:
        return best_assign
    else:
        return [fixed.get(v, rng.random() < 0.5) for v in range(n)]
# EVOLVE-BLOCK-END
