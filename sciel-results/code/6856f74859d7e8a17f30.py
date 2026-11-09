import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 280.0

    rng = random.Random(42)

    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    for c, clause in enumerate(clauses):
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((var, sign))
            var_to_clauses[var].append(c)
        clause_vars.append(cv)

    def is_sat(c, assignment):
        for var, sign in clause_vars[c]:
            if (assignment[var] and sign == 1) or (not assignment[var] and sign == -1):
                return True
        return False

    best_assignment = None
    best_unsat = m + 1

    while time.time() < deadline:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_bool = [False] * m
        unsat_count = 0
        unsat_list = []
        for c in range(m):
            if not is_sat(c, assignment):
                unsat_bool[c] = True
                unsat_count += 1
                unsat_list.append(c)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        steps = 0
        max_steps = 100000
        while steps < max_steps and time.time() < deadline:
            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            c = rng.choice(unsat_list)
            cv = clause_vars[c]

            if rng.random() < 0.4:
                var = rng.choice(cv)[0]
            else:
                best_var = cv[0][0]
                best_score = -1
                for var, sign in cv:
                    score = 0
                    for c2 in var_to_clauses[var]:
                        if c2 != c and unsat_bool[c2]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_var = var
                var = best_var

            assignment[var] = not assignment[var]

            for c2 in var_to_clauses[var]:
                was_unsat = unsat_bool[c2]
                now_unsat = not is_sat(c2, assignment)
                if was_unsat and not now_unsat:
                    unsat_count -= 1
                    unsat_bool[c2] = False
                    try:
                        idx = unsat_list.index(c2)
                        unsat_list.pop(idx)
                    except ValueError:
                        pass
                elif not was_unsat and now_unsat:
                    unsat_count += 1
                    unsat_bool[c2] = True
                    unsat_list.append(c2)

            if unsat_count == 0:
                return assignment
            steps += 1

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [False] * n
