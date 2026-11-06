"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Preprocess: 0-based variable indices and sign booleans per clause
    cvars = []
    csigns = []
    for cl in clauses:
        vs = [abs(l) - 1 for l in cl]
        ss = [l > 0 for l in cl]
        cvars.append(vs)
        csigns.append(ss)

    # Variable -> list of clause indices containing that variable
    var_clauses = [[] for _ in range(n)]
    for ci, vs in enumerate(cvars):
        for v in vs:
            var_clauses[v].append(ci)

    # Preprocessing: pure literal elimination
    pos_count = [0] * n
    neg_count = [0] * n
    for ci in range(m):
        for j in range(3):
            v = cvars[ci][j]
            if csigns[ci][j]:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    assigned = [None] * n
    for v in range(n):
        if pos_count[v] > 0 and neg_count[v] == 0:
            assigned[v] = True
        elif neg_count[v] > 0 and pos_count[v] == 0:
            assigned[v] = False

    # Unit propagation
    changed = True
    while changed:
        changed = False
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            unassigned_count = 0
            unassigned_idx = -1
            satisfied = False
            for j in range(3):
                v = vs[j]
                if assigned[v] is not None:
                    if assigned[v] == ss[j]:
                        satisfied = True
                        break
                else:
                    unassigned_count += 1
                    unassigned_idx = j
            if not satisfied:
                if unassigned_count == 1:
                    v = vs[unassigned_idx]
                    assigned[v] = ss[unassigned_idx]
                    changed = True

    # Build reduced problem
    old_to_new = {}
    new_var_count = 0
    for v in range(n):
        if assigned[v] is None:
            old_to_new[v] = new_var_count
            new_var_count += 1

    new_to_old = [None] * new_var_count
    for v in range(n):
        if assigned[v] is None:
            new_to_old[old_to_new[v]] = v

    reduced_cvars = []
    reduced_csigns = []
    reduced_var_clauses = [[] for _ in range(new_var_count)]
    for ci in range(m):
        vs = cvars[ci]
        ss = csigns[ci]
        satisfied = False
        for j in range(3):
            v = vs[j]
            if assigned[v] is not None and assigned[v] == ss[j]:
                satisfied = True
                break
        if satisfied:
            continue
        r_vs = []
        r_ss = []
        for j in range(3):
            v = vs[j]
            if assigned[v] is None:
                r_vs.append(old_to_new[v])
                r_ss.append(ss[j])
        if len(r_vs) == 0:
            continue
        if len(r_vs) == 1:
            # Find which original variable is unassigned
            for j in range(3):
                v = vs[j]
                if assigned[v] is None:
                    assigned[v] = ss[j]
                    break
            continue
        reduced_cvars.append(r_vs)
        reduced_csigns.append(r_ss)
        ci_new = len(reduced_cvars) - 1
        for v in r_vs:
            reduced_var_clauses[v].append(ci_new)

    m_reduced = len(reduced_cvars)
    if m_reduced == 0:
        final_assignment = [False] * n
        for v in range(n):
            if assigned[v] is not None:
                final_assignment[v] = assigned[v]
        return final_assignment

    n_red = new_var_count
    deadline = time.time() + 320.0

    def count_unsat(assignment):
        cnt = 0
        for ci in range(m_reduced):
            vs = reduced_cvars[ci]
            ss = reduced_csigns[ci]
            sat = False
            for j in range(len(vs)):
                if assignment[vs[j]] == ss[j]:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def walk_sat_optimized(assignment, max_flips, rng, noise_p):
        sat_count = [0] * m_reduced
        unsat_set = set()
        for ci in range(m_reduced):
            vs = reduced_cvars[ci]
            ss = reduced_csigns[ci]
            count = 0
            for j in range(len(vs)):
                if assignment[vs[j]] == ss[j]:
                    count += 1
            sat_count[ci] = count
            if count == 0:
                unsat_set.add(ci)

        m_unsat = len(unsat_set)
        if m_unsat == 0:
            return True, assignment, 0

        flips_used = 0
        unsat_list = list(unsat_set)

        for _ in range(max_flips):
            if m_unsat == 0:
                return True, assignment, 0

            if len(unsat_list) != m_unsat:
                unsat_list = list(unsat_set)

            ci = rng.choice(unsat_list)
            vs = reduced_cvars[ci]
            ss = reduced_csigns[ci]

            if rng.random() < noise_p:
                idx = rng.randint(0, len(vs) - 1)
            else:
                best_idx = 0
                best_score = float('inf')
                for j in range(len(vs)):
                    v = vs[j]
                    damage = 0
                    gain = 0
                    for cj in reduced_var_clauses[v]:
                        if cj == ci:
                            continue
                        cj_vs = reduced_cvars[cj]
                        cj_ss = reduced_csigns[cj]
                        for k in range(len(cj_vs)):
                            if cj_vs[k] == v:
                                if sat_count[cj] == 1 and assignment[v] == cj_ss[k]:
                                    damage += 1
                                elif sat_count[cj] == 0 and (not assignment[v]) == cj_ss[k]:
                                    gain += 1
                                break
                    score = damage - gain
                    if score < best_score:
                        best_score = score
                        best_idx = j
                idx = best_idx

            v = vs[idx]
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val
            flips_used += 1

            for cj in reduced_var_clauses[v]:
                cj_vs = reduced_cvars[cj]
                cj_ss = reduced_csigns[cj]
                for k in range(len(cj_vs)):
                    if cj_vs[k] == v:
                        old_sat = (old_val == cj_ss[k])
                        new_sat = (new_val == cj_ss[k])
                        if old_sat and not new_sat:
                            sat_count[cj] -= 1
                            if sat_count[cj] == 0:
                                unsat_set.add(cj)
                                m_unsat += 1
                        elif not old_sat and new_sat:
                            sat_count[cj] += 1
                            if sat_count[cj] == 1:
                                unsat_set.discard(cj)
                                m_unsat -= 1
                        break

        return False, assignment, flips_used

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = float('inf')

    restart_budgets = [2000, 5000, 10000, 20000, 40000, 80000, 150000, 300000]
    noise_levels = [0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]

    restart_idx = 0
    while time.time() < deadline:
        if restart_idx < len(restart_budgets):
            max_flips = restart_budgets[restart_idx]
            noise_p = noise_levels[restart_idx]
        else:
            max_flips = min(500000, restart_budgets[-1] * 2)
            noise_p = 0.5 + 0.3 * rng.random()

        assignment = [rng.random() < 0.5 for _ in range(n_red)]
        success, final_assignment, _ = walk_sat_optimized(assignment, max_flips, rng, noise_p)

        if success:
            final_full = [False] * n
            for i in range(n_red):
                final_full[new_to_old[i]] = final_assignment[i]
            for v in range(n):
                if assigned[v] is not None:
                    final_full[v] = assigned[v]
            return final_full

        unsat_cnt = count_unsat(final_assignment)
        if unsat_cnt < best_unsat_count:
            best_unsat_count = unsat_cnt
            best_assignment = final_assignment[:]

        restart_idx += 1

    if best_assignment is not None:
        final_full = [False] * n
        for i in range(n_red):
            final_full[new_to_old[i]] = best_assignment[i]
        for v in range(n):
            if assigned[v] is not None:
                final_full[v] = assigned[v]
        return final_full
    return [False] * n
# EVOLVE-BLOCK-END
