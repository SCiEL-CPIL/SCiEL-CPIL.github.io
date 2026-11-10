# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0

    m = len(clauses)
    if m == 0:
        return [True] * n

    # Pre-process
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vs = [abs(l) - 1 for l in c]
        ss = [1 if l > 0 else -1 for l in c]
        clause_vars.append(np.array(vs, dtype=np.int32))
        clause_signs.append(np.array(ss, dtype=np.int8))

    # Variable to clause incidence
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_clauses[v].append(i)

    var_clause_idx = [np.array(vc, dtype=np.int32) if vc else np.array([], dtype=np.int32) for vc in var_clauses]

    # For each variable, also store the sign in each clause it appears in
    var_clause_sign = []
    for v in range(n):
        if var_clauses[v]:
            sigs = []
            for ci in var_clauses[v]:
                # Find the position of v in clause ci
                vs = clause_vars[ci]
                pos = np.where(vs == v)[0][0]
                sigs.append(clause_signs[ci][pos])
            var_clause_sign.append(np.array(sigs, dtype=np.int8))
        else:
            var_clause_sign.append(np.array([], dtype=np.int8))

    np_rng = np.random.RandomState(0)
    best_assignment = None
    best_unsat = m + 1

    def compute_sat_counts(assignment):
        sat_counts = np.zeros(m, dtype=np.int16)
        for i in range(m):
            vs = clause_vars[i]
            ss = clause_signs[i]
            sat_counts[i] = np.sum(assignment[vs] == (ss == 1))
        return sat_counts

    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        assignment = np_rng.rand(n) < 0.5
        sat_counts = compute_sat_counts(assignment)
        total_unsat = int(np.sum(sat_counts == 0))

        if total_unsat == 0:
            return assignment.tolist()

        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment.copy()

        # WalkSAT with clause-weighted variable selection
        # Weight: probability of picking a clause proportional to (3 - sat_count) for unsat, or just uniform over unsat
        # For variable selection within clause, use impact heuristic

        max_steps = 80000 if restart < 5 else 150000
        step = 0
        while step < max_steps:
            if time.time() > deadline:
                break
            step += 1

            if total_unsat == 0:
                return assignment.tolist()

            # Pick random unsat clause
            unsat_mask = (sat_counts == 0)
            unsat_clauses = np.where(unsat_mask)[0]
            if len(unsat_clauses) == 0:
                return assignment.tolist()
            ci = int(np_rng.choice(unsat_clauses))

            vs = clause_vars[ci]
            ss = clause_signs[ci]

            # For each variable in this clause, compute a score
            # score[v] = (# unsat clauses containing v) - alpha * (# sat clauses containing v that would become unsat)
            # Simplified: just count unsat clauses containing v
            scores = []
            for j in range(len(vs)):
                v = vs[j]
                # Count how many unsat clauses v is in
                # This is expensive; use a lighter proxy
                # Just use random with slight bias
                scores.append(1.0)

            # Pick variable uniformly from clause (simple WalkSAT)
            vi = int(np_rng.choice(vs))
            s = ss[np.where(vs == vi)[0][0]]

            # Flip
            assignment[vi] = not assignment[vi]

            # Update sat_counts for all clauses containing vi
            idxs = var_clause_idx[vi]
            sigs_v = var_clause_sign[vi]
            if len(idxs) > 0:
                old_val = not assignment[vi]  # value before flip
                for k in range(len(idxs)):
                    idx = idxs[k]
                    s_k = sigs_v[k]
                    old_contrib = 1 if (old_val == (s_k == 1)) else 0
                    new_contrib = 1 if (assignment[vi] == (s_k == 1)) else 0
                    delta = new_contrib - old_contrib
                    old_sat = sat_counts[idx]
                    new_sat = old_sat + delta
                    sat_counts[idx] = new_sat
                    if old_sat == 0 and new_sat > 0:
                        total_unsat -= 1
                    elif old_sat > 0 and new_sat == 0:
                        total_unsat += 1

            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment.copy()

        if time.time() > deadline:
            break

    if best_assignment is not None:
        return best_assignment.tolist()
    return [True] * n

# EVOLVE-BLOCK-END
