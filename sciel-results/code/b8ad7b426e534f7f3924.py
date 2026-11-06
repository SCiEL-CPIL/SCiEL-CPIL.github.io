"""Improved planted 3-SAT solver with dynamic WalkSAT and smarter restarts."""
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 330s kill

    cnf = [tuple(c) for c in clauses]
    m = len(cnf)

    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
            positions.append((vi, 1 if lit > 0 else -1))
        clause_lit_pos.append(positions)

    def count_unsat(assignment):
        count = 0
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def is_satisfied(assignment):
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Preprocessing: unit propagation and pure literal elimination
    def preprocess():
        assignment = [None] * n
        cnf_work = [list(c) for c in cnf]
        changed = True
        while changed:
            changed = False
            # Remove satisfied clauses
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf

            # Unit propagation
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment
                elif len(c) == 0:
                    return None, None, assignment

            # Pure literal elimination
            lit_counts = [[0, 0] for _ in range(n)]
            for c in cnf_work:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen:
                        continue
                    seen.add(vi)
                    if lit > 0:
                        lit_counts[vi][0] += 1
                    else:
                        lit_counts[vi][1] += 1

            for vi in range(n):
                if assignment[vi] is not None:
                    continue
                if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                    assignment[vi] = False
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True
                    changed = True

        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            if is_satisfied(final_assign):
                return final_assign, None, final_assign
            return None, None, assignment

        return None, cnf_work, assignment

    solved, reduced_cnf, partial_assign = preprocess()
    if solved is not None:
        return solved

    # Build effective CNF after preprocessing
    if reduced_cnf is not None:
        eff_cnf = [tuple(c) for c in reduced_cnf if len(c) > 0]
        eff_m = len(eff_cnf)
        eff_var_to_clauses = [[] for _ in range(n)]
        eff_clause_lit_pos = []
        for ci, c in enumerate(eff_cnf):
            positions = []
            for lit in c:
                vi = abs(lit) - 1
                eff_var_to_clauses[vi].append(ci)
                positions.append((vi, 1 if lit > 0 else -1))
            eff_clause_lit_pos.append(positions)
    else:
        eff_cnf = cnf
        eff_m = m
        eff_var_to_clauses = var_to_clauses
        eff_clause_lit_pos = clause_lit_pos

    def make_full_assignment(partial, seed):
        rng = random.Random(seed)
        assign = []
        for i in range(n):
            if partial[i] is not None:
                assign.append(bool(partial[i]))
            else:
                assign.append(rng.random() < 0.5)
        return assign

    start_seed = 42
    start_assign = make_full_assignment(partial_assign, start_seed) if reduced_cnf else [random.Random(start_seed).random() < 0.5 for _ in range(n)]

    if is_satisfied(start_assign):
        return start_assign

    best_assign = start_assign[:]
    best_unsat = count_unsat(start_assign)

    def walksat_dynamic(assign, max_steps, base_p, seed):
        """WalkSAT with dynamic walk parameter that increases on stagnation."""
        if time.time() > deadline:
            return False, assign, count_unsat(assign)

        rng = random.Random(seed)
        cur = assign[:]

        # Incremental clause satisfaction tracking
        clause_true_count = [0] * eff_m
        for ci in range(eff_m):
            cnt = 0
            for vi, sign in eff_clause_lit_pos[ci]:
                if cur[vi] == (sign > 0):
                    cnt += 1
            clause_true_count[ci] = cnt

        cur_unsat = sum(1 for c in clause_true_count if c == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]

        # Stagnation tracking
        stagnation = 0
        stall_threshold = 500
        p = base_p
        max_p = 0.95
        min_p = 0.3

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0

            # Find unsatisfied clauses
            unsat_clauses = [ci for ci in range(eff_m) if clause_true_count[ci] == 0]
            if not unsat_clauses:
                return True, cur, 0

            # Select a random unsat clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = eff_cnf[ci]

            # Compute flip weights based on impact
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                var_degree = len(eff_var_to_clauses[vi])
                w = 1.0 / (1 + var_degree)

                if cur[vi] == (lit > 0):
                    w *= 0.1
                else:
                    # Estimate help/hurt
                    help_count = 0
                    hurt_count = 0
                    for aci in eff_var_to_clauses[vi]:
                        if aci == ci:
                            continue
                        for vi2, sign2 in eff_clause_lit_pos[aci]:
                            if vi2 == vi:
                                if sign2 > 0:
                                    if not cur[vi]:
                                        help_count += 1
                                    else:
                                        hurt_count += 1
                                else:
                                    if cur[vi]:
                                        help_count += 1
                                    else:
                                        hurt_count += 1
                                break
                    net_help = help_count - hurt_count
                    w *= (1.0 + 0.5 * net_help)
                    if w < 0.1:
                        w = 0.1
                weights.append(w)

            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break

            literal = clause[chosen_idx]
            vi = abs(literal) - 1

            # Dynamic walk parameter: flip to satisfy with prob p, else random
            if rng.random() < p:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]

            # Incremental update
            for aci in eff_var_to_clauses[vi]:
                old_count = clause_true_count[aci]
                new_count = 0
                for vi2, sign2 in eff_clause_lit_pos[aci]:
                    if cur[vi2] == (sign2 > 0):
                        new_count += 1
                clause_true_count[aci] = new_count
                if old_count == 0 and new_count > 0:
                    cur_unsat -= 1
                elif old_count > 0 and new_count == 0:
                    cur_unsat += 1

            # Update stagnation and dynamic p
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
                stagnation = 0
                # Reset p toward base
                p = max(base_p, p - 0.05)
            else:
                stagnation += 1
                if stagnation > stall_threshold:
                    # Increase randomization to escape local minima
                    p = min(max_p, p + 0.05)
                    if p >= max_p:
                        # Full random walk for a bit
                        pass

        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    # Multi-restart with adaptive parameters
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765,
             31415, 27182, 16180, 14142, 17320, 81723, 29176, 12321, 99999, 55555]
    base_p_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82,
                     0.72, 0.88, 0.68, 0.76, 0.62, 0.83, 0.71, 0.79, 0.66, 0.87]

    # Adaptive step budget
    base_steps = 2000
    max_steps_cap = 50000

    while time.time() < deadline - 15:
        seed = seeds[attempt % len(seeds)]
        base_p = base_p_values[attempt % len(base_p_values)]
        # Adaptive steps: increase with attempts but cap
        steps = min(base_steps + attempt * 500, max_steps_cap)

        # Choose starting assignment
        if attempt % 5 == 0:
            # Fresh random start
            start = [random.Random(seed * 7 + attempt).random() < 0.5 for _ in range(n)]
        elif attempt % 10 == 0:
            # Mutate best assignment
            start = best_assign[:]
            rng_mut = random.Random(seed * 13 + attempt)
            for i in range(n):
                if rng_mut.random() < 0.3:
                    start[i] = not start[i]
        else:
            # Start from best
            start = best_assign[:]

        success, result, min_unsat = walksat_dynamic(start, steps, base_p, seed)

        if success:
            return result

        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]

        attempt += 1

    # Final polish: targeted flips for few remaining unsat clauses
    if is_satisfied(best_assign):
        return best_assign

    if best_unsat <= 8:
        for _ in range(10000):
            if time.time() > deadline - 3:
                break
            # Find an unsat clause
            unsat_ci = None
            for ci in range(eff_m):
                sat = False
                for lit in eff_cnf[ci]:
                    vi = abs(lit) - 1
                    if best_assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat_ci = ci
                    break
            if unsat_ci is None:
                return best_assign

            clause = eff_cnf[unsat_ci]
            for lit in clause:
                vi = abs(lit) - 1
                old_val = best_assign[vi]
                best_assign[vi] = not old_val
                u = count_unsat(best_assign)
                if u == 0:
                    return best_assign
                if u < best_unsat:
                    best_unsat = u
                else:
                    best_assign[vi] = old_val

    # Last resort: random hill climbing
    rng = random.Random(0)
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        vi = rng.randrange(n)
        best_assign[vi] = not best_assign[vi]
        if is_satisfied(best_assign):
            return best_assign
        u = count_unsat(best_assign)
        if u > best_unsat:
            best_assign[vi] = not best_assign[vi]

    return best_assign
