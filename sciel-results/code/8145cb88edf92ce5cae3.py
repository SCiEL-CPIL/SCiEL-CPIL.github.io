"""Rewritten planted 3-SAT solver with DPLL + unit propagation + pure literals + restarts."""
import time
import random
import sys


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Preprocess: convert to 0-based, handle tautologies
    clean_clauses = []
    for c in clauses:
        # Remove tautological clauses (contains both x and ~x)
        lit_set = set(c)
        taut = False
        for l in c:
            if -l in lit_set:
                taut = True
                break
        if not taut:
            clean_clauses.append([abs(l) - 1 for l in c])
    # Store signs separately for efficiency
    clause_signs = []
    for c in clean_clauses:
        clause_signs.append([1 if (abs(l) - 1 == abs(l) - 1 and l > 0) else -1 for l in c])
    # Actually let's redo this properly
    clause_vars = []
    clause_signs = []
    for c in clauses:
        lit_set = set(c)
        taut = False
        for l in c:
            if -l in lit_set:
                taut = True
                break
        if not taut:
            vars_in_clause = [abs(l) - 1 for l in c]
            signs_in_clause = [1 if l > 0 else -1 for l in c]
            clause_vars.append(vars_in_clause)
            clause_signs.append(signs_in_clause)

    m = len(clause_vars)
    if m == 0:
        return [True] * n

    # Build clause index: for each variable, list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, (vars_c, signs_c) in enumerate(zip(clause_vars, clause_signs)):
        for vi, si in zip(vars_c, signs_c):
            var_to_clauses[vi].append((ci, si))

    # DPLL with unit propagation and pure literal elimination
    def dpll(assignment, trail, depth):
        # assignment: list of None/True/False
        # trail: list of (var, value) for backtracking
        if time.time() > deadline:
            return None

        # Unit propagation
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            for ci in range(m):
                if len(clause_vars[ci]) == 0:
                    # Empty clause: contradiction
                    return None
                if len(clause_vars[ci]) == 1:
                    v = clause_vars[ci][0]
                    s = clause_signs[ci][0]
                    val = (s == 1)
                    if assignment[v] is None:
                        assignment[v] = val
                        trail.append(v)
                        changed = True
                    elif assignment[v] != val:
                        return None
            # Remove satisfied clauses and shrink clauses
            if changed:
                new_clause_vars = []
                new_clause_signs = []
                for ci in range(m):
                    satisfied = False
                    new_vars = []
                    new_signs = []
                    for vi, si in zip(clause_vars[ci], clause_signs[ci]):
                        av = assignment[vi]
                        if av is not None:
                            if (av == (si == 1)):
                                satisfied = True
                                break
                        else:
                            new_vars.append(vi)
                            new_signs.append(si)
                    if not satisfied:
                        new_clause_vars.append(new_vars)
                        new_clause_signs.append(new_signs)
                # Reassign
                nonlocal clause_vars, clause_signs
                clause_vars = new_clause_vars
                clause_signs = new_clause_signs
                # Rebuild var_to_clauses
                for i in range(n):
                    var_to_clauses[i] = []
                for ci, (vars_c, signs_c) in enumerate(zip(clause_vars, clause_signs)):
                    for vi, si in zip(vars_c, signs_c):
                        var_to_clauses[vi].append((ci, si))

        # Check if all satisfied
        for ci in range(m):
            if len(clause_vars[ci]) == 0:
                return None
            if len(clause_vars[ci]) == 1:
                # Should have been handled by unit propagation, but check
                v = clause_vars[ci][0]
                if assignment[v] is not None:
                    continue
        # Actually let's just check if any clause is unsatisfied
        for ci in range(m):
            all_false = True
            for vi, si in zip(clause_vars[ci], clause_signs[ci]):
                av = assignment[vi]
                if av is None:
                    all_false = False
                    break
                if av == (si == 1):
                    all_false = False
                    break
            if all_false and len(clause_vars[ci]) > 0:
                # All assigned and false -> contradiction
                return None
            if all_false and len(clause_vars[ci]) == 0:
                return None

        # Check if all variables assigned
        all_assigned = True
        for v in range(n):
            if assignment[v] is None:
                all_assigned = False
                break
        if all_assigned:
            return assignment[:]

        # Pure literal elimination
        pure_literal_found = True
        while pure_literal_found:
            pure_literal_found = False
            for v in range(n):
                if assignment[v] is not None:
                    continue
                pos = False
                neg = False
                for ci, si in var_to_clauses[v]:
                    if si == 1:
                        pos = True
                    else:
                        neg = True
                    if pos and neg:
                        break
                if pos and not neg:
                    assignment[v] = True
                    trail.append(v)
                    pure_literal_found = True
                elif neg and not pos:
                    assignment[v] = False
                    trail.append(v)
                    pure_literal_found = True
            if pure_literal_found:
                # Re-run unit propagation and rebuild
                changed = True
                while changed:
                    changed = False
                    for ci in range(m):
                        if len(clause_vars[ci]) == 0:
                            return None
                        if len(clause_vars[ci]) == 1:
                            v = clause_vars[ci][0]
                            s = clause_signs[ci][0]
                            val = (s == 1)
                            if assignment[v] is None:
                                assignment[v] = val
                                trail.append(v)
                                changed = True
                            elif assignment[v] != val:
                                return None
                    if changed:
                        new_cv = []
                        new_cs = []
                        for ci in range(m):
                            sat = False
                            nv = []
                            ns = []
                            for vi, si in zip(clause_vars[ci], clause_signs[ci]):
                                av = assignment[vi]
                                if av is not None:
                                    if av == (si == 1):
                                        sat = True
                                        break
                                else:
                                    nv.append(vi)
                                    ns.append(si)
                            if not sat:
                                new_cv.append(nv)
                                new_cs.append(ns)
                        clause_vars = new_cv
                        clause_signs = new_cs
                        for i in range(n):
                            var_to_clauses[i] = []
                        for ci, (vc, sc) in enumerate(zip(clause_vars, clause_signs)):
                            for vi, si in zip(vc, sc):
                                var_to_clauses[vi].append((ci, si))

        # Check again if solved
        for ci in range(m):
            if len(clause_vars[ci]) == 0:
                return None
            all_f = True
            for vi, si in zip(clause_vars[ci], clause_signs[ci]):
                av = assignment[vi]
                if av is None:
                    all_f = False
                    break
                if av == (si == 1):
                    all_f = False
                    break
            if all_f:
                return None

        all_assigned = True
        for v in range(n):
            if assignment[v] is None:
                all_assigned = False
                break
        if all_assigned:
            return assignment[:]

        # Choose a variable to branch on
        # Pick variable appearing in most unsatisfied clauses
        best_var = -1
        best_count = 0
        for v in range(n):
            if assignment[v] is not None:
                continue
            cnt = 0
            for ci, si in var_to_clauses[v]:
                cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_var = v

        if best_var == -1:
            return assignment[:]

        # Try True first (planted instances often have bias)
        for val in [True, False]:
            assignment[best_var] = val
            trail.append(best_var)

            # Save state
            saved_cv = clause_vars[:]
            saved_cs = clause_signs[:]
            saved_vtc = [lst[:] for lst in var_to_clauses]

            result = dpll(assignment, trail, depth + 1)
            if result is not None:
                return result

            # Backtrack
            assignment[best_var] = None
            trail.pop()
            clause_vars = saved_cv
            clause_signs = saved_cs
            var_to_clauses = saved_vtc

        return None

    # Try DPLL with a few different random orderings
    best_result = None
    for attempt in range(5):
        if time.time() > deadline:
            break
        assignment = [None] * n
        trail = []
        # Restore original clauses
        clause_vars_orig = clause_vars[:]
        clause_signs_orig = clause_signs[:]
        clause_vars = clause_vars_orig[:]
        clause_signs = clause_signs_orig[:]
        for i in range(n):
            var_to_clauses[i] = []
        for ci, (vc, sc) in enumerate(zip(clause_vars, clause_signs)):
            for vi, si in zip(vc, sc):
                var_to_clauses[vi].append((ci, si))

        result = dpll(assignment, trail, 0)
        if result is not None:
            # Convert to boolean list
            return [bool(x) for x in result]

    # Fallback: WalkSAT with time limit
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    def count_unsat(assign):
        cnt = 0
        for c in clauses:
            sat = False
            for l in c:
                if assign[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    cur_unsat = count_unsat(assignment)
    best_assign = assignment[:]
    best_unsat = cur_unsat

    while time.time() < deadline:
        if cur_unsat == 0:
            return assignment
        # Pick a random unsatisfied clause
        unsat_clauses = []
        for ci, c in enumerate(clauses):
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        if not unsat_clauses:
            return assignment
        ci = rng.choice(unsat_clauses)
        c = clauses[ci]
        # With probability 0.5, flip a variable that satisfies the clause; else random
        if rng.random() < 0.5:
            # Flip a literal that is currently false in this clause
            candidates = []
            for l in c:
                if assignment[abs(l) - 1] != (l > 0):
                    candidates.append(abs(l) - 1)
            if candidates:
                v = rng.choice(candidates)
                assignment[v] = not assignment[v]
            else:
                v = rng.choice([abs(l) - 1 for l in c])
                assignment[v] = not assignment[v]
        else:
            v = rng.randrange(n)
            assignment[v] = not assignment[v]

        new_unsat = count_unsat(assignment)
        if new_unsat < cur_unsat:
            cur_unsat = new_unsat
            if new_unsat < best_unsat:
                best_unsat = new_unsat
                best_assign = assignment[:]
        else:
            # Accept with some probability or revert
            if new_unsat <= cur_unsat + 1:
                cur_unsat = new_unsat
            else:
                assignment = best_assign[:]
                cur_unsat = best_unsat

    return best_assign
