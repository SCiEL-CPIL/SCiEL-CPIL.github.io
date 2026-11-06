"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
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

    # Pre-process: build per-variable clause lists and per-clause literal data
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_pol = []  # True if positive literal

    for i, c in enumerate(clauses):
        cv = []
        cp = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            cv.append(v)
            cp.append(p)
            var_to_clauses[v].append(i)
        clause_vars.append(cv)
        clause_pol.append(cp)

    # State
    assignment = [False] * n
    sat_count = [0] * m  # number of satisfying literals per clause
    unsat = []  # list of unsatisfied clause indices

    def init_state(asm):
        unsat.clear()
        for i in range(m):
            cnt = 0
            cv = clause_vars[i]
            cp = clause_pol[i]
            for j in range(len(cv)):
                if asm[cv[j]] == cp[j]:
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat.append(i)

    def flip(v):
        assignment[v] = not assignment[v]
        for ci in var_to_clauses[v]:
            cv = clause_vars[ci]
            cp = clause_pol[ci]
            cnt = 0
            for j in range(len(cv)):
                if assignment[cv[j]] == cp[j]:
                    cnt += 1
            was_sat = sat_count[ci] > 0
            sat_count[ci] = cnt
            is_sat = cnt > 0
            if was_sat and not is_sat:
                unsat.append(ci)
            elif not was_sat and is_sat:
                try:
                    unsat.remove(ci)
                except ValueError:
                    pass

    def breakage(v):
        """Number of currently-sat clauses that become unsat if v flips."""
        b = 0
        for ci in var_to_clauses[v]:
            if sat_count[ci] > 0:
                # Check if v is the only satisfying literal
                cv = clause_vars[ci]
                cp = clause_pol[ci]
                v_satisfies = False
                for j in range(len(cv)):
                    if cv[j] == v:
                        if assignment[v] == cp[j]:
                            v_satisfies = True
                        break
                if v_satisfies and sat_count[ci] == 1:
                    b += 1
        return b

    def gain(v):
        """Number of currently-unsat clauses that become sat if v flips."""
        g = 0
        for ci in var_to_clauses[v]:
            if sat_count[ci] == 0:
                cv = clause_vars[ci]
                cp = clause_pol[ci]
                for j in range(len(cv)):
                    if cv[j] == v:
                        if (not assignment[v]) == cp[j]:
                            g += 1
                        break
        return g

    def exact_local_solve(max_iters=5000):
        """
        Try to find a satisfying assignment by flipping only variables
        that appear in currently unsatisfied clauses, preferring flips
        that reduce unsat count. Returns (assignment, unsat_count) or None.
        """
        if not unsat:
            return assignment[:], 0

        cur_unsat = len(unsat)
        best_asm = assignment[:]
        best_unsat = cur_unsat

        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not unsat:
                return assignment[:], 0

            # Collect all variables in unsat clauses
            candidate_vars = set()
            for ci in unsat:
                for v in clause_vars[ci]:
                    candidate_vars.add(v)

            if not candidate_vars:
                break

            # For each candidate, compute net gain (unsat reduced - sat broken)
            best_v = None
            best_net = -10**9
            for v in candidate_vars:
                # Compute gain: unsat clauses that would become sat
                g = 0
                for ci in var_to_clauses[v]:
                    if sat_count[ci] == 0:
                        cv = clause_vars[ci]
                        cp = clause_pol[ci]
                        for j in range(len(cv)):
                            if cv[j] == v:
                                if (not assignment[v]) == cp[j]:
                                    g += 1
                                break

                # Compute breakage: sat clauses that would become unsat
                b = 0
                for ci in var_to_clauses[v]:
                    if sat_count[ci] > 0:
                        cv = clause_vars[ci]
                        cp = clause_pol[ci]
                        v_satisfies = False
                        for j in range(len(cv)):
                            if cv[j] == v:
                                if assignment[v] == cp[j]:
                                    v_satisfies = True
                                break
                        if v_satisfies and sat_count[ci] == 1:
                            b += 1

                net = g - b
                if net > best_net:
                    best_net = net
                    best_v = v

            if best_v is None or best_net <= 0:
                break

            flip(best_v)
            cur_unsat = len(unsat)

            if cur_unsat == 0:
                return assignment[:], 0

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_asm = assignment[:]

        return best_asm, best_unsat

    rng = random.Random(42)
    best_asm = None
    best_unsat = m + 1

    def local_search(noise, max_steps, stuck_limit):
        nonlocal assignment, unsat
        cur = len(unsat)
        if cur == 0:
            return assignment[:], 0

        best_local = cur
        best_local_asm = assignment[:]
        stuck = 0
        steps = 0

        while steps < max_steps:
            if time.time() > deadline - 3.0:
                break
            if not unsat:
                return assignment[:], 0

            ci = rng.choice(unsat)
            cv = clause_vars[ci]

            if rng.random() < noise:
                v = rng.choice(cv)
            else:
                # Weighted selection: minimize breakage, maximize gain
                scores = []
                for v in cv:
                    b = breakage(v)
                    g = gain(v)
                    # Soft score: penalize breakage heavily, reward gain
                    score = b * 2.0 - g * 1.5 + rng.random() * 0.5
                    scores.append((score, v))
                scores.sort()
                v = scores[0][1]

            flip(v)
            steps += 1
            cur = len(unsat)

            if cur == 0:
                return assignment[:], 0

            if cur < best_local:
                best_local = cur
                best_local_asm = assignment[:]
                stuck = 0
            else:
                stuck += 1

            if stuck > stuck_limit:
                break

        return best_local_asm, best_local

    max_restarts = 2000
    max_steps = 50000
    noise = 0.3

    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]
        init_state(assignment)
        cur = len(unsat)

        if cur == 0:
            return assignment[:]

        if cur < best_unsat:
            best_unsat = cur
            best_asm = assignment[:]

        assignment, cur = local_search(noise, max_steps, 8000)

        if cur == 0:
            return assignment[:]

        if cur < best_unsat:
            best_unsat = cur
            best_asm = assignment[:]

        # If close to solution, try exact local solve
        if best_unsat <= max(5, m // 40) and time.time() < deadline - 15.0:
            assignment = best_asm[:]
            init_state(assignment)
            cur = len(unsat)
            if cur == 0:
                return assignment[:]

            # Try exact local solve
            exact_result = exact_local_solve(max_iters=10000)
            if exact_result is not None:
                exact_asm, exact_cur = exact_result
                if exact_cur == 0:
                    return exact_asm
                if exact_cur < best_unsat:
                    best_unsat = exact_cur
                    best_asm = exact_asm

            # Also try low-noise local search
            assignment, cur = local_search(0.1, 100000, 20000)
            if cur == 0:
                return assignment[:]
            if cur < best_unsat:
                best_unsat = cur
                best_asm = assignment[:]

    # Final polish: if close to solution, do extended low-noise search
    if best_asm and best_unsat > 0 and time.time() < deadline - 5.0:
        assignment = best_asm[:]
        init_state(assignment)
        cur = len(unsat)
        if cur == 0:
            return assignment[:]

        # Try exact local solve first
        exact_result = exact_local_solve(max_iters=20000)
        if exact_result is not None:
            exact_asm, exact_cur = exact_result
            if exact_cur == 0:
                return exact_asm
            if exact_cur < best_unsat:
                best_unsat = exact_cur
                best_asm = exact_asm

        # Progressive noise reduction for final polish
        for polish_noise in [0.05, 0.02, 0.0]:
            if time.time() > deadline - 5.0:
                break
            assignment, cur = local_search(polish_noise, 200000, 50000)
            if cur == 0:
                return assignment[:]
            if cur < best_unsat:
                best_unsat = cur
                best_asm = assignment[:]

    if best_asm:
        return best_asm
    return assignment

# EVOLVE-BLOCK-END
