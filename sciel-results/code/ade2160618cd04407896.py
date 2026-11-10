"""Planted 3-SAT solver (50 variables) with WalkSAT heuristic and multi-restart."""
import time
import random
import sys


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # safety margin before 330s kill

    # Preprocess: build clause list with 0-based indices, track variable appearances
    # Convert clauses to lists of (var_index, is_positive)
    cl_data = []
    for c in clauses:
        entry = []
        for lit in c:
            v = abs(lit) - 1
            pos = (lit > 0)
            entry.append((v, pos))
        cl_data.append(entry)

    # Build variable-to-clause incidence for efficiency
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(cl_data):
        for v, _ in c:
            var_clauses[v].append(ci)

    best_assignment = None
    best_unsat = m + 1

    # WalkSAT with clause-weighted heuristic
    def walk_sat_init(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        return assignment

    def count_unsat(assignment, cl_data):
        cnt = 0
        for c in cl_data:
            sat = False
            for v, pos in c:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def walk_sat(assignment, cl_data, var_clauses, n, rng, max_steps, noise=0.5):
        """WalkSAT step: pick random unsat clause, flip var with clause-weighted heuristic."""
        for step in range(max_steps):
            # Find unsatisfied clauses
            unsat = []
            for ci, c in enumerate(cl_data):
                sat = False
                for v, pos in c:
                    if assignment[v] == pos:
                        sat = True
                        break
                if not sat:
                    unsat.append(ci)

            if not unsat:
                return assignment, 0

            # Pick random unsatisfied clause
            ci = rng.choice(unsat)
            clause = cl_data[ci]

            # With probability noise, pick random var from clause
            # Otherwise, pick var that satisfies most other clauses (weighted by 1/clause_size)
            if rng.random() < noise:
                v, _ = rng.choice(clause)
            else:
                best_gain = -1
                best_v = -1
                for v, pos in clause:
                    # Compute gain: how many currently-unsatisfied clauses would be satisfied
                    # if we set variable v to satisfy this clause
                    # i.e., flip v to the value that satisfies this literal
                    target_val = pos
                    if assignment[v] == target_val:
                        # Already satisfies this clause (shouldn't happen for unsat clause, but safety)
                        gain = 0
                    else:
                        gain = 0
                        for other_ci in var_clauses[v]:
                            if other_ci == ci:
                                continue
                            other_clause = cl_data[other_ci]
                            # Check if this other clause is currently unsatisfied
                            other_sat = False
                            for ov, opos in other_clause:
                                if assignment[ov] == opos:
                                    other_sat = True
                                    break
                            if not other_sat:
                                # Would setting v to target_val satisfy this other clause?
                                # Only if this other clause has the literal (v, target_val)
                                for ov, opos in other_clause:
                                    if ov == v and opos == target_val:
                                        gain += 1.0 / len(other_clause)
                                        break
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v >= 0:
                    v = best_v
                else:
                    v, _ = rng.choice(clause)

            # Flip variable
            assignment[v] = not assignment[v]

        # Final count
        cnt = 0
        for c in cl_data:
            sat = False
            for v, pos in c:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return assignment, cnt

    # Multi-restart strategy
    rng = random.Random()
    restart_count = 0
    steps_per_restart = 5000

    while time.time() < deadline:
        assignment = walk_sat_init(rng)
        steps = steps_per_restart if restart_count < 5 else 3000
        assignment, unsat_count = walk_sat(assignment, cl_data, var_clauses, n, rng, steps, noise=0.3)

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        if best_unsat == 0:
            break

        restart_count += 1
        # Gradually reduce noise to converge
        if restart_count % 10 == 0 and steps_per_restart > 1000:
            steps_per_restart = max(1000, steps_per_restart // 2)

    if best_assignment is None:
        best_assignment = [False] * n

    return best_assignment
