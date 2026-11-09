"""WalkSAT + DLS hybrid with adaptive restarts for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 300s kill

    # Precompute: for each variable, which clauses contain it (and with what sign)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append((ci, lit > 0))

    # Precompute clause lengths for DLS
    clause_lens = np.array([len(c) for c in clauses])

    # Build a lookup: for each clause, list of (var_idx, is_positive)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            cv.append((abs(lit) - 1, lit > 0))
        clause_vars.append(cv)

    def make_satisfied_array(assignment):
        """Return a boolean array: satisfied[ci] = True if clause ci is satisfied."""
        sat = np.zeros(m, dtype=bool)
        for ci in range(m):
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    sat[ci] = True
                    break
        return sat

    def flip_var(assignment, var_idx, sat):
        """Flip variable var_idx and update sat array incrementally."""
        assignment[var_idx] = not assignment[var_idx]
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == assignment[var_idx]:
                sat[ci] = True
            else:
                # Check if other literals in this clause are satisfied
                still_sat = False
                for v, pos in clause_vars[ci]:
                    if v == var_idx:
                        continue
                    if assignment[v] == pos:
                        still_sat = True
                        break
                sat[ci] = still_sat

    def walksat_step(assignment, sat, rng, p_walk):
        """One WalkSAT step: pick an unsat clause, flip a variable.
        With probability p_walk, use WalkSAT heuristic; otherwise flip randomly."""
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return
        ci = int(rng.choice(unsat_idx))
        clause = clause_vars[ci]
        if rng.random() < p_walk:
            # WalkSAT: pick variable with best (gains - losses)
            best_var = -1
            best_score = -1000
            for v, is_pos in clause:
                # Compute gains and losses if we flip v
                gains = 0
                losses = 0
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if is_pos2 == assignment[v]:
                        # Currently satisfies c2; check if others do too
                        other_sat = False
                        for v2, pos2 in clause_vars[c2]:
                            if v2 == v:
                                continue
                            if assignment[v2] == pos2:
                                other_sat = True
                                break
                        if not other_sat:
                            losses += 1
                    else:
                        gains += 1
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat)
            else:
                v, _ = rng.choice(clause)
                flip_var(assignment, v, sat)
        else:
            v, _ = rng.choice(clause)
            flip_var(assignment, v, sat)

    def dls_step(assignment, sat, rng):
        """DLS step: pick the shortest unsatisfied clause, flip a random var in it."""
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return
        # Find shortest unsat clause
        min_len = 10
        best_ci = int(unsat_idx[0])
        for ci in unsat_idx:
            l = len(clauses[ci])
            if l < min_len:
                min_len = l
                best_ci = int(ci)
                if l <= 1:
                    break
        clause = clause_vars[best_ci]
        v, _ = rng.choice(clause)
        flip_var(assignment, v, sat)

    best_assignment = None
    best_unsat = m + 1

    for restart in range(300):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat = make_satisfied_array(assignment)
        cur_unsat = int(np.sum(~sat))
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        steps_in_restart = 0
        max_steps = 8000
        dls_interval = 150  # do DLS every 150 steps
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.65

        for step in range(max_steps):
            if time.time() > deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            # WalkSAT step
            walksat_step(assignment, sat, rng, p_walk)
            steps_in_restart += 1

            # Periodic DLS for escape
            if steps_in_restart % dls_interval == 0:
                dls_step(assignment, sat, rng)

            # Check progress every 30 steps (more frequent)
            if step % 30 == 0:
                cur_unsat = int(np.sum(~sat))
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    p_walk = max(0.35, p_walk - 0.015)
                else:
                    stall_count += 1
                prev_unsat = cur_unsat

            # Adaptive restart within this restart attempt
            if stall_count > 15:
                break

    return best_assignment if best_assignment else [False] * n
# EVOLVE-BLOCK-END
