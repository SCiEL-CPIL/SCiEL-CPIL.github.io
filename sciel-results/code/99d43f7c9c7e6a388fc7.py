"""Improved planted 3-SAT solver using support-count WalkSAT with incremental tracking."""
# EVOLVE-BLOCK-START
import random
import time

DEADLINE_FRACTION = 0.95


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # Precompute clause structure
    clause_vars = []
    for clause in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in clause])

    # For each variable, precompute which clauses it appears in and with what sign
    var_clauses = [[] for _ in range(n)]
    for ci, cvars in enumerate(clause_vars):
        for v, sign in cvars:
            var_clauses[v].append((ci, sign))

    # Support count initialization
    def make_support_counts(assignment):
        support = [0] * m
        for ci, cvars in enumerate(clause_vars):
            s = 0
            for v, sign in cvars:
                if assignment[v] == sign:
                    s += 1
            support[ci] = s
        return support

    def flip_gain(v, assignment, support):
        gain = 0
        for ci, sign in var_clauses[v]:
            if assignment[v] == sign:
                if support[ci] == 1:
                    gain -= 1
            else:
                if support[ci] == 0:
                    gain += 1
        return gain

    def apply_flip(v, assignment, support, cur_unsat):
        """Flip v, update support and cur_unsat. Returns new cur_unsat."""
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        delta_unsat = 0
        for ci, sign in var_clauses[v]:
            old_support = support[ci]
            if new_val == sign:
                new_support = old_support + 1
            else:
                new_support = old_support - 1
            support[ci] = new_support
            if old_support == 0 and new_support == 1:
                delta_unsat -= 1
            elif old_support == 1 and new_support == 0:
                delta_unsat += 1
        return cur_unsat + delta_unsat

    # Structural bias initialization
    pos_count = [0] * n
    neg_count = [0] * n
    for cvars in clause_vars:
        for v, sign in cvars:
            if sign:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    rng = random.Random(42)

    def initial_assignment():
        assignment = []
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                p = 0.6
            elif neg_count[v] > pos_count[v]:
                p = 0.4
            else:
                p = 0.5
            assignment.append(rng.random() < p)
        return assignment

    best_assignment = None
    best_unsat = m + 1

    walksat_restarts = 200
    max_steps_per_restart = 2000
    noise_param = 0.3

    for restart in range(walksat_restarts):
        if time.time() > deadline:
            break

        assignment = initial_assignment()
        support = make_support_counts(assignment)
        cur_unsat = sum(1 for s in support if s == 0)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Choose variable to flip
            if rng.random() < noise_param:
                v = rng.randint(0, n - 1)
            else:
                # Greedy: find flip with max gain among all variables
                best_gain = -100
                best_v = -1
                for v in range(n):
                    g = flip_gain(v, assignment, support)
                    if g > best_gain:
                        best_gain = g
                        best_v = v
                if best_v == -1:
                    v = rng.randint(0, n - 1)
                else:
                    v = best_v

            cur_unsat = apply_flip(v, assignment, support, cur_unsat)

            if cur_unsat == 0:
                return assignment

    # Final polish: local search from best
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        support = make_support_counts(assignment)
        cur_unsat = sum(1 for s in support if s == 0)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            if cur_unsat == 0:
                return assignment
            for v in range(n):
                g = flip_gain(v, assignment, support)
                if g > 0:
                    cur_unsat = apply_flip(v, assignment, support, cur_unsat)
                    improved = True
                    if cur_unsat == 0:
                        return assignment
                    break

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
