"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Schöning-style biased random restarts with incremental clause tracking.
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280.0  # safety margin before 300s kill

    # Precompute clauses as (var_idx, sign) tuples
    clauses_pre = []
    for c in clauses:
        clauses_pre.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses_pre):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(i)

    rng = random.Random(12345)

    def make_initial_assignment():
        # Biased coin: p=2/3 for True, 1/3 for False (Schöning)
        return [rng.random() < 2.0 / 3.0 for _ in range(n)]

    def compute_unsat(assignment):
        unsat_counts = []
        for clause in clauses_pre:
            cnt = 0
            for var_idx, sign in clause:
                val = assignment[var_idx]
                if (sign == 1 and val) or (sign == -1 and not val):
                    cnt += 1
            unsat_counts.append(3 - cnt)
        return unsat_counts

    def total_unsat(unsat_counts):
        return sum(unsat_counts)

    best_assignment = None
    best_unsat = m + 1

    while time.time() < deadline:
        assignment = make_initial_assignment()
        unsat_counts = compute_unsat(assignment)
        cur_unsat = total_unsat(unsat_counts)

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Local search with incremental updates
        max_flips = 8000
        for _ in range(max_flips):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return assignment

            # Pick a random unsatisfied clause
            # Use rejection sampling: pick random clause, retry if satisfied
            for _try in range(10):
                clause_idx = rng.randrange(m)
                if unsat_counts[clause_idx] > 0:
                    break

            if unsat_counts[clause_idx] == 0:
                continue

            clause = clauses_pre[clause_idx]
            # Pick a literal to flip (uniform among 3, Schöning style)
            var_idx, sign = clause[rng.randrange(3)]

            # Flip the variable
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Update unsat counts for clauses containing var_idx
            for ci in var_to_clauses[var_idx]:
                cl = clauses_pre[ci]
                cnt = 0
                for v, s in cl:
                    val = assignment[v]
                    if (s == 1 and val) or (s == -1 and not val):
                        cnt += 1
                unsat_counts[ci] = 3 - cnt

            cur_unsat = total_unsat(unsat_counts)

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
