# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    # Deadline guard: finish comfortably before the 330s kill.
    deadline = time.time() + 295.0

    m = len(clauses)
    # lit[i] in {-1,0,1}: sign of clause i's literal for variable v (0 = absent)
    lit = np.zeros((m, n), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for l in cl:
            v = abs(l) - 1
            lit[i, v] = 1 if l > 0 else -1

    rng = random.Random(0)
    np_rng = np.random.RandomState(0)

    # Random initial assignment
    x = np_rng.rand(n) < 0.5

    def satisfied_all():
        # sat[i] = any literal in clause i true
        s = x[np.arange(n)]
        # value of literal = lit * (2*s-1); >0 means true
        vals = lit * (2 * s[np.newaxis, :] - 1)
        return np.any(vals > 0, axis=1)

    sat = satisfied_all()
    if not sat.all():
        # Main search loop
        while time.time() < deadline:
            unsat_idx = np.nonzero(~sat)[0]
            if unsat_idx.size == 0:
                break

            # Pick a random unsatisfied clause
            ci = int(np_rng.randint(0, unsat_idx.size))
            c = unsat_idx[ci]
            cvars = np.nonzero(lit[c] != 0)[0]
            if cvars.size == 0:
                break

            # Impact score per variable: how many unsat clauses it appears in
            # with the "fixing" literal (the value that would make it true).
            # If we set x[v] = (lit[c,v] > 0), we fix this clause.
            # Impact = number of other unsat clauses that would also be fixed
            #            minus number of currently-sat clauses that would break.

            # For each candidate variable v in this clause:
            #   desired = lit[c, v] > 0  (the value that satisfies this clause)
            #   if x[v] != desired, flipping fixes this clause.
            #   benefit[v] = count of unsat clauses containing v with lit sign == desired
            #   cost[v]    = count of sat clauses containing v with lit sign != desired
            #   score[v]   = benefit[v] - cost[v]

            unsat_rows = ~sat
            # For each variable v, desired value d_v = lit[c, v] > 0
            # A clause j is fixed by flipping v to d_v if lit[j, v] != 0 and
            #   (x[v] != d_v) and lit[j,v] == (1 if d_v else -1)
            # i.e. setting x[v]=d_v makes literal true when lit[j,v]==d_v_sign.

            # Build desired sign array for candidate vars
            cand = cvars
            dsign = lit[c, cand]  # +1 or -1

            # Current x for cand
            xc = x[cand]
            # which are currently wrong (flipping would fix this clause)
            wrong = (xc != (dsign > 0))

            # benefit: for each cand var, number of unsat clauses where
            #   lit[j, v] == dsign  (flipping to dsign makes it true)
            # We only count clauses that would become satisfied. A clause is
            # satisfied by this flip iff its literal for v becomes true AND it
            # wasn't already satisfied (it's in unsat set).
            # lit[j, v] == dsign means after flip literal is true.
            L = lit  # (m, n)
            # For each cand column v, count over unsat rows where L[j, v] == dsign
            benefit = np.zeros(cand.size, dtype=np.int32)
            cost = np.zeros(cand.size, dtype=np.int32)
            # Vectorize over candidate variables
            # L_unsat = L[unsat_rows]  -> (U, n)
            L_u = L[unsat_rows]
            L_s = L[sat]
            for k, v in enumerate(cand):
                dv = dsign[k]
                benefit[k] = np.count_nonzero(L_u[:, v] == dv)
                cost[k] = np.count_nonzero(L_s[:, v] == -dv)
            score = benefit - cost

            # Only consider flipping variables that are currently wrong for this
            # clause (so the flip actually fixes the chosen clause).
            # Add a large bonus to wrong vars so they're preferred, and random tie-break.
            jitter = np_rng.rand(cand.size) * 0.5
            score = score + jitter
            # Force chosen-clause-fixing: add big bonus to wrong vars
            score = score + 1000.0 * wrong.astype(np.int32)

            kbest = int(np.argmax(score))
            v = int(cand[kbest])
            x[v] = not x[v]

            # Update sat incrementally for affected clauses (those containing v)
            affected = np.nonzero(lit[:, v] != 0)[0]
            for j in affected:
                # recompute satisfaction of clause j
                vals = lit[j] * (2 * x[np.arange(n)] - 1)
                sat[j] = np.any(vals > 0)

    # Final verification / polish: quick local pass to confirm and try to fix
    # any remaining unsat with a targeted scan (bounded).
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        unsat_idx = np.nonzero(~sat)[0]
        if unsat_idx.size == 0:
            break
        # try flipping each variable once to see if it reduces unsat count
        best_v = -1
        best_delta = 0
        cur_unsat = int(np.count_nonzero(~sat))
        for v in range(n):
            if lit[:, v].max() == 0:  # variable not in any clause
                continue
            old = x[v]
            x[v] = not x[v]
            affected = np.nonzero(lit[:, v] != 0)[0]
            for j in affected:
                vals = lit[j] * (2 * x[np.arange(n)] - 1)
                sat[j] = np.any(vals > 0)
            new_unsat = int(np.count_nonzero(~sat))
            delta = cur_unsat - new_unsat
            if delta > best_delta:
                best_delta = delta
                best_v = v
            x[v] = old
        if best_v >= 0:
            x[best_v] = not x[best_v]
            affected = np.nonzero(lit[:, best_v] != 0)[0]
            for j in affected:
                vals = lit[j] * (2 * x[np.arange(n)] - 1)
                sat[j] = np.any(vals > 0)
        else:
            break

    return x.tolist()
# EVOLVE-BLOCK-END
