"""Planted 3-SAT solver: spectral init + incremental WalkSAT + SA escapes + repair."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin vs 330s kill

    # Pre-index: for each variable, list of (clause_idx, literal)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_clauses[abs(lit) - 1].append((ci, lit))

    # Incremental state tracking
    def init_state(a):
        """Initialize clause satisfaction counts and unsat list from assignment a."""
        clause_sat_count = [0] * m
        unsat = []
        for ci, cl in enumerate(clauses):
            cnt = 0
            for lit in cl:
                if a[abs(lit) - 1] == (lit > 0):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat.append(ci)
        return clause_sat_count, unsat

    def update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set):
        """Update clause state after flipping variable v from old_val to new_val."""
        for (ci, lit) in var_clauses[v]:
            old_sat = (old_val == (lit > 0))
            new_sat = (new_val == (lit > 0))
            if old_sat == new_sat:
                continue
            old_cnt = clause_sat_count[ci]
            new_cnt = old_cnt + (1 if new_sat else -1)
            clause_sat_count[ci] = new_cnt
            if old_cnt == 0 and new_cnt > 0:
                # Clause became satisfied
                if ci in unsat_set:
                    unsat_set.discard(ci)
                    # Remove from unsat list (lazy deletion)
            elif old_cnt > 0 and new_cnt == 0:
                # Clause became unsatisfied
                unsat.append(ci)
                unsat_set.add(ci)
        # Clean up unsat list: remove clauses that are now satisfied
        cleaned = []
        for ci in unsat:
            if clause_sat_count[ci] == 0:
                cleaned.append(ci)
        unsat[:] = cleaned

    best_assignment = None
    best_unsat = m + 1

    def update_best(a):
        nonlocal best_assignment, best_unsat
        clause_sat_count, unsat = init_state(a)
        u = len(unsat)
        if u < best_unsat:
            best_unsat = u
            best_assignment = a[:]
        return u

    rng = random.Random(12345)

    def spectral_init():
        """Use spectral method on literal co-occurrence matrix for initial assignment."""
        # Build literal co-occurrence matrix: M[i][j] = number of clauses where
        # literal i and literal j appear together (with same sign preference)
        # For planted SAT, correlated literals in same clause suggest same truth value
        size = 2 * n
        # Use a sparse-friendly approach: build co-occurrence counts
        # M[lit_a][lit_b] += 1 for each clause containing both lit_a and lit_b
        # where lit is 1-based signed
        # We'll use numpy for the eigenvector computation
        M = np.zeros((size, size), dtype=np.float64)
        for cl in clauses:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    li = cl[i]
                    lj = cl[j]
                    # Convert to 0-based index in [0, 2n)
                    # Positive literal v -> index v-1, negative literal -v -> index n+v-1
                    idx_i = li - 1 if li > 0 else n + li - 1
                    idx_j = lj - 1 if lj > 0 else n + lj - 1
                    M[idx_i, idx_j] += 1
                    M[idx_j, idx_i] += 1
        # Add diagonal to make it positive semi-definite for power iteration
        np.fill_diagonal(M, 0.0)

        # Power iteration for principal eigenvector
        v = np.random.RandomState(42).rand(size)
        v /= np.linalg.norm(v)
        for _ in range(50):
            v_new = M @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-15:
                break
            v_new /= norm
            # Ensure consistent sign
            if v_new[0] < 0:
                v_new = -v_new
            v = v_new

        # Convert eigenvector to boolean assignment
        # For variable v (0-based), positive literal index = v, negative = n+v
        # If eigenvector component for positive literal > negative, set to True
        a = [False] * n
        for var in range(n):
            pos_val = v[var]
            neg_val = v[n + var]
            a[var] = pos_val > neg_val
        return a

    def walk_sat(a, max_steps):
        """Incremental WalkSAT with clause weights."""
        clause_weights = [1.0] * m
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)

        for step in range(max_steps):
            if time.time() > deadline:
                return a

            # Clean unsat list
            if len(unsat) != len(unsat_set):
                unsat[:] = [ci for ci in unsat if clause_sat_count[ci] == 0]
                unsat_set = set(unsat)

            if not unsat:
                return a

            # Pick a clause weighted by clause_weights
            total_w = sum(clause_weights[ci] for ci in unsat)
            if total_w <= 0:
                ci = rng.choice(unsat)
            else:
                r = rng.random() * total_w
                acc = 0.0
                ci = unsat[-1]
                for c in unsat:
                    acc += clause_weights[c]
                    if r <= acc:
                        ci = c
                        break
            cl = clauses[ci]

            # Score each literal in the clause
            best_var = None
            best_score = -1e18
            best_val = None
            for lit in cl:
                v = abs(lit) - 1
                desired = (lit > 0)
                if a[v] == desired:
                    continue
                # Flipping v to desired
                gain = 0.0
                loss = 0.0
                for (ci2, lit2) in var_clauses[v]:
                    old_sat = (a[v] == (lit2 > 0))
                    new_sat = (desired == (lit2 > 0))
                    if old_sat == new_sat:
                        continue
                    old_cnt = clause_sat_count[ci2]
                    new_cnt = old_cnt + (1 if new_sat else -1)
                    if old_cnt == 0 and new_cnt > 0:
                        gain += clause_weights[ci2]
                    elif old_cnt > 0 and new_cnt == 0:
                        loss += clause_weights[ci2]
                score = gain - 1.0 * loss
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_val = desired

            if best_var is None:
                v = abs(cl[0]) - 1
                old_val = a[v]
                new_val = not old_val
                a[v] = new_val
                update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set)
                continue

            old_val = a[best_var]
            a[best_var] = best_val
            update_state(best_var, old_val, best_val, clause_sat_count, unsat, unsat_set)

            # Update clause weights
            clause_weights[ci] *= 1.05
            if step % 500 == 0:
                max_w = max(clause_weights) if clause_weights else 1.0
                if max_w > 0:
                    clause_weights = [w / max_w for w in clause_weights]

        return a

    def sa_escape(a, steps, t0):
        """Simulated annealing escape with incremental updates."""
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)

        for step in range(steps):
            if time.time() > deadline:
                return a

            if not unsat:
                return a

            ci = rng.choice(unsat)
            cl = clauses[ci]
            v = abs(rng.choice(cl)) - 1
            new_val = not a[v]
            old_val = a[v]

            # Compute delta in # unsat
            old_unsat_count = len(unsat_set)
            a[v] = new_val
            update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set)
            new_unsat_count = len(unsat_set)
            delta = new_unsat_count - old_unsat_count

            if delta <= 0 or rng.random() < math.exp(-delta / max(t0, 1e-9)):
                pass  # accept
            else:
                a[v] = old_val
                update_state(v, new_val, old_val, clause_sat_count, unsat, unsat_set)

            t0 *= 0.9995
        return a

    # Main loop with restarts
    restart = 0
    base_steps = 5000

    # First: try spectral initialization
    if time.time() < deadline - 10.0:
        a_spec = spectral_init()
        a_spec = walk_sat(a_spec, 10000)
        u_spec = update_best(a_spec)
        if u_spec == 0:
            return a_spec

    while time.time() < deadline:
        if restart == 0:
            # Second: try a few more spectral variants with different random seeds
            a = spectral_init()
        else:
            a = [rng.random() < 0.5 for _ in range(n)]
        max_steps = base_steps + restart * 1000
        a = walk_sat(a, max_steps)
        u = update_best(a)
        if u == 0:
            return a

        # If stuck, try SA escape from current state
        if time.time() < deadline - 5.0:
            a2 = a[:]
            a2 = sa_escape(a2, 2000, 5.0)
            u2 = update_best(a2)
            if u2 == 0:
                return a2

        restart += 1
        if restart > 200:
            break

    # Final: try repair from best assignment
    if best_assignment is not None and time.time() < deadline - 2.0:
        a = best_assignment[:]
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)
        # Greedy repair: repeatedly flip variable that reduces unsat most
        for _ in range(500):
            if time.time() > deadline:
                break
            if not unsat:
                return a
            # Find variable flip that reduces unsat the most
            best_v = -1
            best_gain = 0
            best_val = None
            candidates = list(range(n))
            rng.shuffle(candidates)
            for v in candidates[:80]:
                for val in (True, False):
                    if a[v] == val:
                        continue
                    # Simulate flip
                    old_val = a[v]
                    a[v] = val
                    update_state(v, old_val, val, clause_sat_count, unsat, unsat_set)
                    u = len(unsat_set)
                    gain = best_unsat - u
                    a[v] = old_val
                    update_state(v, val, old_val, clause_sat_count, unsat, unsat_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_val = val
            if best_v >= 0:
                old_val = a[best_v]
                a[best_v] = best_val
                update_state(best_v, old_val, best_val, clause_sat_count, unsat, unsat_set)
                update_best(a)
            else:
                break

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
