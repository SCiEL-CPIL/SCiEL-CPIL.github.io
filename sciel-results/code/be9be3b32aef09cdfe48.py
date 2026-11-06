"""Rewritten planted 3-SAT solver; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    # Deadline guard: evaluator kills at 330s wall-clock; keep a safety margin.
    start_time = time.time()
    deadline = start_time + 320.0

    m = len(clauses)
    if m == 0:
        return [True] * n

    # ---- Build clause structure (signed 1-based literals) ----
    # For each clause: vars (0-based), signs (+1 / -1)
    cv = np.empty((m, 3), dtype=np.int32)   # variable index (0-based)
    cs = np.empty((m, 3), dtype=np.int8)    # sign: +1 (pos) or -1 (neg)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            if lit > 0:
                cv[i, j] = lit - 1
                cs[i, j] = 1
            else:
                cv[i, j] = -lit - 1
                cs[i, j] = -1

    # ---- Variable -> list of (clause, position) occurrences ----
    var_occ = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_occ[cv[i, j]].append((i, j))

    # For fast random access, also store per-variable occurrence arrays
    var_occ_arr = [np.array(v, dtype=np.int32) for v in var_occ]  # each (deg, 2)

    # ---- Incremental clause satisfaction state ----
    # sat_count[i] = number of true literals in clause i (0..3)
    # A clause is satisfied iff sat_count[i] >= 1.
    # cur_true[i]  = whether clause i is currently satisfied (bool)

    def init_state(assign):
        """assign: np.array of bools. Build sat_count and cur_true."""
        sat_count = np.zeros(m, dtype=np.int8)
        for v in range(n):
            av = assign[v]
            occ = var_occ[v]
            if not occ:
                continue
            for (i, j) in occ:
                # literal true if (sign==1 and av) or (sign==-1 and not av)
                if (cs[i, j] == 1) == av:
                    sat_count[i] += 1
        cur_true = sat_count >= 1
        return sat_count, cur_true

    def check_all(sat_count):
        return int(np.count_nonzero(sat_count == 0)) == 0

    # ---- Random walk with incremental updates ----
    def walk(assign, max_flips, rng, p_walk):
        """
        assign: np.array of bools (modified in place).
        Returns True if a satisfying assignment is found.
        """
        sat_count, cur_true = init_state(assign)
        unsat = int(np.count_nonzero(~cur_true))
        if unsat == 0:
            return True

        for _ in range(max_flips):
            if time.time() > deadline - 2.0:
                return False

            # Pick a random unsatisfied clause
            # We keep a list of unsat clause indices for fast random pick.
            # Rebuild lazily is expensive; instead pick by sampling.
            # Strategy: maintain an unsat index buffer.
            # For simplicity & speed, sample until we hit an unsat clause.
            # Expected unsat fraction can be small late; use a cached buffer.
            # We'll maintain `unsat_buf` as a list; rebuild when empty or stale.
            # Simpler robust approach: keep a list and refresh periodically.
            # Use a global-ish buffer passed via closure would be cleaner; here rebuild
            # only when needed.

            # Choose clause: sample a random clause; if unsat, use it; else retry (bounded).
            clause_idx = -1
            for _ in range(12):
                cand = rng.randrange(m)
                if sat_count[cand] == 0:
                    clause_idx = cand
                    break
            if clause_idx == -1:
                # Fallback: scan (rare)
                nz = np.where(sat_count == 0)[0]
                if len(nz) == 0:
                    return True
                clause_idx = nz[rng.randrange(len(nz))]

            # Choose which variable in the clause to flip.
            # Score each of the 3 literals: prefer flipping a variable that
            # appears in many other unsatisfied clauses (helps global progress),
            # with a random walk probability to escape local minima.
            # Compute per-literal "impact" = number of unsat clauses containing that var.
            # We approximate with a cached per-variable unsat-occurrence count.
            # Maintain var_unsat_count[v] = number of unsat clauses where v appears.
            # Update incrementally on each flip.

            # (var_unsat_count is maintained below)
            v0 = cv[clause_idx, 0]
            v1 = cv[clause_idx, 1]
            v2 = cv[clause_idx, 2]
            s0 = var_unsat_count[v0]
            s1 = var_unsat_count[v1]
            s2 = var_unsat_count[v2]

            # Weighted selection (smoothed to avoid zero weights)
            w0 = s0 + 1
            w1 = s1 + 1
            w2 = s2 + 1
            total = w0 + w1 + w2
            r = rng.random() * total
            if r < w0:
                flip_v = v0
            elif r < w0 + w1:
                flip_v = v1
            else:
                flip_v = v2

            # With probability p_walk, instead pick a purely random variable
            # from the clause (to escape plateaus).
            if rng.random() < p_walk:
                flip_v = clause_vars_triplet[clause_idx][rng.randrange(3)]

            # ---- Apply flip: update sat_count for all clauses containing flip_v ----
            occ = var_occ[flip_v]
            for (i, j) in occ:
                was_true = (cs[i, j] == 1) == assign[flip_v]
                new_sat = sat_count[i]
                if was_true:
                    # literal becomes false
                    new_sat -= 1
                else:
                    # literal becomes true
                    new_sat += 1
                sat_count[i] = new_sat
                was_cur = cur_true[i]
                is_cur = new_sat >= 1
                if was_cur != is_cur:
                    cur_true[i] = is_cur
                    # update var_unsat_count for the three vars of clause i
                    if is_cur:
                        # clause became satisfied: decrement unsat count of its 3 vars
                        var_unsat_count[cv[i, 0]] -= 1
                        var_unsat_count[cv[i, 1]] -= 1
                        var_unsat_count[cv[i, 2]] -= 1
                    else:
                        # clause became unsatisfied: increment
                        var_unsat_count[cv[i, 0]] += 1
                        var_unsat_count[cv[i, 1]] += 1
                        var_unsat_count[cv[i, 2]] += 1

            assign[flip_v] = not assign[flip_v]

            # Track unsat count quickly
            # (we don't maintain a global counter; check via a cheap heuristic)
            # To avoid O(m) scan each step, maintain unsat_total incrementally.
            # We'll update it in the loop above via cur_true transitions.

        # Final check
        return int(np.count_nonzero(sat_count == 0)) == 0

    # We need var_unsat_count and clause_vars_triplet in scope for walk().
    # Rebuild walk to capture them via closure by defining inside a setup that
    # initializes these arrays. To keep it clean, we restructure: do everything
    # in the main loop with local arrays.

    # ============ MAIN SEARCH ============
    rng = random.Random(0)

    def run_search():
        global deadline
        # Per-variable unsat-occurrence count, maintained incrementally.
        # Precompute triplet lists for random access.
        clause_vars_triplet = [cv[i].tolist() for i in range(m)]

        # Precompute per-variable occurrence as list of (clause, sign) for fast flip
        # (var_occ already holds (i, j); we need sign lookup via cs[i,j])
        # For speed, store per-var list of (clause_idx, sign)
        var_occ2 = [[] for _ in range(n)]
        for i in range(m):
            for j in range(3):
                var_occ2[cv[i, j]].append((i, cs[i, j]))

        # Initial random assignment
        assign = np.array([rng.random() < 0.5 for _ in range(n)])

        # Build initial state
        sat_count = np.zeros(m, dtype=np.int8)
        for v in range(n):
            av = assign[v]
            for (i, s) in var_occ2[v]:
                if (s == 1) == av:
                    sat_count[i] += 1
        cur_true = sat_count >= 1

        # var_unsat_count[v] = number of unsatisfied clauses containing v
        var_unsat_count = np.zeros(n, dtype=np.int32)
        for i in range(m):
            if not cur_true[i]:
                var_unsat_count[cv[i, 0]] += 1
                var_unsat_count[cv[i, 1]] += 1
                var_unsat_count[cv[i, 2]] += 1

        unsat_total = int(np.count_nonzero(~cur_true))
        if unsat_total == 0:
            return assign.tolist()

        # Adaptive p_walk: start low (greedy), increase over time to escape.
        p_walk_start = 0.15
        p_walk_end = 0.55
        max_flips = 200000  # generous; deadline will stop us

        for step in range(max_flips):
            if time.time() > deadline - 3.0:
                break

            # Adaptive walk probability
            frac = step / max_flips
            p_walk = p_walk_start + (p_walk_end - p_walk_start) * frac

            # Pick an unsatisfied clause.
            # Use rejection sampling on random clause index; expected cost ~ m/unsat.
            # To keep it bounded, cap retries then fall back to a scan.
            clause_idx = -1
            for _ in range(20):
                cand = rng.randrange(m)
                if sat_count[cand] == 0:
                    clause_idx = cand
                    break
            if clause_idx == -1:
                nz = np.where(sat_count == 0)[0]
                if len(nz) == 0:
                    return assign.tolist()
                clause_idx = nz[rng.randrange(len(nz))]

            # Choose variable to flip within the clause.
            v0, v1, v2 = cv[clause_idx, 0], cv[clause_idx, 1], cv[clause_idx, 2]
            s0 = var_unsat_count[v0]
            s1 = var_unsat_count[v1]
            s2 = var_unsat_count[v2]

            if rng.random() < p_walk:
                # Pure random from the clause
                flip_v = clause_vars_triplet[clause_idx][rng.randrange(3)]
            else:
                # Weighted by unsat-occurrence (smoothed)
                w0 = s0 + 1
                w1 = s1 + 1
                w2 = s2 + 1
                r = rng.random() * (w0 + w1 + w2)
                if r < w0:
                    flip_v = v0
                elif r < w0 + w1:
                    flip_v = v1
                else:
                    flip_v = v2

            # Apply flip incrementally.
            av = assign[flip_v]
            for (i, s) in var_occ2[flip_v]:
                was_true = (s == 1) == av
                new_sat = sat_count[i] + (1 if not was_true else -1)
                sat_count[i] = new_sat
                is_cur = new_sat >= 1
                if cur_true[i] != is_cur:
                    cur_true[i] = is_cur
                    if is_cur:
                        # became satisfied
                        unsat_total -= 1
                        var_unsat_count[cv[i, 0]] -= 1
                        var_unsat_count[cv[i, 1]] -= 1
                        var_unsat_count[cv[i, 2]] -= 1
                    else:
                        # became unsatisfied
                        unsat_total += 1
                        var_unsat_count[cv[i, 0]] += 1
                        var_unsat_count[cv[i, 1]] += 1
                        var_unsat_count[cv[i, 2]] += 1

            assign[flip_v] = not av

            if unsat_total == 0:
                return assign.tolist()

        # Exhausted budget for this restart
        return None

    # Multiple restarts with different seeds; each restart is a fresh random walk.
    # The incremental state makes each flip O(degree) ~ O(3 * avg_degree).
    best = None
    restart = 0
    while time.time() < deadline - 5.0:
        rng = random.Random(restart * 7919 + 13)
        # Re-seed the inner RNG by reusing `rng` (already randomized per restart)
        res = run_search()
        if res is not None:
            best = res
            break
        restart += 1
        # If we keep failing, the instance may be hard; keep trying with fresh
        # random starts. The deadline guard bounds total time.

    if best is not None:
        return best

    # Fallback: return a random assignment (valid format, likely unsat).
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
