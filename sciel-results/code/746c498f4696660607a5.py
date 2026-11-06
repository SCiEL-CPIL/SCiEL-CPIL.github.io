# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    start = time.time()
    deadline = start + 290.0  # safety margin before 300s kill
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    if m == 0:
        return [False] * n

    # ---- Build compact signed-literal matrix (row per clause) ----
    lit = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for k, l in enumerate(c):
            lit[i, k] = l
    abs_lit = np.abs(lit)
    pos = lit > 0

    # ---- Precompute variable -> (clause rows, column) for fast damage ----
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for k in range(3):
            var_clauses[abs_lit[i, k] - 1].append((i, k))

    def time_up():
        return time.time() >= deadline

    def is_sat(assign):
        # assign: bool array; clause satisfied if any literal matches
        for i in range(m):
            ok = False
            for k in range(3):
                if assign[abs_lit[i, k] - 1] == pos[i, k]:
                    ok = True
                    break
            if not ok:
                return False
        return True

    # ---- Heuristic 1: planted-assignment voting ----
    # A planted clause has its planted literal true; across all clauses, the
    # sign that appears most often for each variable tends toward the planted value.
    score = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for k in range(3):
            v = abs_lit[i, k] - 1
            if pos[i, k]:
                score[v] += 1
            else:
                score[v] -= 1
    planted = score >= 0  # bool array
    if is_sat(planted):
        return planted.tolist()

    # ---- Vectorized WalkSAT with clause learning across restarts ----
    rng = random.Random()

    # clause_weights: bias toward repeatedly-failing clauses
    clause_weights = np.ones(m, dtype=np.float64)

    best_assign = planted.copy()
    best_unsat = m

    def unsat_count(assign):
        # return number of unsatisfied clauses (vectorized)
        # for each clause, check if any literal is satisfied
        # build sat matrix: sat[i,k] = assign[abs_lit[i,k]-1] == pos[i,k]
        vals = assign[abs_lit - 1]          # (m,3) bool
        sat = vals == pos                   # (m,3) bool
        return int(np.sum(~sat.any(axis=1)))

    def walk(assign, max_steps, base_noise, max_noise):
        nonlocal best_assign, best_unsat
        # init support (satisfied-literal count per clause)
        vals = assign[abs_lit - 1]
        support = (vals == pos).sum(axis=1)  # (m,)
        unsat_mask = support == 0
        cur_unsat = int(unsat_mask.sum())
        if cur_unsat == 0:
            return assign
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = assign.copy()

        # weighted clause selection over unsat clauses
        w = clause_weights * unsat_mask
        wsum = w.sum()
        if wsum <= 0:
            w = unsat_mask.astype(np.float64)
            wsum = w.sum()

        noise = base_noise
        for step in range(max_steps):
            if time_up():
                break
            if cur_unsat == 0:
                return assign

            # pick a random unsat clause (weighted)
            if wsum <= 0:
                idx = int(rng.randrange(m))
                while support[idx] != 0:
                    idx = int(rng.randrange(m))
            else:
                r = rng.random() * wsum
                cum = 0.0
                idx = m - 1
                for j in range(m):
                    cum += w[j]
                    if r <= cum:
                        idx = j
                        break

            clause = lit[idx]
            # compute damage for each of the 3 literals (vectorized over their clauses)
            damages = []
            for k in range(3):
                v = abs_lit[idx, k] - 1
                rows, cols = var_clauses[v]
                if not rows:
                    damages.append(0)
                    continue
                # damage = number of currently-satisfied clauses (support==1) that
                # would lose their only satisfied literal when we flip v to the
                # clause's polarity.
                # A clause c is affected if it contains v. It loses satisfaction
                # only if support[c]==1 and the satisfied literal in c is v.
                r_arr = np.array(rows, dtype=np.int32)
                c_arr = np.array(cols, dtype=np.int32)
                s = support[r_arr]
                # satisfied literal in c is v iff assign[v] == pos[c, col]
                # i.e. current value of v equals the polarity of v in c
                sat_v = (assign[v] == pos[r_arr, c_arr])
                dmg = int(np.sum((s == 1) & sat_v))
                damages.append(dmg)

            # choose flip: greedy with noise
            if rng.random() < noise:
                k = int(rng.randrange(3))
            else:
                k = int(np.argmin(damages))
            v = abs_lit[idx, k] - 1

            # flip v
            assign[v] = not assign[v]

            # update support for clauses containing v
            rows, cols = var_clauses[v]
            r_arr = np.array(rows, dtype=np.int32)
            c_arr = np.array(cols, dtype=np.int32)
            # was v satisfied in c before flip?
            was_sat = (not assign[v]) == pos[r_arr, c_arr]  # before flip value
            # after flip, v satisfied iff assign[v] == pos
            now_sat = assign[v] == pos[r_arr, c_arr]
            support[r_arr] = support[r_arr] + (now_sat.astype(np.int32) - was_sat.astype(np.int32))

            # recompute unsat count
            cur_unsat = int(np.sum(support == 0))
            if cur_unsat == 0:
                return assign
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assign = assign.copy()

            # update weights: increase weight of the chosen (still-unsat) clause
            if support[idx] == 0:
                clause_weights[idx] *= 1.05
            # adaptive noise: raise when stuck, lower when improving
            if step % 200 == 0 and step > 0:
                if cur_unsat >= best_unsat:
                    noise = min(max_noise, noise + 0.03)
                else:
                    noise = max(base_noise, noise - 0.02)
            # keep wsum fresh-ish
            if step % 50 == 0:
                w = clause_weights * (support == 0)
                wsum = w.sum()
                if wsum <= 0:
                    w = (support == 0).astype(np.float64)
                    wsum = w.sum()

        return None

    # ---- Main loop with restarts ----
    restart = 0
    while time.time() < deadline and restart < 400:
        if time_up():
            break
        # fresh random assignment, biased by planted vote
        assign = np.random.random(n) < 0.5
        # blend: start from planted vote with some randomization
        assign = planted ^ (np.random.random(n) < 0.25)

        res = walk(assign, max_steps=20000, base_noise=0.4, max_noise=0.6)
        if res is not None:
            return res.tolist()
        # learn: increase weights of unsat clauses in best_assign
        vals = best_assign[abs_lit - 1]
        sat = (vals == pos).any(axis=1)
        unsat_rows = np.where(~sat)[0]
        if len(unsat_rows):
            clause_weights[unsat_rows] *= 1.1
            # normalize occasionally to avoid overflow
            mx = clause_weights.max()
            if mx > 1e6:
                clause_weights /= mx
        restart += 1

    # ---- Final: return best found ----
    if best_unsat == 0:
        return best_assign.tolist()
    return best_assign.tolist()

# EVOLVE-BLOCK-END
