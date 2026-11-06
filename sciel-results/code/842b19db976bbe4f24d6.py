import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    start_time = time.time()
    deadline = start_time + 320.0

    m = len(clauses)
    if m == 0:
        return [True] * n

    # Preprocess: var_idx[m,3] 0-based, sign[m,3] 1=pos 0=neg
    var_idx = np.empty((m, 3), dtype=np.int32)
    sign = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            var_idx[i, j] = abs(lit) - 1
            sign[i, j] = 1 if lit > 0 else 0

    # Build variable -> clause/pos lookup
    var_clauses = [[] for _ in range(n)]
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            var_clauses[v].append(i)
            var_pos[v].append(j)

    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.empty(0, dtype=np.int32) for vc in var_clauses]
    var_pos_np = [np.array(vp, dtype=np.int32) if vp else np.empty(0, dtype=np.int32) for vp in var_pos]

    def init_assignment(rng):
        return (rng.random(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        return (assignment[var_idx] == sign).sum(axis=1).astype(np.int32)

    def unit_propagate(assignment, num_true):
        changed = False
        for _ in range(3 * n + 10):
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                sat_pos = np.where(assignment[var_idx[c]] == sign[c])[0]
                if sat_pos.size == 0:
                    continue
                p = int(sat_pos[0])
                v = int(var_idx[c, p])
                target = int(sign[c, p])
                if int(assignment[v]) == target:
                    continue
                assignment[v] = target
                changed = True
                progress = True
                c_list = var_clauses_np[v]
                p_list = var_pos_np[v]
                if c_list.size == 0:
                    continue
                s = sign[c_list, p_list]
                now_sat = (assignment[v] == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta
            if not progress:
                break
        return assignment, num_true, changed

    def flip_var(assignment, num_true, v, new_val):
        assignment[v] = new_val
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size > 0:
            s = sign[c_list, p_list]
            now_sat = (new_val == s)
            delta = (2 * now_sat.astype(np.int32)) - 1
            num_true[c_list] += delta

    def compute_delta(assignment, num_true, v):
        """Return change in number of satisfied clauses if we flip v."""
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return 0
        s = sign[c_list, p_list]
        now_sat = (assignment[v] == s)
        # After flip: new_sat = ~now_sat
        # delta_sat = sum(new_sat) - sum(now_sat) = (len - sum(now_sat)) - sum(now_sat) = len - 2*sum(now_sat)
        # But we only care about net change in unsat count:
        # Clauses currently unsat (num_true==0) that have v's literal false become sat: +1 each
        # Clauses currently sat that had v as their only true literal become unsat: -1 each
        unsat_mask = (num_true[c_list] == 0)
        v_lit_false = ~now_sat
        gain = int(np.sum(unsat_mask & v_lit_false))
        # Loss: clauses where num_true==1 and v's literal is true (flipping makes them 0)
        sat_only_v = (num_true[c_list] == 1) & now_sat
        loss = int(np.sum(sat_only_v))
        return gain - loss

    rng = np.random.default_rng(12345)
    best_unsat = m + 1
    best_assign = None

    max_restarts = 50000
    base_steps = 8000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Vary seed per restart for diversity
        rng_r = np.random.default_rng(12345 + restart)
        assignment = init_assignment(rng_r)
        num_true = init_num_true(assignment)

        # Unit propagation
        assignment, num_true, _ = unit_propagate(assignment, num_true)

        unsat_count = int((num_true == 0).sum())
        if unsat_count == 0:
            return assignment.astype(bool).tolist()

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment.copy()

        steps = base_steps + restart * 200
        if steps > 30000:
            steps = 30000

        for step in range(steps):
            if time.time() > deadline:
                break

            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()

            unsat_count = int(unsat_mask.sum())
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            # Pick random unsatisfied clause
            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng_r.integers(unsat_idx.size)])

            # Compute delta for each of the 3 variables in clause c
            best_v = -1
            best_delta = -999
            deltas = []
            for j in range(3):
                v = int(var_idx[c, j])
                d = compute_delta(assignment, num_true, v)
                deltas.append(d)
                if d > best_delta:
                    best_delta = d
                    best_v = v

            # WalkSAT heuristic: flip best with prob based on gain, else random
            # If best_delta > 0, high chance to flip it; if negative, lower chance
            if best_delta > 0:
                flip_best_prob = 0.7 + 0.1 * min(best_delta, 5)
            elif best_delta == 0:
                flip_best_prob = 0.4
            else:
                flip_best_prob = 0.15

            if rng_r.random() < flip_best_prob:
                v = best_v
            else:
                # Random variable from clause
                j = int(rng_r.integers(3))
                v = int(var_idx[c, j])

            # Perform flip
            old_val = int(assignment[v])
            new_val = 1 - old_val
            flip_var(assignment, num_true, v, new_val)

        # After walk, check
        unsat_count = int((num_true == 0).sum())
        if unsat_count == 0:
            return assignment.astype(bool).tolist()
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment.copy()

    if best_assign is not None:
        return best_assign.astype(bool).tolist()
    else:
        rng_f = np.random.default_rng(999)
        return (rng_f.random(n) < 0.5).tolist()
