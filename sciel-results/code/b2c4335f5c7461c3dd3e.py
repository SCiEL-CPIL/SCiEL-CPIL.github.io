import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [True] * n

    start_time = time.time()
    deadline = start_time + 320.0   # safety margin before 330s kill

    # ---- Preprocessing: vectorize clause literals ----
    lits = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lits[i, j] = l

    var_idx = np.abs(lits) - 1              # [m,3] 0-based var
    sign = (lits > 0).astype(np.int32)      # [m,3] 1 if positive

    # ---- Variable -> (clauses, positions) lookup, sorted by clause id ----
    var_clauses = [[] for _ in range(n)]
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            var_clauses[v].append(i)
            var_pos[v].append(j)

    for v in range(n):
        if var_clauses[v]:
            order = np.argsort(var_clauses[v], kind="stable")
            var_clauses[v] = [var_clauses[v][k] for k in order]
            var_pos[v] = [var_pos[v][k] for k in order]
    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.empty(0, dtype=np.int32)
                      for vc in var_clauses]
    var_pos_np = [np.array(vp, dtype=np.int32) if vp else np.empty(0, dtype=np.int32)
                  for vp in var_pos]

    # Precompute, per variable, the literal sign of v within each incident clause
    var_signs = [np.array([sign[c, p] for c, p in zip(var_clauses[v], var_pos[v])],
                          dtype=np.int32) if var_clauses[v] else np.empty(0, dtype=np.int32)
                 for v in range(n)]

    def init_assignment():
        return (np.random.rand(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        # satisfied iff assignment[var] == sign
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def check_solved(num_true):
        return num_true.min() > 0

    def update_var(assignment, num_true, v, new_val):
        """Flip/set variable v to new_val and update num_true for incident clauses."""
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return
        s = sign[c_list, p_list]
        now_sat = (new_val == s)
        delta = (2 * now_sat.astype(np.int32)) - 1
        num_true[c_list] += delta

    def unit_propagate(assignment, num_true):
        changed = False
        # Use a worklist approach: repeatedly find unit clauses and propagate
        for _ in range(10 * n + 20):
            # Find all unit clauses (exactly one literal satisfied)
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                # Find which position in this clause is satisfied
                sat_pos = np.where((assignment[var_idx[c]] == sign[c]))[0]
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
                update_var(assignment, num_true, v, target)
            if not progress:
                break
        return assignment, num_true, changed

    def var_unsat_counts(num_true):
        # number of unsatisfied clauses each variable appears in
        unsat = (num_true == 0)
        if not unsat.any():
            return None
        counts = np.zeros(n, dtype=np.int32)
        for c in np.where(unsat)[0]:
            counts[var_idx[c]] += 1
        return counts

    def gain_flip(assignment, num_true, v):
        # number of currently-unsatisfied clauses (containing v) that would be
        # satisfied if we set v to its satisfying value in that clause.
        c_list = var_clauses_np[v]
        if c_list.size == 0:
            return 0
        s = var_signs[v]
        now_sat = (assignment[v] == s)
        unsat_mask = (num_true[c_list] == 0)
        return int(np.sum(unsat_mask & ~now_sat))

    def weighted_pick(c, num_true, var_counts, rng):
        """Pick a variable to flip from unsatisfied clause c, biased toward
        variables that appear in many unsatisfied clauses."""
        cands = var_idx[c]
        if var_counts is not None:
            # Weight by frequency of appearance in unsatisfied clauses
            w = var_counts[cands].astype(np.float64) + 0.5
            # Also consider the gain: how many unsat clauses this var could satisfy
            gains = np.array([gain_flip(assignment_ref[0], num_true, int(v)) for v in cands],
                             dtype=np.float64)
            # Combine: prefer vars with high frequency AND high gain
            w = w * (1.0 + gains)
            total = w.sum()
            r = rng.random() * total
            cum = 0.0
            for k in range(3):
                cum += w[k]
                if r <= cum:
                    return int(cands[k])
            return int(cands[2])
        # fallback: gain-based
        gains = np.array([gain_flip(assignment_ref[0], num_true, int(v)) for v in cands],
                         dtype=np.float64)
        w = gains + 0.5
        total = w.sum()
        r = rng.random() * total
        cum = 0.0
        for k in range(3):
            cum += w[k]
            if r <= cum:
                return int(cands[k])
        return int(cands[2])

    rng = np.random.default_rng(0)
    best_unsat = None
    best_assign = None
    assignment_ref = [None]  # closure holder for weighted_pick fallback

    # Adaptive restart: start with moderate steps, increase if making progress
    base_steps = 2000
    max_restarts = 100000
    current_steps = base_steps
    consecutive_failures = 0

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        assignment = init_assignment()
        assignment_ref[0] = assignment
        num_true = init_num_true(assignment)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        assignment, num_true, _ = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        unsat_before = int((num_true == 0).sum())
        
        for step in range(current_steps):
            if time.time() > deadline - 5.0:
                break

            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()

            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            # Update var_counts more frequently for better guidance
            if step % 4 == 0:
                var_counts = var_unsat_counts(num_true)
            else:
                var_counts = None

            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng.integers(unsat_idx.size)])

            v = weighted_pick(c, num_true, var_counts, rng)

            # decide flip: with some probability use the weighted choice, else a
            # random variable from the clause (local exploration)
            if rng.random() < 0.75:
                v = int(v)
            else:
                p = int(rng.integers(3))
                v = int(var_idx[c, p])

            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

        unsat_after = int((num_true == 0).sum())
        
        # Adaptive restart: if we made progress, keep going; else restart
        if unsat_after < unsat_before:
            consecutive_failures = 0
            current_steps = min(current_steps * 2, 20000)
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                current_steps = base_steps
                consecutive_failures = 0

        if time.time() > deadline - 5.0:
            break

    # ---- Final high-precision polish pass on the best assignment found ----
    if best_assign is not None:
        assignment = best_assign.copy()
        num_true = init_num_true(assignment)
        assignment, num_true, _ = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        polish_deadline = deadline - 1.0
        var_counts = var_unsat_counts(num_true)
        
        # Use a more targeted approach: focus on the most constrained variables
        for _ in range(300000):
            if time.time() > polish_deadline:
                break
            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()
            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()
            
            # Update var_counts every few steps to keep it fresh
            if _ % 2 == 0:
                var_counts = var_unsat_counts(num_true)
            
            # Pick the most "critical" unsatisfied clause (one with vars that appear in many unsat clauses)
            unsat_idx = np.where(unsat_mask)[0]
            if var_counts is not None:
                # Score each unsat clause by sum of var_counts of its variables
                clause_scores = np.array([var_counts[var_idx[c]].sum() for c in unsat_idx])
                # Pick with probability proportional to score (prefer critical clauses)
                total_score = clause_scores.sum()
                if total_score > 0:
                    r = rng.random() * total_score
                    cum = 0.0
                    c = int(unsat_idx[0])
                    for i in range(len(unsat_idx)):
                        cum += clause_scores[i]
                        if r <= cum:
                            c = int(unsat_idx[i])
                            break
                else:
                    c = int(unsat_idx[rng.integers(unsat_idx.size)])
            else:
                c = int(unsat_idx[rng.integers(unsat_idx.size)])
            
            v = weighted_pick(c, num_true, var_counts, rng)
            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

    if best_assign is not None:
        return best_assign.astype(bool).tolist()

    return init_assignment().astype(bool).tolist()
