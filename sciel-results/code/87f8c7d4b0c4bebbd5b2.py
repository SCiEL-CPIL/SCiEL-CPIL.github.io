# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_masks(clauses, n):
    pos = []
    neg = []
    for c in clauses:
        p = 0
        q = 0
        for l in c:
            if l > 0:
                p |= (1 << (l - 1))
            else:
                q |= (1 << (-l - 1))
        pos.append(p)
        neg.append(q)
    return pos, neg


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0

    # Precompute clause structures
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Build variable occurrence lists: var -> list of (clause_idx, is_pos_in_clause)
    var_occ = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, is_pos in cv:
            var_occ[var].append((ci, is_pos))

    # Build numpy arrays for vectorized satisfaction checking
    pos_arr = np.zeros(m, dtype=np.uint64)
    neg_arr = np.zeros(m, dtype=np.uint64)
    for ci, cv in enumerate(clause_vars):
        p = 0
        q = 0
        for var, is_pos in cv:
            if is_pos:
                p |= (1 << var)
            else:
                q |= (1 << var)
        pos_arr[ci] = p
        neg_arr[ci] = q

    full_mask = (1 << n) - 1

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    def get_unsat_set(v):
        """Return set of unsatisfied clause indices using numpy."""
        assign_np = np.uint64(v)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & full_mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return set(np.where(~satisfied)[0].tolist())

    def update_unsat_after_flip(v, unsat_set, var):
        """After flipping var, update unsat_set incrementally."""
        bit = 1 << var
        for ci, is_pos_in_clause in var_occ[var]:
            if is_pos_in_clause:
                now_satisfied = (v & bit) != 0
            else:
                now_satisfied = (v & bit) == 0

            if now_satisfied:
                unsat_set.discard(ci)
            else:
                clause = clauses[ci]
                other_satisfied = False
                for l in clause:
                    if abs(l) == var + 1:
                        continue
                    if l > 0:
                        if (v & (1 << (l - 1))) != 0:
                            other_satisfied = True
                            break
                    else:
                        if (v & (1 << (-l - 1))) == 0:
                            other_satisfied = True
                            break
                if not other_satisfied:
                    unsat_set.add(ci)
                else:
                    unsat_set.discard(ci)
        return unsat_set

    # Precompute per-variable clause indices and polarities as separate lists for fast access
    var_clauses = [None] * n
    var_polarity_arr = [None] * n
    for var in range(n):
        var_clauses[var] = [ci for ci, _ in var_occ[var]]
        var_polarity_arr[var] = [is_pos for _, is_pos in var_occ[var]]

    # Count clause frequency for weighting
    clause_freq = np.ones(m, dtype=np.float64)
    for var in range(n):
        for ci in var_clauses[var]:
            clause_freq[ci] += 1

    # Normalize clause frequencies
    clause_freq_norm = clause_freq / clause_freq.sum()

    # Precompute for each clause, the list of (var, polarity) as numpy arrays
    clause_var_arrays = []
    clause_pol_arrays = []
    for ci in range(m):
        vars_in = [v for v, _ in clause_vars[ci]]
        pols_in = [p for _, p in clause_vars[ci]]
        clause_var_arrays.append(np.array(vars_in, dtype=np.int32))
        clause_pol_arrays.append(np.array(pols_in, dtype=np.bool_))

    best_assign = None
    best_sat = -1

    # Phase 0: Quick random initializations to find a good starting point
    for attempt in range(32):
        if time.time() > deadline:
            break
        r = random.Random(attempt * 7919)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_set = get_unsat_set(v)
        s = m - len(unsat_set)
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign

    # Phase 1: Multi-restart WalkSAT with improved heuristics
    for restart in range(20000):
        if time.time() > deadline:
            break

        r = random.Random(restart * 104729 + 1)

        # Start from best assignment found so far with perturbation, or random
        if best_assign is not None and r.random() < 0.4:
            v = assign_to_int(best_assign)
            # Perturb: flip 1-3 random variables to escape local minima
            n_flip = 1 + r.randint(0, 2)
            for _ in range(n_flip):
                var_to_flip = r.randrange(n)
                v ^= (1 << var_to_flip)
        else:
            assign = [r.random() < 0.5 for _ in range(n)]
            v = assign_to_int(assign)

        unsat_set = get_unsat_set(v)
        cur_sat = m - len(unsat_set)

        # Adaptive step budget
        base_steps = 4000
        if cur_sat > m * 0.95:
            steps_budget = 12000
        elif cur_sat > m * 0.9:
            steps_budget = 8000
        elif cur_sat > m * 0.8:
            steps_budget = 6000
        else:
            steps_budget = base_steps

        noise_base = 0.25

        # Track progress to detect stagnation
        last_improve_step = 0
        last_improve_sat = cur_sat

        for step in range(steps_budget):
            if time.time() > deadline:
                break

            if len(unsat_set) == 0:
                return int_to_assign(v)

            # Detect stagnation: if no improvement for too many steps, break early
            if step - last_improve_step > steps_budget * 0.4 and cur_sat == last_improve_sat:
                break

            # Select an unsatisfied clause with weighting by frequency
            unsat_list = list(unsat_set)
            if len(unsat_list) > 1:
                # Weighted selection: prefer less frequent clauses (easier to satisfy)
                weights = np.array([1.0 / clause_freq[ci] for ci in unsat_list])
                weights = weights / weights.sum()
                idx = np.random.default_rng(r.random()).choice(len(unsat_list), p=weights)
            else:
                idx = 0
            ci = unsat_list[idx]

            # Variable Selection: Score based on net benefit
            cv = clause_vars[ci]
            best_var = None
            best_score = -1e9

            # Build candidate list
            candidates = []
            for var, is_pos in cv:
                candidates.append((var, is_pos))

            for var, is_pos in candidates:
                # Compute score: how many unsat clauses would be satisfied vs broken
                # Use the precomputed var_clauses and var_polarity_arr
                vc = var_clauses[var]
                vp = var_polarity_arr[var]

                # Count clauses in unsat_set where this variable appears
                # and whether flipping helps or hurts
                score = 0.0
                for i in range(len(vc)):
                    uci = vc[i]
                    if uci in unsat_set:
                        if i == 0 or vc[i-1] != ci:  # skip the selected clause
                            if vp[i] == is_pos:
                                score += 1.0
                            else:
                                score -= 0.5

                # Add noise
                score += r.random() * noise_base
                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var is None:
                best_var = cv[r.randrange(len(cv))][0]

            # Flip the variable
            bit = 1 << best_var
            v ^= bit
            unsat_set = update_unsat_after_flip(v, unsat_set, best_var)
            cur_sat = m - len(unsat_set)

            # Track best
            if cur_sat > best_sat:
                best_sat = cur_sat
                best_assign = int_to_assign(v)
            if cur_sat > last_improve_sat:
                last_improve_sat = cur_sat
                last_improve_step = step

            if best_sat == m:
                return best_assign

        # Update best after restart
        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = int_to_assign(v)

        if best_sat == m:
            return best_assign

    # Phase 2: Final polish with very tight local search
    if best_assign is not None and best_sat < m:
        v = assign_to_int(best_assign)
        unsat_set = get_unsat_set(v)
        polish_deadline = min(time.time() + 25.0, deadline)

        for polish_restart in range(1000):
            if time.time() > polish_deadline:
                break

            v_cur = v
            unsat_cur = set(unsat_set)

            for pstep in range(8000):
                if time.time() > polish_deadline:
                    break

                if len(unsat_cur) == 0:
                    return int_to_assign(v_cur)

                unsat_list = list(unsat_cur)
                ci = unsat_list[r.randrange(len(unsat_list))]
                cv = clause_vars[ci]

                # Greedy with minimal noise
                best_var = None
                best_score = -1e9
                for var, is_pos in cv:
                    vc = var_clauses[var]
                    vp = var_polarity_arr[var]
                    score = 0.0
                    for i in range(len(vc)):
                        if vc[i] in unsat_cur and vc[i] != ci:
                            if vp[i] == is_pos:
                                score += 1.0
                            else:
                                score -= 0.5
                    score += r.random() * 0.03
                    if score > best_score:
                        best_score = score
                        best_var = var

                if best_var is None:
                    best_var = cv[r.randrange(len(cv))][0]

                v_cur ^= (1 << best_var)
                unsat_cur = update_unsat_after_flip(v_cur, unsat_cur, best_var)

            s = m - len(unsat_cur)
            if s > best_sat:
                best_sat = s
                best_assign = int_to_assign(v_cur)
                v = v_cur
                unsat_set = unsat_cur
            if best_sat == m:
                return best_assign

    if best_assign is None:
        return [False] * n
    return best_assign
# EVOLVE-BLOCK-END
