import time
import numpy as np
import random

D = 11
DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _build_candidate_pool():
    """Build a diverse pool of candidate integer vectors in R^11."""
    pool = []
    pool_set = set()

    def add(v):
        t = tuple(v)
        if t not in pool_set:
            pool_set.add(t)
            pool.append(v)

    # 1-sparse vectors
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            add(v)

    # 2-sparse vectors
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    add(v)

    # 3-sparse vectors (all sign combos)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            add(v)

    # 4-sparse vectors (subset of sign combos for diversity)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = [0] * D
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    add(v)

    # Random vectors with entries in {-2,-1,0,1,2}
    for _ in range(3000):
        v = [random.randint(-2, 2) for _ in range(D)]
        if any(x != 0 for x in v):
            add(v)

    # Random vectors with entries in {-3,-2,-1,1,2,3}
    for _ in range(2000):
        v = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(D)]
        add(v)

    return pool


def _norm_sq(v):
    return sum(x * x for x in v)


def _dist_sq(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(D))


def _greedy_construct(pool):
    """Greedy: add vectors that satisfy the constraint w.r.t. all selected."""
    selected = []
    max_norm_sq = 0
    for v in pool:
        ns = _norm_sq(v)
        if ns <= 0:
            continue
        # Check against all selected: dist^2 >= max(selected norms^2, ns)
        ok = True
        for u in selected:
            d2 = _dist_sq(v, u)
            u_ns = _norm_sq(u)
            if d2 < max(u_ns, ns):
                ok = False
                break
        if ok:
            selected.append(v)
            if ns > max_norm_sq:
                max_norm_sq = ns
    return selected


def _check_valid(selected):
    """Verify all pairwise constraints hold."""
    n = len(selected)
    for i in range(n):
        ni = _norm_sq(selected[i])
        for j in range(i + 1, n):
            nj = _norm_sq(selected[j])
            d2 = _dist_sq(selected[i], selected[j])
            if d2 < max(ni, nj):
                return False
    return True


def _sa_optimize(initial, pool, pool_index, time_budget):
    """Simulated annealing with conflict-count array."""
    selected = list(initial)
    n = len(selected)
    # Precompute norms squared
    norms = [_norm_sq(v) for v in selected]
    max_ns = max(norms) if norms else 0

    # Conflict count: for each selected vector, how many others violate
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist_sq(selected[i], selected[j])
            if d2 < max(norms[i], norms[j]):
                conflicts[i] += 1
                conflicts[j] += 1

    total_conflicts = sum(conflicts) // 2

    # Map from vector tuple to index in selected
    v_to_idx = {}
    for idx, v in enumerate(selected):
        v_to_idx[tuple(v)] = idx

    # Pool index: set of pool vectors NOT in selected
    in_selected = set(v_to_idx.keys())

    best_selected = list(selected)
    best_conflicts = total_conflicts
    best_score = n - total_conflicts  # score proxy: size minus conflicts

    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    rng = random.Random(42)

    steps = 0
    start = time.time()
    while time.time() - start < time_budget and T > T_min:
        steps += 1
        # Choose move type: add, remove, or swap
        move_type = rng.random()
        if move_type < 0.4:
            # Add a random pool vector not already selected
            candidates = [v for v in pool if tuple(v) not in in_selected]
            if not candidates:
                continue
            v = rng.choice(candidates)
            vkey = tuple(v)
            # Compute conflicts this vector would add
            new_ns = _norm_sq(v)
            new_max = max(max_ns, new_ns)
            added_conflicts = 0
            for i in range(n):
                d2 = _dist_sq(v, selected[i])
                if d2 < max(new_max, norms[i]):
                    added_conflicts += 1
            delta_conf = added_conflicts
            # Also, adding v changes max_ns which may affect existing conflicts
            # This is approximate; we'll recompute after
            if rng.random() < np.exp(-delta_conf / T) or delta_conf < 0:
                # Accept: add v
                idx = n
                selected.append(v)
                norms.append(new_ns)
                v_to_idx[vkey] = idx
                in_selected.add(vkey)
                n += 1
                # Recompute max_ns
                if new_ns > max_ns:
                    max_ns = new_ns
                # Update conflicts for all pairs involving new element
                for i in range(n - 1):
                    d2 = _dist_sq(v, selected[i])
                    if d2 < max(max_ns, norms[i]):
                        conflicts[i] += 1
                        conflicts[idx] += 1
                    else:
                        pass
                # Handle max_ns change: if new_ns > old max, some previously ok pairs may now conflict
                if new_ns > max_ns:
                    for i in range(n - 1):
                        for j in range(i + 1, n - 1):
                            d2 = _dist_sq(selected[i], selected[j])
                            if d2 < new_ns and d2 >= max(norms[i], norms[j]):
                                # Was ok before, now may not be
                                if d2 < max(norms[i], norms[j]):
                                    pass  # already counted
                                else:
                                    # New conflict due to raised max
                                    # Actually constraint is d2 >= max(norm_i, norm_j), not max_ns
                                    # Wait, the constraint is d2 >= max(||u||^2, ||v||^2) for each pair
                                    # So max_ns doesn't directly affect pair constraints
                                    # Each pair uses its own two norms
                                    pass
                total_conflicts = sum(conflicts) // 2
                if total_conflicts < best_conflicts:
                    best_conflicts = total_conflicts
                    best_selected = list(selected)
                    best_score = n - total_conflicts

        elif move_type < 0.7:
            # Remove a random selected vector (prefer high-conflict)
            if n <= 1:
                continue
            # Weighted selection by conflicts
            weights = [max(1, conflicts[i]) for i in range(n)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            remove_idx = 0
            for i in range(n):
                cum += weights[i]
                if r <= cum:
                    remove_idx = i
                    break
            v = selected[remove_idx]
            vkey = tuple(v)
            removed_ns = norms[remove_idx]
            # Update conflicts
            for i in range(n):
                if i == remove_idx:
                    continue
                d2 = _dist_sq(v, selected[i])
                if d2 < max(removed_ns, norms[i]):
                    conflicts[i] -= 1
                    conflicts[remove_idx] -= 1
            # Remove from lists
            selected.pop(remove_idx)
            norms.pop(remove_idx)
            del v_to_idx[vkey]
            in_selected.discard(vkey)
            conflicts.pop(remove_idx)
            n -= 1
            if n > 0:
                max_ns = max(norms)
            else:
                max_ns = 0
            total_conflicts = sum(conflicts) // 2
            if total_conflicts < best_conflicts:
                best_conflicts = total_conflicts
                best_selected = list(selected)
                best_score = n - total_conflicts

        else:
            # Swap: remove one, add another
            if n <= 1:
                continue
            weights = [max(1, conflicts[i]) for i in range(n)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            remove_idx = 0
            for i in range(n):
                cum += weights[i]
                if r <= cum:
                    remove_idx = i
                    break
            v_rem = selected[remove_idx]
            vkey_rem = tuple(v_rem)
            ns_rem = norms[remove_idx]

            candidates = [v for v in pool if tuple(v) not in in_selected]
            if not candidates:
                continue
            v_add = rng.choice(candidates)
            vkey_add = tuple(v_add)
            ns_add = _norm_sq(v_add)

            # Compute delta conflicts approximately
            # Remove v_rem: decreases conflicts with others
            rem_conf_decrease = 0
            for i in range(n):
                if i == remove_idx:
                    continue
                d2 = _dist_sq(v_rem, selected[i])
                if d2 < max(ns_rem, norms[i]):
                    rem_conf_decrease += 1

            # Add v_add: increases conflicts with others (excluding v_rem)
            add_conf_increase = 0
            for i in range(n):
                if i == remove_idx:
                    continue
                d2 = _dist_sq(v_add, selected[i])
                if d2 < max(ns_add, norms[i]):
                    add_conf_increase += 1

            delta_conf = add_conf_increase - rem_conf_decrease
            if rng.random() < np.exp(-delta_conf / T) or delta_conf < 0:
                # Execute swap
                # Remove v_rem
                for i in range(n):
                    if i == remove_idx:
                        continue
                    d2 = _dist_sq(v_rem, selected[i])
                    if d2 < max(ns_rem, norms[i]):
                        conflicts[i] -= 1
                        conflicts[remove_idx] -= 1
                # Add v_add at position remove_idx
                selected[remove_idx] = v_add
                norms[remove_idx] = ns_add
                v_to_idx.pop(vkey_rem)
                in_selected.discard(vkey_rem)
                v_to_idx[vkey_add] = remove_idx
                in_selected.add(vkey_add)
                # Update conflicts for new element
                for i in range(n):
                    if i == remove_idx:
                        continue
                    d2 = _dist_sq(v_add, selected[i])
                    if d2 < max(ns_add, norms[i]):
                        conflicts[i] += 1
                        conflicts[remove_idx] += 1
                if n > 0:
                    max_ns = max(norms)
                total_conflicts = sum(conflicts) // 2
                if total_conflicts < best_conflicts:
                    best_conflicts = total_conflicts
                    best_selected = list(selected)
                    best_score = n - total_conflicts

        T *= cooling

    return best_selected


def construct():
    start_time = time.time()
    pool = _build_candidate_pool()
    pool_len = len(pool)

    # Greedy initial solution
    selected = _greedy_construct(pool)
    best = list(selected)
    best_len = len(selected)

    # Multiple SA restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > DEADLINE - 5:
            break
        # Perturb the greedy solution
        rng = random.Random(1000 + r)
        init = list(selected)
        # Randomly remove some and add others
        for _ in range(min(10, len(init) // 3)):
            if len(init) > 1:
                idx = rng.randrange(len(init))
                init.pop(idx)
        # Add random pool vectors
        for _ in range(min(5, len(pool) // 10)):
            v = rng.choice(pool)
            if tuple(v) not in [tuple(x) for x in init]:
                init.append(v)

        # SA optimize
        budget = min(5.0, max(0.5, (DEADLINE - 5 - time.time()) / max(1, n_restarts - r)))
        result = _sa_optimize(init, pool, None, budget)

        if len(result) > best_len:
            best = result
            best_len = len(result)

    # Final polish: try to add any pool vector that fits
    if time.time() < DEADLINE - 2:
        current = list(best)
        norms = [_norm_sq(v) for v in current]
        changed = True
        while changed and time.time() < DEADLINE - 2:
            changed = False
            for v in pool:
                if time.time() > DEADLINE - 2:
                    break
                if tuple(v) in [tuple(x) for x in current]:
                    continue
                ns = _norm_sq(v)
                ok = True
                for u in current:
                    d2 = _dist_sq(v, u)
                    if d2 < max(_norm_sq(u), ns):
                        ok = False
                        break
                if ok:
                    current.append(v)
                    changed = True

        if len(current) > len(best):
            best = current

    # Verify
    if not _check_valid(best):
        # Fallback to greedy
        best = _greedy_construct(pool)

    return np.array(best, dtype=np.int64)
