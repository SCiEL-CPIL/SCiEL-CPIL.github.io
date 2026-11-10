import numpy as np
import sys
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    time_limit = 600  # seconds, leave 30s margin for 630s total

    # Forbidden differences mod 205
    forbidden_205 = set()
    for x in range(1, m):
        forbidden_205.add(pow(x, 2, m))
    forbidden_205.discard(0)

    # Nonzero squares mod 41
    squares_41 = set()
    for x in range(1, 41):
        squares_41.add(pow(x, 2, 41))
    squares_41.discard(0)

    # Build adjacency for Paley(41)
    n = 41
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = (i - j) % n
            if d in squares_41:
                adj[i].add(j)
                adj[j].add(i)

    # Greedy with different orderings
    def greedy_order(order):
        s = set()
        for v in order:
            can_add = True
            for u in s:
                if (v - u) % n in squares_41:
                    can_add = False
                    break
            if can_add:
                s.add(v)
        return s

    # Multiple greedy with different orderings
    best_set = set()
    best_size = 0

    for seed in range(200):
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        s = greedy_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    print(f"Best greedy size: {best_size}, set: {sorted(best_set)}", file=sys.stderr)

    # Improved simulated annealing with multiple restarts
    def sa_run(initial_set, time_budget, rng_seed):
        rng = random.Random(rng_seed)
        current_set = set(initial_set)
        current_size = len(current_set)
        best_local = set(current_set)
        best_local_size = current_size

        T_start = 2.0
        T_end = 0.005
        max_iters = 500000
        t0 = time.time()

        for it in range(max_iters):
            if time.time() - t0 > time_budget:
                break

            T = T_start * (T_end / T_start) ** (it / max_iters)

            # Choose move type
            r = rng.random()
            if r < 0.25:
                # Try to add a vertex
                candidates = [v for v in range(n) if v not in current_set]
                if not candidates:
                    continue
                v = rng.choice(candidates)
                conflicts = [u for u in current_set if (v - u) % n in squares_41]
                if not conflicts:
                    new_set = set(current_set)
                    new_set.add(v)
                    delta = 1
                else:
                    continue
            elif r < 0.40:
                # Try to remove a vertex
                if not current_set:
                    continue
                v = rng.choice(list(current_set))
                new_set = set(current_set)
                new_set.discard(v)
                delta = -1
            elif r < 0.70:
                # Swap: remove one, add another
                if not current_set:
                    continue
                v_remove = rng.choice(list(current_set))
                candidates = [v for v in range(n) if v not in current_set]
                if not candidates:
                    continue
                v_add = rng.choice(candidates)
                temp_set = set(current_set)
                temp_set.discard(v_remove)
                conflicts = [u for u in temp_set if (v_add - u) % n in squares_41]
                if not conflicts:
                    new_set = set(temp_set)
                    new_set.add(v_add)
                    delta = 0
                else:
                    continue
            else:
                # Double swap: remove two, add two
                if len(current_set) < 2:
                    continue
                v1, v2 = rng.sample(list(current_set), 2)
                candidates = [v for v in range(n) if v not in current_set]
                if len(candidates) < 2:
                    continue
                w1, w2 = rng.sample(candidates, 2)
                temp_set = set(current_set)
                temp_set.discard(v1)
                temp_set.discard(v2)
                ok = True
                for w in (w1, w2):
                    for u in temp_set:
                        if (w - u) % n in squares_41:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and (w1 - w2) % n not in squares_41:
                    new_set = set(temp_set)
                    new_set.add(w1)
                    new_set.add(w2)
                    delta = 0
                else:
                    continue

            # Accept/reject
            if delta > 0:
                current_set = new_set
                current_size += delta
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local = set(current_set)
            elif delta < 0:
                # Always accept worsening moves at high T, reject at low T
                if rng.random() < min(1.0, T / abs(delta)):
                    current_set = new_set
                    current_size += delta
            else:
                # delta == 0, accept with probability based on T
                if rng.random() < 0.3 + 0.7 * T / T_start:
                    current_set = new_set

        return best_local, best_local_size

    # Run multiple SA restarts from different starting points
    best_overall = set(best_set)
    best_overall_size = best_size

    # Restart from best greedy
    t0 = time.time()
    remaining = time_limit - 30 - (t0 - start_time)
    if remaining > 2:
        s, sz = sa_run(best_set, remaining * 0.4, 42)
        if sz > best_overall_size:
            best_overall_size = sz
            best_overall = set(s)

    # Additional restarts from perturbed versions
    restart_count = 0
    t0 = time.time()
    while time.time() - start_time < time_limit - 30 and restart_count < 8:
        # Perturb best set
        rng = random.Random(1000 + restart_count)
        perturbed = set(best_overall)
        for _ in range(rng.randint(1, 3)):
            if perturbed:
                v = rng.choice(list(perturbed))
                perturbed.discard(v)
                candidates = [x for x in range(n) if x not in perturbed]
                if candidates:
                    perturbed.add(rng.choice(candidates))

        time_left = time_limit - 30 - (time.time() - start_time)
        if time_left < 1:
            break

        s, sz = sa_run(perturbed, time_left * 0.3, 2000 + restart_count)
        if sz > best_overall_size:
            best_overall_size = sz
            best_overall = set(s)
        restart_count += 1

    print(f"Best overall size: {best_overall_size}, set: {sorted(best_overall)}", file=sys.stderr)

    # Construct mod 205 answer using CRT
    # Residues 0 and 2 mod 5 (since 2 is a non-square mod 5)
    result = set()
    for k in best_overall:
        result.add(5 * k % 205)
        result.add((2 + 5 * k) % 205)

    result_arr = np.array(sorted(result), dtype=np.int64)

    # Verify
    valid = True
    for i in range(len(result_arr)):
        for j in range(i+1, len(result_arr)):
            d1 = (result_arr[i] - result_arr[j]) % 205
            d2 = (result_arr[j] - result_arr[i]) % 205
            if d1 in forbidden_205 or d2 in forbidden_205:
                valid = False
                break
        if not valid:
            break

    if not valid:
        print("VALIDATION FAILED, using trivial set", file=sys.stderr)
        return np.array([0], dtype=np.int64)

    print(f"Final result size: {len(result_arr)}", file=sys.stderr)
    return result_arr
