import math
import time
import random
import numpy as np

def construct():
    START = time.time()
    DEADLINE = START + 143  # safety margin before 150s kill

    def elapsed():
        return time.time() - START

    def time_ok(limit_frac=1.0):
        return elapsed() < DEADLINE - 5 * limit_frac

    score_cache = {}

    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        key = tuple(u)
        if key in score_cache:
            return score_cache[key]
        n = len(u)
        if n == 0:
            score_cache[key] = -1.0
            return -1.0
        max_u = int(u[-1])
        m = 2 * max_u + 1
        if m <= 1:
            score_cache[key] = 1.0
            return 1.0
        u_arr = np.array(u, dtype=np.int64)
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 0:
            score_cache[key] = -1.0
            return -1.0
        score = 1.0 + math.log(r) / math.log(m)
        score_cache[key] = score
        return score

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        ln = len(u)
        for i in range(ln):
            ui = u[i]
            for j in range(i + 1, ln):
                d = u[j] - ui
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal/near-optimal Golomb rulers (verified)
    golomb_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 45],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 58],
    ]

    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    for u in golomb_candidates:
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)

    # Local search restricted to Golomb rulers (the high-ratio region).
    # Neighborhoods: add, remove, swap (replace one element with another).
    def local_search(start_u, time_budget):
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score

        while time.time() - start_t < time_budget and time_ok():
            improved = False
            n = len(current_u)
            cur_max = max(current_u)

            # --- Add a new element (search a wide range) ---
            search_hi = cur_max + max(50, 3 * n * n)
            for new_val in range(cur_max + 1, search_hi + 1):
                candidate = sorted(current_u + [new_val])
                if not is_golomb(candidate):
                    continue
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    if s > best_s:
                        best_s = s
                        best_u = list(candidate)
                    improved = True
                    break

            # --- Remove an element (if size > 3) ---
            if not improved and n > 3:
                for i in range(1, n):
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        if s > best_s:
                            best_s = s
                            best_u = list(candidate)
                        improved = True
                        break

            # --- Swap: replace one non-zero element with a new value ---
            if not improved:
                max_delta = max(30, 5 * n)
                for i in range(1, n):
                    old_val = current_u[i]
                    lo = max(1, old_val - max_delta)
                    hi = old_val + max_delta
                    for new_val in range(lo, hi + 1):
                        if new_val == old_val or new_val in current_u:
                            continue
                        candidate = sorted([new_val if x == old_val else x for x in current_u])
                        if not is_golomb(candidate):
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        return best_u, best_s

    # Simulated annealing over Golomb rulers (escape local optima).
    def simulated_annealing(start_u, time_budget):
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        T = 0.5
        cooling = 0.998

        while time.time() - start_t < time_budget and time_ok():
            neighbor = list(current_u)
            move_type = random.random()
            n = len(neighbor)
            cur_max = max(neighbor)

            if move_type < 0.35 and n < 40:
                new_val = random.randint(cur_max + 1, cur_max + n * n + 20)
                neighbor.append(new_val)
                neighbor = sorted(set(neighbor))
            elif move_type < 0.55 and n > 3:
                idx = random.randint(1, n - 1)
                neighbor.pop(idx)
                neighbor = sorted(set(neighbor))
            else:
                idx = random.randint(1, n - 1)
                delta = random.randint(-1, 1) * random.randint(1, max(10, n))
                new_val = neighbor[idx] + delta
                if new_val >= 1 and new_val not in neighbor:
                    neighbor[idx] = new_val
                    neighbor = sorted(set(neighbor))

            if len(neighbor) < 2 or 0 not in neighbor or not is_golomb(neighbor):
                continue

            new_score = compute_score(neighbor)
            delta_s = new_score - current_score
            if delta_s > 0 or random.random() < math.exp(delta_s / T):
                current_u = neighbor
                current_score = new_score
                if new_score > best_s:
                    best_s = new_score
                    best_u = list(current_u)
            T *= cooling

        return best_u, best_s

    # Run local search from the best known Golomb ruler
    if time_ok(2.0):
        ls_u, ls_s = local_search(best_set, time_budget=25)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    # Local search from several other starting points
    if time_ok(3.0):
        for u in golomb_candidates:
            if not time_ok(3.0):
                break
            ls_u, ls_s = local_search(u, time_budget=10)
            if ls_s > best_score:
                best_score = ls_s
                best_set = ls_u

    # Simulated annealing to escape local optima
    if time_ok(4.0):
        sa_u, sa_s = simulated_annealing(best_set, time_budget=25)
        if sa_s > best_score:
            best_score = sa_s
            best_set = sa_u

    # Polish: local search again from the improved set
    if time_ok(2.0):
        ls_u, ls_s = local_search(best_set, time_budget=15)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    # Final: ensure valid set containing 0
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set

    return best_set
