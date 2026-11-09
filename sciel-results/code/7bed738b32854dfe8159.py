"""
B_3[1] set construction: maximize cardinality with unique unordered triple sums.

Strategy: Fast greedy with incremental triple-sum tracking using a dict
of sum->count for O(1) collision checks. Multiple randomized restarts
with diverse candidate orderings, followed by a gap-filling local improve.
"""
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    N = 4096

    def elapsed():
        return time.time() - t_start

    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            vi = vals[i]
            for j in range(i, m):
                vj = vals[j]
                for k in range(j, m):
                    s = vi + vj + vals[k]
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    def build_sum_dict(vals):
        """Build dict: triple_sum -> count (should be 1 for valid set)."""
        d = {}
        m = len(vals)
        for i in range(m):
            vi = vals[i]
            for j in range(i, m):
                vj = vals[j]
                for k in range(j, m):
                    s = vi + vj + vals[k]
                    d[s] = d.get(s, 0) + 1
        return d

    def try_add(v, chosen, sum_dict):
        """
        Try to add v to chosen. Returns True if successful, updates sum_dict in place.
        """
        # Check all new triple sums involving v
        # (v,v,v), (v,v,u), (v,u,w) for u<=w in chosen
        m = len(chosen)
        # (v,v,v)
        s = 3 * v
        if s in sum_dict:
            return False
        # (v,v,u) and (v,u,u) for each u in chosen
        for i in range(m):
            u = chosen[i]
            s = 2 * v + u
            if s in sum_dict:
                return False
            s = v + 2 * u
            if s in sum_dict:
                return False
        # (v,u,w) for u <= w in chosen
        for i in range(m):
            u = chosen[i]
            for j in range(i, m):
                w = chosen[j]
                s = v + u + w
                if s in sum_dict:
                    return False

        # No collisions: add v and update sum_dict
        sum_dict[3 * v] = 1
        for i in range(m):
            u = chosen[i]
            sum_dict[2 * v + u] = 1
            sum_dict[v + 2 * u] = 1
        for i in range(m):
            u = chosen[i]
            for j in range(i, m):
                w = chosen[j]
                sum_dict[v + u + w] = 1
        chosen.append(v)
        return True

    def greedy_build(candidate_order, limit_time):
        chosen = []
        sum_dict = {}
        chosen_set = set()

        for v in candidate_order:
            if time.time() > limit_time:
                break
            if v in chosen_set:
                continue
            if v < 1 or v > N:
                continue
            if try_add(v, chosen, sum_dict):
                chosen_set.add(v)

        return sorted(chosen)

    def local_improve(vals, limit_time):
        if not vals:
            return vals
        chosen = list(vals)
        chosen_set = set(chosen)
        sum_dict = build_sum_dict(chosen)

        # Try adding each missing value in ascending order
        for v in range(1, N + 1):
            if time.time() > limit_time:
                break
            if v in chosen_set:
                continue
            if try_add(v, chosen, sum_dict):
                chosen_set.add(v)

        return sorted(chosen)

    def local_improve_desc(vals, limit_time):
        if not vals:
            return vals
        chosen = list(vals)
        chosen_set = set(chosen)
        sum_dict = build_sum_dict(chosen)

        for v in range(N, 0, -1):
            if time.time() > limit_time:
                break
            if v in chosen_set:
                continue
            if try_add(v, chosen, sum_dict):
                chosen_set.add(v)

        return sorted(chosen)

    best = []
    best_len = 0

    def update_best(result):
        nonlocal best, best_len
        if result and is_valid(result) and len(result) > best_len:
            best = result
            best_len = len(result)

    # ---------- Strategy 1: Greedy with natural ascending order ----------
    if elapsed() < 3.0:
        result = greedy_build(range(1, N + 1), DEADLINE - 5.0)
        update_best(result)

    # ---------- Strategy 2: Greedy with various structured orderings ----------
    random.seed(42)
    structured_orderings = [
        list(range(1, N + 1, 2)),       # odds
        list(range(2, N + 1, 2)),       # evens
        [x for x in range(1, N + 1) if x % 3 != 0],
        [x for x in range(1, N + 1) if x % 4 in (1, 3)],
        [x for x in range(1, N + 1) if x % 5 in (1, 2, 3, 4)],
        list(range(N, 0, -1)),          # descending
        [x for x in range(1, N + 1) if (x * 7) % 13 < 7],  # pseudo-random via modular
    ]

    for gen in structured_orderings:
        if elapsed() > DEADLINE - 15.0:
            break
        result = greedy_build(gen, DEADLINE - 5.0)
        update_best(result)

    # ---------- Strategy 3: Randomized multi-restart greedy ----------
    # Use varied bases and biases for diverse candidate pools
    num_restarts = 50
    for it in range(num_restarts):
        if elapsed() > DEADLINE - 10.0:
            break
        base = random.randint(2, 12)
        p_extra = random.uniform(0.05, 0.25)
        pool = []
        for x in range(1, N + 1):
            if (x - 1) % base == 0:
                pool.append(x)
            elif random.random() < p_extra:
                pool.append(x)
        random.shuffle(pool)

        result = greedy_build(pool, DEADLINE - 5.0)
        update_best(result)

    # ---------- Strategy 4: More randomized restarts with different structures ----------
    for it in range(30):
        if elapsed() > DEADLINE - 8.0:
            break
        # Random permutation of a subset
        subset_size = random.randint(N // 2, N)
        pool = random.sample(range(1, N + 1), subset_size)
        # Bias toward larger numbers (they tend to have fewer collisions with small triples)
        if random.random() < 0.5:
            pool.sort(key=lambda x: -x)
        else:
            random.shuffle(pool)

        result = greedy_build(pool, DEADLINE - 5.0)
        update_best(result)

    # ---------- Strategy 5: Local improvement passes on best ----------
    if elapsed() < DEADLINE - 4.0 and best:
        improved = local_improve(best, DEADLINE - 2.0)
        update_best(improved)

    if elapsed() < DEADLINE - 3.0 and best:
        improved = local_improve_desc(best, DEADLINE - 2.0)
        update_best(improved)

    # ---------- Strategy 6: Another round of restarts if time permits ----------
    for it in range(20):
        if elapsed() > DEADLINE - 6.0:
            break
        base = random.randint(3, 15)
        pool = [x for x in range(1, N + 1) if (x - 1) % base == 0]
        random.shuffle(pool)
        result = greedy_build(pool, DEADLINE - 4.0)
        update_best(result)

    # ---------- Final local improve ----------
    if elapsed() < DEADLINE - 2.0 and best:
        improved = local_improve(best, DEADLINE - 1.0)
        update_best(improved)

    # ---------- Final sanity check ----------
    best = [v for v in best if 1 <= v <= N]
    best = sorted(set(best))
    if is_valid(best):
        return best

    # Fallback: return a small valid set
    fallback = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    if is_valid(fallback):
        return fallback
    return [1, 2, 4, 8, 16]
