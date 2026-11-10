import time
import random

def construct():
    deadline = time.time() + 85.0
    N = 256
    MAX_SIZE = 60

    def compute_sums(vals):
        """Compute bitset DP per cardinality. Returns list of bitsets."""
        n = len(vals)
        sums = [0] * (n + 1)
        sums[0] = 1
        for v in vals:
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        return sums

    def check_valid(vals):
        """Check cardinality-separated subset sums using bitset DP per cardinality."""
        if not vals:
            return True
        sums = compute_sums(vals)
        seen = 0
        for i in range(len(vals) + 1):
            if sums[i] & seen:
                return False
            seen |= sums[i]
        return True

    def count_conflicts(vals):
        """Count total cross-cardinality sum collisions. Returns (total_conflicts, conflict_map).
        conflict_map[v] = number of collisions involving v (approximate)."""
        sums = compute_sums(vals)
        n = len(vals)
        total = 0
        seen = 0
        for i in range(n + 1):
            overlap = sums[i] & seen
            # Count bits in overlap
            total += bin(overlap).count('1')
            seen |= sums[i]
        
        # Compute per-element conflict contribution via leave-one-out
        conflict_map = {}
        for idx, v in enumerate(vals):
            without_v = vals[:idx] + vals[idx+1:]
            sums_wo = compute_sums(without_v)
            n_wo = len(without_v)
            total_wo = 0
            seen_wo = 0
            for i in range(n_wo + 1):
                overlap = sums_wo[i] & seen_wo
                total_wo += bin(overlap).count('1')
                seen_wo |= sums_wo[i]
            conflict_map[v] = total - total_wo
        return total, conflict_map

    def try_greedy(pool, max_size, reverse=True):
        result = []
        for v in sorted(pool, reverse=reverse):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_grow(initial, all_vals):
        cur = list(initial)
        cur_set = set(cur)
        for v in sorted(all_vals):
            if v in cur_set:
                continue
            if len(cur) >= MAX_SIZE:
                break
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
        return cur

    def conflict_resolve(initial_set, pool, max_size, time_limit):
        """Start with a large set, iteratively remove elements involved in most conflicts."""
        cur = list(initial_set)
        cur_set = set(cur)
        while time.time() < time_limit:
            if not cur:
                break
            total, conflict_map = count_conflicts(cur)
            if total == 0:
                # Valid set, try to grow
                for v in sorted(pool):
                    if v in cur_set:
                        continue
                    if len(cur) >= max_size:
                        break
                    cand = cur + [v]
                    if check_valid(cand):
                        cur = cand
                        cur_set.add(v)
                # If still can grow, try more
                improved = True
                while improved and time.time() < time_limit:
                    improved = False
                    for v in sorted(pool):
                        if v in cur_set:
                            continue
                        if len(cur) >= max_size:
                            break
                        cand = cur + [v]
                        if check_valid(cand):
                            cur = cand
                            cur_set.add(v)
                            improved = True
                break
            
            # Find element with highest conflict contribution
            max_conf = -1
            max_v = None
            for v, c in conflict_map.items():
                if c > max_conf:
                    max_conf = c
                    max_v = v
            
            if max_v is not None and max_conf > 0:
                cur.remove(max_v)
                cur_set.discard(max_v)
            else:
                break
        
        return cur

    def conflict_local_search(cur, pool, max_size, time_limit):
        """Local search guided by conflict count reduction."""
        cur = list(cur)
        cur_set = set(cur)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding elements
            for v in sorted(pool):
                if v in cur_set:
                    continue
                if len(cur) >= max_size:
                    break
                cand = cur + [v]
                if check_valid(cand):
                    cur = cand
                    cur_set.add(v)
                    improved = True
            
            # Try swaps: remove one, add one, guided by conflict reduction
            if len(cur) < max_size and not improved:
                total_cur, _ = count_conflicts(cur) if len(cur) > 0 else (0, {})
                best_swap = None
                best_new_total = total_cur + 1  # Must improve
                for v_out in sorted(cur, key=lambda x: x, reverse=True):
                    for v_in in sorted(pool):
                        if v_in in cur_set:
                            continue
                        if v_in == v_out:
                            continue
                        cand = [x for x in cur if x != v_out] + [v_in]
                        if check_valid(cand):
                            # Valid swap is always good
                            cur = cand
                            cur_set = set(cur)
                            improved = True
                            break
                        else:
                            # Check if this swap reduces conflicts
                            total_new, _ = count_conflicts(cand)
                            if total_new < best_new_total:
                                best_new_total = total_new
                                best_swap = (v_out, v_in)
                    if improved:
                        break
                if not improved and best_swap is not None:
                    v_out, v_in = best_swap
                    cur = [x for x in cur if x != v_out] + [v_in]
                    cur_set = set(cur)
                    improved = True
        
        return cur

    pool = list(range(1, N + 1))
    best = []

    # Baseline: superincreasing
    sup = [1]
    for i in range(1, 60):
        nxt = sum(sup) + 1
        if nxt <= N:
            sup.append(nxt)
        else:
            break
    if check_valid(sup):
        best = list(sup)

    # Greedy descending
    desc = try_greedy(pool, MAX_SIZE, reverse=True)
    if len(desc) > len(best):
        best = desc

    if time.time() > deadline:
        return [int(x) for x in best]

    # Greedy ascending
    asc = try_greedy(pool, MAX_SIZE, reverse=False)
    if len(asc) > len(best):
        best = asc

    # Grow from superincreasing
    grown = try_grow(sup, pool)
    if len(grown) > len(best):
        best = grown

    # Conflict-based resolution: start with large random sets and remove conflicts
    remaining = deadline - time.time()
    if remaining > 10:
        rng = random.Random(123)
        iters = min(20, int(remaining * 2))
        for _ in range(iters):
            if time.time() > deadline - 5:
                break
            # Start with a random subset of size 40-60
            start_size = min(MAX_SIZE, 40 + rng.randint(0, 20))
            initial = rng.sample(pool, start_size)
            result = conflict_resolve(initial, pool, MAX_SIZE, time_limit=deadline - 5)
            if len(result) > len(best):
                best = result

    # Local search with conflict guidance
    remaining = deadline - time.time()
    if remaining > 5:
        ls_time = min(20.0, remaining * 0.5)
        ls_result = conflict_local_search(best, pool, MAX_SIZE, time_limit=ls_time)
        if len(ls_result) > len(best):
            best = ls_result

    # Randomized restarts
    remaining = deadline - time.time()
    if remaining > 3:
        rng = random.Random(42)
        iters = min(30, int(remaining * 8))
        for _ in range(iters):
            if time.time() > deadline - 1:
                break
            shuffled = pool[:]
            rng.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= MAX_SIZE:
                    break
                candidate = result + [v]
                if check_valid(candidate):
                    result = candidate
            if len(result) > len(best):
                best = result
            # Try to grow
            grown = try_grow(result, pool)
            if len(grown) > len(best):
                best = grown

    return [int(x) for x in best]
