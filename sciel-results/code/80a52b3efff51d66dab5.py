"""Narrow admissible tuple: fixed-diameter maximization in [0, N]."""
import time
import math
import random

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()

    # Get primes up to N+1 (max possible tuple size is N+1)
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    num_primes = len(all_primes)

    # Precompute residue tables: res_table[pi][v] = v % all_primes[pi]
    # This makes the add-back check O(1) per (candidate, critical prime).
    res_table = []
    for p in all_primes:
        res_table.append([v % p for v in range(N + 1)])

    def greedy_remove(points, seed=None):
        """Greedy removal with randomized tie-breaking. Returns (points, order_used)."""
        rng = random.Random(seed)
        pts = list(points)
        while True:
            if time.time() - start_time > DEADLINE:
                break
            m = len(pts)
            if m == 0:
                break
            changed = False
            # primes to check: p <= m
            # find them by scanning all_primes (small)
            for p in all_primes:
                if p > m:
                    break
                counts = [0] * p
                rt = res_table[all_primes.index(p)] if False else None
                # use direct modulo; m is small so this is fine
                for v in pts:
                    counts[v % p] += 1
                min_c = min(counts)
                if min_c > 0:
                    # collect all min residues
                    min_res = [r for r in range(p) if counts[r] == min_c]
                    target = rng.choice(min_res)
                    pts = [v for v in pts if v % p != target]
                    changed = True
                    break
            if not changed:
                break
        return pts

    def try_add_back(initial_points, candidates):
        """Fast add-back using maintained residue counts and precomputed residues."""
        current = list(initial_points)
        current_set = set(current)
        m = len(current)

        # Build residue count structures for all primes
        res_counts = []
        missing_count = []
        for pi, p in enumerate(all_primes):
            rc = [0] * p
            rt = res_table[pi]
            for v in current:
                rc[rt[v]] += 1
            mc = sum(1 for r in rc if r == 0)
            res_counts.append(rc)
            missing_count.append(mc)

        # critical[pi] = the unique missing residue (if missing_count[pi]==1)
        critical = {}
        for pi, p in enumerate(all_primes):
            if missing_count[pi] == 1:
                rc = res_counts[pi]
                for r in range(p):
                    if rc[r] == 0:
                        critical[pi] = r
                        break

        for v in candidates:
            if v in current_set:
                continue

            m_new = m + 1
            safe = True

            # Check active primes where missing_count == 1
            for pi, miss_r in critical.items():
                p = all_primes[pi]
                if p > m:
                    continue
                if res_table[pi][v] == miss_r:
                    safe = False
                    break

            # Check if m_new is prime (newly active prime)
            if safe and m_new in prime_set:
                p = m_new
                # find pi for this prime
                # m_new <= N+1, and it's prime, so it's in all_primes
                pi = all_primes.index(p)
                rc = res_counts[pi]
                rv = res_table[pi][v]
                # After adding v, all residues covered iff
                #   rc[rv] == 0 (it was the missing one) AND missing_count[pi] == 1
                if rc[rv] == 0 and missing_count[pi] == 1:
                    safe = False

            if safe:
                current.append(v)
                current_set.add(v)
                m = m_new

                # Update residue counts for all primes
                for pi, p in enumerate(all_primes):
                    rv = res_table[pi][v]
                    old_count = res_counts[pi][rv]
                    res_counts[pi][rv] = old_count + 1
                    if old_count == 0:
                        missing_count[pi] -= 1
                    # Update critical
                    if pi in critical:
                        if missing_count[pi] != 1:
                            del critical[pi]
                    if missing_count[pi] == 1:
                        rc = res_counts[pi]
                        for r in range(p):
                            if rc[r] == 0:
                                critical[pi] = r
                                break

        return current

    def try_swap_improve(points, candidates, max_rounds=3):
        """Try to improve by swapping elements out for better candidates."""
        current = list(points)
        current_set = set(current)
        m = len(current)

        res_counts = []
        missing_count = []
        for pi, p in enumerate(all_primes):
            rc = [0] * p
            rt = res_table[pi]
            for v in current:
                rc[rt[v]] += 1
            mc = sum(1 for r in rc if r == 0)
            res_counts.append(rc)
            missing_count.append(mc)

        critical = {}
        for pi, p in enumerate(all_primes):
            if missing_count[pi] == 1:
                rc = res_counts[pi]
                for r in range(p):
                    if rc[r] == 0:
                        critical[pi] = r
                        break

        improved = False
        for _ in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            found_swap = False
            # Try removing each element and see if we can add a better candidate
            # Simple: for each element, temporarily remove, try to add a candidate
            for i in range(len(current)):
                v_out = current[i]
                # Temporarily remove
                for pi, p in enumerate(all_primes):
                    rv = res_table[pi][v_out]
                    res_counts[pi][rv] -= 1
                    if res_counts[pi][rv] == 0:
                        missing_count[pi] += 1
                # Update critical
                for pi in list(critical.keys()):
                    del critical[pi]
                for pi, p in enumerate(all_primes):
                    if missing_count[pi] == 1:
                        rc = res_counts[pi]
                        for r in range(p):
                            if rc[r] == 0:
                                critical[pi] = r
                                break

                m_temp = m - 1
                # Try to add a candidate
                for v_in in candidates:
                    if v_in in current_set:
                        continue
                    safe = True
                    for pi, miss_r in critical.items():
                        p = all_primes[pi]
                        if p > m_temp:
                            continue
                        if res_table[pi][v_in] == miss_r:
                            safe = False
                            break
                    if safe and (m_temp + 1) in prime_set:
                        p = m_temp + 1
                        pi = all_primes.index(p)
                        rv = res_table[pi][v_in]
                        if res_counts[pi][rv] == 0 and missing_count[pi] == 1:
                            safe = False
                    if safe:
                        # Do the swap
                        current[i] = v_in
                        current_set.remove(v_out)
                        current_set.add(v_in)
                        for pi, p in enumerate(all_primes):
                            rv = res_table[pi][v_in]
                            res_counts[pi][rv] += 1
                            if res_counts[pi][rv] == 1:
                                missing_count[pi] -= 1
                        # Update critical
                        for pi in list(critical.keys()):
                            del critical[pi]
                        for pi, p in enumerate(all_primes):
                            if missing_count[pi] == 1:
                                rc = res_counts[pi]
                                for r in range(p):
                                    if rc[r] == 0:
                                        critical[pi] = r
                                        break
                        found_swap = True
                        break

                if found_swap:
                    improved = True
                    break

                # Undo the removal
                for pi, p in enumerate(all_primes):
                    rv = res_table[pi][v_out]
                    res_counts[pi][rv] += 1
                    if res_counts[pi][rv] == 1:
                        missing_count[pi] -= 1
                for pi in list(critical.keys()):
                    del critical[pi]
                for pi, p in enumerate(all_primes):
                    if missing_count[pi] == 1:
                        rc = res_counts[pi]
                        for r in range(p):
                            if rc[r] == 0:
                                critical[pi] = r
                                break

            if not found_swap:
                break

        return current

    # Phase 1: Multiple randomized greedy removals
    best_points = None
    best_len = 0

    # Run several randomized greedy removals
    num_removals = 10
    for seed in range(num_removals):
        if time.time() - start_time > DEADLINE - 60:
            break
        pts = greedy_remove(list(range(N + 1)), seed=seed)
        if len(pts) > best_len:
            best_points = pts
            best_len = len(pts)

    # Phase 2: Fast add-back with multiple random orders
    point_set = set(best_points)
    candidates = [v for v in range(N + 1) if v not in point_set]

    num_trials = 50
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - 30:
            break

        random.shuffle(candidates)

        current = try_add_back(best_points, candidates)

        if len(current) > best_len:
            best_points = current
            best_len = len(current)
            point_set = set(best_points)
            candidates = [v for v in range(N + 1) if v not in point_set]

    # Phase 3: Swap improvement
    if time.time() - start_time < DEADLINE - 10:
        point_set = set(best_points)
        candidates = [v for v in range(N + 1) if v not in point_set]
        improved = try_swap_improve(best_points, candidates)
        if len(improved) > best_len:
            best_points = improved
            best_len = len(improved)

    return best_points
