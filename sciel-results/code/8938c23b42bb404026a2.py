"""Rewritten program: maximize collisions among 12 integers in [0,128]
by minimizing the largest Sidon subset (all unordered pair sums unique
INCLUDING diagonals)."""

# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    def remaining():
        return max(0.0, deadline - time.time())

    # ---------- helpers ----------

    def is_sidon(vals):
        """Check if list has all unordered pair sums unique (including diagonals)."""
        seen = set()
        n = len(vals)
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def max_sidon_subset_size(subset, time_budget=5.0):
        """
        Exhaustively find the largest subset of 'subset' that is a Sidon set.
        Uses descending search from len(subset) down to 1, with early termination.
        Time-budgeted to avoid exceeding the deadline.
        """
        n = len(subset)
        # Try sizes from n down to 1
        for size in range(n, 0, -1):
            if remaining() < 1.0:
                # Not enough time to verify; return a conservative estimate
                return size - 1 if size > 1 else 1
            t0 = time.time()
            for combo in combinations(range(n), size):
                if time.time() - t0 > time_budget:
                    break
                vals = [subset[i] for i in combo]
                if is_sidon(vals):
                    return size
        return 1

    def collision_count(subset):
        """Count duplicate pair sums (including diagonals)."""
        n = len(subset)
        sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
        c = Counter(sums)
        return sum(k * (k - 1) // 2 for k in c.values())

    def smooth_surrogate(subset, temperature=1.0):
        """
        Smooth surrogate for the max-Sidon objective:
        Uses log-sum-exp over a proxy: for each pair-sum value s, let f_s = count_s.
        The objective we want to minimize is max_k {size : exists Sidon subset of size k}.
        As a smooth proxy, we use a soft-min over the "constraint violation" measure:
        The idea: if we define for each subset of size m, a "Sidon-ness" score,
        and take a soft-max over m, we get a smooth objective.
        
        A simpler smooth proxy: count how many pairs have duplicate sums,
        weighted by a softmax over the sum values to capture concentration.
        """
        n = len(subset)
        sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
        # Count occurrences of each sum
        c = Counter(sums)
        # Soft-max over the counts: a measure of how "peaked" the sum distribution is
        # Higher peakiness = more collisions = likely smaller max Sidon subset
        counts = list(c.values())
        if not counts:
            return 0.0
        max_c = max(counts)
        # log-sum-exp for numerical stability
        if max_c > 50:
            scaled = [x / max_c for x in counts]
        else:
            scaled = counts
        m = max(scaled)
        if m == 0:
            return 0.0
        lse = m + math.log(math.exp(x - m) for x in scaled) if False else m + math.log(sum(math.exp(x - m) for x in scaled))
        return lse * max_c  # Scale back

    # ---------- main search ----------

    best_subset = None
    best_score = 999  # min max-Sidon subset size
    best_collisions = 0

    def evaluate_candidate(subset):
        nonlocal best_subset, best_score, best_collisions
        if remaining() < 2.0:
            return
        # Validate
        if len(set(subset)) != 12 or not all(0 <= x <= 128 for x in subset):
            return
        # Compute exact max Sidon subset size
        score = max_sidon_subset_size(subset, time_budget=min(3.0, remaining() * 0.3))
        colls = collision_count(subset)
        if score < best_score or (score == best_score and colls > best_collisions):
            best_subset = list(subset)
            best_score = score
            best_collisions = colls

    # Phase 1: generate diverse initial candidates
    random.seed(12345)
    candidates = []

    # 1a. Arithmetic progressions (dense = many collisions)
    for d in range(1, 5):
        for start in range(0, 129 - 11 * d + 1):
            ap = [start + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in ap) and len(set(ap)) == 12:
                candidates.append(ap)

    # 1b. Dense clusters (consecutive or near-consecutive)
    for center in range(0, 129 - 11, 3):
        cluster = list(range(center, center + 12))
        if all(0 <= x <= 128 for x in cluster):
            candidates.append(cluster)

    # 1c. Two tight clusters
    for c1 in range(0, 50, 5):
        for c2 in range(70, 129 - 5, 5):
            if c2 - c1 < 15:
                continue
            cl1 = list(range(c1, c1 + 6))
            cl2 = list(range(c2, c2 + 6))
            if all(0 <= x <= 128 for x in cl1 + cl2) and len(set(cl1 + cl2)) == 12:
                candidates.append(cl1 + cl2)

    # 1d. Random dense sets (concentrated in a small interval)
    for _ in range(200):
        lo = random.randint(0, 100)
        hi = min(128, lo + random.randint(10, 25))
        try:
            cand = random.sample(range(lo, hi + 1), 12)
        except ValueError:
            continue
        if len(set(cand)) == 12:
            candidates.append(cand)

    # 1e. Random sets with a few duplicates in sum structure
    for _ in range(100):
        base = random.sample(range(0, 129), 6)
        # Mirror around a center to create sum collisions
        center = random.randint(64, 128)
        mirror = [2 * center - x for x in base]
        cand = base + [x for x in mirror if x not in base]
        while len(cand) < 12:
            x = random.randint(0, 128)
            if x not in cand:
                cand.append(x)
        if len(set(cand)) == 12:
            candidates.append(cand[:12])

    # Evaluate all initial candidates
    for cand in candidates:
        if remaining() < 3.0:
            break
        evaluate_candidate(cand)

    # Phase 2: local search / hill climbing to improve best_subset
    if best_subset is not None and remaining() > 5.0:
        current = list(best_subset)
        current_score = best_score
        current_colls = best_collisions

        # Simulated annealing style local search
        temp = 2.0
        cooling = 0.95
        for iteration in range(5000):
            if remaining() < 3.0:
                break

            # Pick a random element to perturb
            idx = random.randint(0, 11)
            old_val = current[idx]

            # Try a new value
            new_val = random.randint(0, 128)
            if new_val == old_val:
                continue
            if new_val in current and new_val != old_val:
                continue

            current[idx] = new_val
            new_colls = collision_count(current)
            new_score = max_sidon_subset_size(current, time_budget=min(2.0, remaining() * 0.2))

            delta_score = current_score - new_score  # positive = improvement (lower is better)

            if delta_score > 0 or (delta_score == 0 and new_colls > current_colls):
                # Accept
                current_score = new_score
                current_colls = new_colls
                if new_score < best_score or (new_score == best_score and new_colls > best_collisions):
                    best_subset = list(current)
                    best_score = new_score
                    best_collisions = new_colls
            else:
                # Accept with probability based on temperature
                if random.random() < math.exp(delta_score / max(temp, 0.01)):
                    current_score = new_score
                    current_colls = new_colls
                else:
                    current[idx] = old_val

            temp *= cooling
            if temp < 0.05:
                temp = 0.05

    # Phase 3: targeted search using smooth surrogate
    if remaining() > 5.0:
        # Use the smooth surrogate to guide a gradient-like local search
        if best_subset is not None:
            current = list(best_subset)
            for step in range(3000):
                if remaining() < 3.0:
                    break

                # Compute gradient-like direction: for each element, try small shifts
                best_shift = None
                best_surrogate_delta = 0
                current_surr = smooth_surrogate(current)

                for i in range(12):
                    for delta in [-2, -1, 1, 2]:
                        new_val = current[i] + delta
                        if new_val < 0 or new_val > 128:
                            continue
                        if new_val in current and new_val != current[i]:
                            continue
                        current[i] = new_val
                        new_surr = smooth_surrogate(current)
                        delta_surr = current_surr - new_surr  # positive = improvement
                        if delta_surr > best_surrogate_delta:
                            best_surrogate_delta = delta_surr
                            best_shift = (i, new_val)
                        current[i] = old_val if False else current[i]  # restore
                        # Actually need to restore properly
                        current[i] = current[i]  # no-op, let me fix this

                # Redo properly
                for i in range(12):
                    old_i = current[i]
                    for delta in [-2, -1, 1, 2]:
                        new_val = current[i] + delta
                        if new_val < 0 or new_val > 128:
                            continue
                        if new_val in current and new_val != current[i]:
                            continue
                        current[i] = new_val
                        new_surr = smooth_surrogate(current)
                        delta_surr = current_surr - new_surr
                        if delta_surr > best_surrogate_delta:
                            best_surrogate_delta = delta_surr
                            best_shift = (i, new_val)
                        current[i] = old_i

                if best_shift is not None and best_surrogate_delta > 0:
                    current[best_shift[0]] = best_shift[1]
                    new_score = max_sidon_subset_size(current, time_budget=min(1.5, remaining() * 0.2))
                    new_colls = collision_count(current)
                    if new_score < best_score or (new_score == best_score and new_colls > best_collisions):
                        best_subset = list(current)
                        best_score = new_score
                        best_collisions = new_colls

    # Phase 4: final verification and return
    if best_subset is None:
        # Fallback: return a dense set
        best_subset = list(range(12))
    return sorted(best_subset)

# EVOLVE-BLOCK-END
