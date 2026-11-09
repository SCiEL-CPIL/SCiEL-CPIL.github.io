"""Delta-based local search for 3-term progression differences."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def evaluate(S):
        """Count distinct d>0 such that a, a+d, a+2d all in S for some a."""
        S_set = set(S)
        valid_diffs = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)

    def build_diff_structure(S):
        """Build (diffs_set, diff_counts) where diff_counts[d] = number of 3APs with difference d."""
        S_set = set(S)
        n = len(S)
        diff_counts = {}
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        return diff_counts

    def local_search_delta(S, max_iters=100000):
        """Delta-based hill climbing: incrementally update diff counts on element swap."""
        S = S[:]
        S_set = set(S)
        diff_counts = build_diff_structure(S)
        current_score = len(diff_counts)

        def remove_element(x):
            """Remove x from S_set and update diff_counts. Returns list of (d, delta) changes."""
            changes = []
            # For each other element y in S_set:
            # Check all 3APs involving x
            for y in S_set:
                if y == x:
                    continue
                # x as first term: x, y, 2y-x
                d1 = y - x
                c1 = 2 * y - x
                if c1 in S_set and c1 != x and c1 != y:
                    if d1 in diff_counts:
                        diff_counts[d1] -= 1
                        changes.append((d1, -1))
                        if diff_counts[d1] <= 0:
                            del diff_counts[d1]
                # x as second term: y, x, 2x-y (if y < x) or 2x-y, x, y
                # Case: y < x, so y, x, 2x-y
                if y < x:
                    d2 = x - y
                    c2 = 2 * x - y
                    if c2 in S_set and c2 != x and c2 != y:
                        if d2 in diff_counts:
                            diff_counts[d2] -= 1
                            changes.append((d2, -1))
                            if diff_counts[d2] <= 0:
                                del diff_counts[d2]
                # Case: y > x, so 2x-y < x < y, i.e., 2x-y, x, y
                else:
                    d3 = y - x
                    c3 = 2 * x - y
                    if c3 in S_set and c3 != x and c3 != y:
                        if d3 in diff_counts:
                            diff_counts[d3] -= 1
                            changes.append((d3, -1))
                            if diff_counts[d3] <= 0:
                                del diff_counts[d3]
            S_set.remove(x)
            return changes

        def add_element(x):
            """Add x to S_set and update diff_counts. Returns list of (d, delta) changes."""
            changes = []
            for y in S_set:
                if y == x:
                    continue
                # x as first term: x, y, 2y-x
                d1 = y - x
                c1 = 2 * y - x
                if c1 in S_set and c1 != x and c1 != y:
                    diff_counts[d1] = diff_counts.get(d1, 0) + 1
                    changes.append((d1, 1))
                # x as second term
                if y < x:
                    # y, x, 2x-y
                    d2 = x - y
                    c2 = 2 * x - y
                    if c2 in S_set and c2 != x and c2 != y:
                        diff_counts[d2] = diff_counts.get(d2, 0) + 1
                        changes.append((d2, 1))
                else:
                    # 2x-y, x, y
                    d3 = y - x
                    c3 = 2 * x - y
                    if c3 in S_set and c3 != x and c3 != y:
                        diff_counts[d3] = diff_counts.get(d3, 0) + 1
                        changes.append((d3, 1))
            S_set.add(x)
            return changes

        for it in range(max_iters):
            if time.time() >= deadline:
                break
            idx = random.randint(0, 31)
            old_val = S[idx]

            new_val = random.randint(1, 512)
            if new_val in S_set:
                continue

            # Remove old, add new
            remove_element(old_val)
            score_after_remove = len(diff_counts)
            add_element(new_val)
            new_score = len(diff_counts)

            if new_score >= current_score:
                S[idx] = new_val
                current_score = new_score
            else:
                # Revert
                remove_element(new_val)
                add_element(old_val)
                S[idx] = old_val

        return S, current_score

    # Generate diverse initial candidates
    candidates = []

    # 1. Consecutive integers
    candidates.append(list(range(1, 33)))

    # 2. Arithmetic progressions with various steps
    for step in [2, 3, 4, 5, 7, 10, 15, 20]:
        start = 1
        end = start + 31 * step
        if end <= 512:
            candidates.append([start + i * step for i in range(32)])

    # 3. Random sets
    random.seed(42)
    for _ in range(20):
        candidates.append(random.sample(range(1, 513), 32))

    # 4. Structured: mix of small and large values
    for _ in range(10):
        small = random.sample(range(1, 100), 16)
        large = random.sample(range(400, 513), 16)
        candidates.append(small + large)

    # 5. Quadratic-like sets
    for a in [1, 2, 3, 5]:
        vals = [a * i * i + 1 for i in range(32)]
        if max(vals) <= 512:
            candidates.append(vals)

    # 6. Grid-based construction (4x8)
    M = 64
    S = set()
    for i in range(4):
        for j in range(8):
            val = i * M + j + 1
            if 1 <= val <= 512:
                S.add(val)
    candidates.append(list(S))

    # Evaluate all candidates
    best_S = None
    best_score = 0
    for cand in candidates:
        if len(set(cand)) != 32:
            continue
        score = evaluate(cand)
        if score > best_score:
            best_score = score
            best_S = cand[:]

    # Multi-start local search with delta evaluation
    num_starts = 20
    for s in range(num_starts):
        if time.time() >= deadline:
            break

        # Create a perturbed version of best_S
        if s == 0:
            start_S = best_S[:]
        else:
            start_S = best_S[:]
            # Perturb 5-15 random elements
            num_perturb = random.randint(5, 15)
            indices = random.sample(range(32), num_perturb)
            S_set = set(start_S)
            for idx in indices:
                new_val = random.randint(1, 512)
                while new_val in S_set:
                    new_val = random.randint(1, 512)
                S_set.remove(start_S[idx])
                S_set.add(new_val)
                start_S[idx] = new_val

        # Run delta-based local search
        result_S, result_score = local_search_delta(start_S, max_iters=50000)

        if result_score > best_score:
            best_score = result_score
            best_S = result_S[:]

    return best_S
