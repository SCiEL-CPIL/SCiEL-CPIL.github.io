import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total = q ** n

    if total > 5_000_000:
        if q == 3 and n >= 4 and radius == 1:
            _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in _TETRA
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1(deadline)

    return _generic_greedy(q, n, radius, deadline)


def _solve_3_8_1(deadline):
    q = 3
    n = 8
    radius = 1
    total_points = q ** n
    ball_size = 1 + n * (q - 1)

    # Pre-compute neighbors using base-3 arithmetic
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    # Precompute powers of 3
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    for idx in range(total_points):
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q

        neighbors[idx, 0] = idx
        k = 1
        for pos in range(n):
            orig = word[pos]
            for delta in [1, 2]:
                new_digit = (orig + delta) % q
                new_idx = idx + (new_digit - orig) * powers[pos]
                neighbors[idx, k] = new_idx
                k += 1

    def prune(selected):
        if not selected:
            return []
        selected = list(selected)
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            coverage[neighbors[idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                if np.min(coverage[neighbors[idx]]) > 1:
                    selected.pop(i)
                    coverage[neighbors[idx]] -= 1
                    changed = True
        return selected

    def greedy_init(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            if time.time() > deadline - 10:
                break

            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break

            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]

            selected.append(int(best_idx))
            is_covered[neighbors[best_idx]] = True

        return selected

    def local_search(initial_code, seed):
        rng = np.random.default_rng(seed)
        code = prune(list(initial_code))
        best_code = code[:]
        best_len = len(code)

        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1

        code_set = set(code)

        # Kick-and-prune loop
        while time.time() < deadline - 2:
            improved = False

            # Try to prune
            pruned = prune(code)
            if len(pruned) < len(code):
                code = pruned
                code_set = set(code)
                coverage = np.zeros(total_points, dtype=np.int32)
                for idx in code:
                    coverage[neighbors[idx]] += 1
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                improved = True
                continue

            if not code:
                break

            # Pick a random codeword to move
            c_idx = rng.integers(len(code))
            c = code[c_idx]

            # Pick a target: neighbor or random
            if rng.random() < 0.8:
                n_idx = rng.integers(1, ball_size)
                c_prime = int(neighbors[c, n_idx])
            else:
                c_prime = int(rng.integers(total_points))

            if c_prime in code_set:
                continue

            # Check swap validity
            new_coverage = coverage.copy()
            new_coverage[neighbors[c]] -= 1
            new_coverage[neighbors[c_prime]] += 1

            if np.min(new_coverage) >= 0:
                # Valid swap
                code[c_idx] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                coverage = new_coverage
                improved = True

            if not improved:
                # Try a random kick
                if rng.random() < 0.3:
                    # Remove a random codeword
                    rem_idx = rng.integers(len(code))
                    rem_c = code[rem_idx]
                    new_coverage = coverage.copy()
                    new_coverage[neighbors[rem_c]] -= 1

                    if np.min(new_coverage) >= 0:
                        code.pop(rem_idx)
                        code_set.remove(rem_c)
                        coverage = new_coverage
                        improved = True

            if not improved:
                # Try adding a random codeword (kick)
                if rng.random() < 0.2:
                    add_c = int(rng.integers(total_points))
                    if add_c not in code_set:
                        code.append(add_c)
                        code_set.add(add_c)
                        coverage = np.zeros(total_points, dtype=np.int32)
                        for idx in code:
                            coverage[neighbors[idx]] += 1

        return best_code

    # Run multiple restarts
    best_code = None
    best_len = float('inf')

    for seed in range(20):
        if time.time() > deadline - 5:
            break

        init_code = greedy_init(seed)
        if not init_code:
            continue

        code = local_search(init_code, seed + 1000)
        code = prune(code)

        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]

    # Final polish
    if best_code is not None:
        # Try additional pruning and small local search
        while time.time() < deadline - 1:
            pruned = prune(best_code)
            if len(pruned) < len(best_code):
                best_code = pruned
                continue
            break

    # Final validity check
    if best_code is not None:
        covered = np.zeros(total_points, dtype=bool)
        for w_idx in best_code:
            covered[neighbors[w_idx]] = True
        if not np.all(covered):
            # Fallback
            _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in _TETRA
                    for tail in itertools.product(range(q), repeat=n - 4)]

        return [list(_int_to_word(int(c), q, n)) for c in best_code]

    # Fallback
    _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]


def _int_to_word(idx, q, n):
    word = [0] * n
    for i in range(n):
        word[i] = idx % q
        idx //= q
    return word


def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    balls = []
    for w in all_words:
        w_idx = word_to_idx[w]
        neighbors = [w_idx]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)

    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    while covered_count > 0 and time.time() < deadline - 5:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1

    def _prune(chosen_list, balls):
        if not chosen_list:
            return []
        total = len(balls)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen_list:
            for j in balls[c]:
                coverage_count[j] += 1

        keep = [True] * len(chosen_list)
        any_removed = True
        while any_removed:
            any_removed = False
            for i, c in enumerate(chosen_list):
                if not keep[i]:
                    continue
                redundant = True
                for j in balls[c]:
                    if coverage_count[j] <= 1:
                        redundant = False
                        break
                if redundant:
                    keep[i] = False
                    any_removed = True
                    for j in balls[c]:
                        coverage_count[j] -= 1
        return [chosen_list[i] for i in range(len(chosen_list)) if keep[i]]

    chosen = _prune(chosen, balls)

    if len(chosen) > 0:
        covered = np.zeros(total, dtype=bool)
        for w_idx in chosen:
            for j in balls[w_idx]:
                covered[j] = True
        if not np.all(covered):
            return [list(w) for w in all_words]

    return [list(all_words[i]) for i in chosen]
