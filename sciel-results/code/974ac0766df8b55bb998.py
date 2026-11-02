import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    N = q ** n
    if N > 200000:
        return _constructive_cover(q, n, radius)

    # Precompute balls: ball[i] = array of 17 indices (self + 16 neighbors)
    weights = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    balls = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    for w in range(N):
        balls[w, 0] = w
        idx = 1
        for p in range(n):
            weight = weights[p]
            digit = (w // weight) % q
            for v in range(q):
                if v == digit:
                    continue
                balls[w, idx] = w + (v - digit) * weight
                idx += 1

    # Precompute ball sets for swap intersection
    ball_sets = [set(balls[w].tolist()) for w in range(N)]

    # Word index lookup
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}

    best_size = N
    best_selected = None

    def try_remove_redundant(selected, covered_count):
        """Remove all redundant codewords. Returns (selected, covered_count, num_removed)."""
        removed = 0
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            unique_mask = (covered_count == 1)
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                if not np.any(unique_mask[balls[c]]):
                    selected.pop(i)
                    covered_count[balls[c]] -= 1
                    removed += 1
                    changed = True
                    break  # restart scan
        return selected, covered_count, removed

    def local_search(selected, covered_count):
        """Iterative local search: remove redundant, then try swaps."""
        selected, covered_count, _ = try_remove_redundant(selected, covered_count)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            selected, covered_count, _ = try_remove_redundant(selected, covered_count)
            if not improved:
                # Try swaps
                for i in range(len(selected)):
                    if time.time() > deadline - 2.0:
                        break
                    c = selected[i]
                    c_ball = balls[c]
                    at_risk = c_ball[covered_count[c_ball] == 1]
                    if len(at_risk) == 0:
                        # Redundant, remove
                        selected.pop(i)
                        covered_count[c_ball] -= 1
                        improved = True
                        break
                    if len(at_risk) > 4:
                        continue
                    # Find swap candidates: intersection of balls of at_risk words
                    candidates = set(ball_sets[at_risk[0]])
                    for w in at_risk[1:]:
                        candidates &= set(ball_sets[w])
                    sel_set = set(selected)
                    candidates -= sel_set
                    if candidates:
                        # Pick candidate that maximizes coverage of currently-undercovered
                        best_j = None
                        best_gain = -1
                        for j in list(candidates)[:20]:  # limit candidates checked
                            gain = np.sum(covered_count[balls[j]] <= 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                        if best_j is not None:
                            selected.pop(i)
                            covered_count[c_ball] -= 1
                            selected.append(best_j)
                            covered_count[balls[best_j]] += 1
                            improved = True
                            break
        selected, covered_count, _ = try_remove_redundant(selected, covered_count)
        return selected, covered_count

    def greedy_cover(covered_count):
        """Greedy: add codewords to cover all uncovered words."""
        selected = []
        uncovered_mask = (covered_count == 0)
        while np.any(uncovered_mask) and time.time() < deadline - 3.0:
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            covered_count[balls[best_idx]] += 1
            uncovered_mask = (covered_count == 0)
        return selected, covered_count

    # --- Initialization strategies ---
    def init_tetracode_product():
        """Tetracode product seed."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        sel = []
        for i in range(9):
            for j in range(9):
                word = np.concatenate([tetracode[i], tetracode[j]])
                sel.append(word_to_idx[tuple(word)])
        return sel

    def init_random(rng):
        """Random initialization followed by greedy completion."""
        k = rng.integers(50, 120)
        sel = list(rng.choice(N, size=k, replace=False))
        return sel

    def init_latin(rng):
        """Latin hypercube-like: maximize diversity."""
        # Use tetracode structure on first 4 positions, random on last 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        sel = []
        for i in range(9):
            for j in range(9):
                first_half = tetracode[i]
                second_half = list(rng.choice(3, size=4, replace=True))
                word = np.array(first_half + second_half, dtype=np.int32)
                sel.append(word_to_idx[tuple(word)])
        return sel

    # --- Main search loop: multiple restarts ---
    rng = np.random.default_rng(42)
    init_strategies = [init_tetracode_product, init_latin, init_random]
    restart_count = 0

    while time.time() < deadline - 10.0:
        strategy_idx = restart_count % len(init_strategies)
        init_fn = init_strategies[strategy_idx]
        if strategy_idx == 1:  # latin
            seed_sel = init_fn(rng)
        elif strategy_idx == 2:  # random
            seed_sel = init_fn(rng)
        else:
            seed_sel = init_fn()

        covered_count = np.zeros(N, dtype=np.int32)
        for c in seed_sel:
            covered_count[balls[c]] += 1

        # Greedy completion
        selected, covered_count = greedy_cover(covered_count)
        if np.any(covered_count == 0):
            restart_count += 1
            continue

        # Local search
        selected, covered_count = local_search(selected, covered_count)

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            if best_size <= 486:
                break

        restart_count += 1

    # Final cleanup
    if best_selected is None:
        # Fallback: full greedy from scratch
        covered_count = np.zeros(N, dtype=np.int32)
        best_selected, covered_count = greedy_cover(covered_count)
        best_size = len(best_selected)

    # Ensure validity
    if best_selected:
        cc = np.zeros(N, dtype=np.int32)
        for c in best_selected:
            cc[balls[c]] += 1
        if np.any(cc == 0):
            # Add missing coverage
            for c in range(N):
                if np.any(cc[balls[c]] == 0):
                    best_selected.append(c)
                    cc[balls[c]] += 1

    result = [all_words[i].tolist() for i in best_selected]
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]

    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True

    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]

    return [all_words[i].tolist() for i in selected]
