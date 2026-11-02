import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n

    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: each word is covered by 1 + 2*n = 17 words (itself + 16 neighbors)
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    best_size = total
    best_set = None

    def greedy_construct():
        """Deterministic greedy construction."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]

            selected.append(int(best_idx))
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        return selected, cover_count

    def greedy_construct_random():
        """Randomized greedy construction."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(int(best_idx))
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        return selected, cover_count

    def remove_redundant(selected, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count

    def local_search_1swap(selected, cover_count, deadline):
        """1-swap local search."""
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]

                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find candidates that cover all critical words
                candidates = set(balls[critical_words[0]].tolist())
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break

                candidates.discard(c)

                if candidates:
                    c_prime = random.choice(list(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)

            selected = new_selected

        return selected, cover_count

    def local_search_2exchange(selected, cover_count, deadline):
        """2-exchange local search: remove 2 codewords, add 1."""
        if len(selected) <= 403:
            return selected, cover_count

        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Try random pairs
            num_trials = min(5000, len(selected) * (len(selected) - 1) // 2)

            for _ in range(num_trials):
                if time.time() > deadline - 3.0:
                    break

                idx1 = random.randint(0, len(selected) - 1)
                idx2 = random.randint(0, len(selected) - 1)
                if idx1 == idx2:
                    continue

                c1 = selected[idx1]
                c2 = selected[idx2]

                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1

                uncovered_after = (cover_count == 0)

                if np.any(uncovered_after):
                    uncovered_words = np.where(uncovered_after)[0]

                    candidates = set(balls[uncovered_words[0]].tolist())
                    for uw in uncovered_words[1:]:
                        candidates.intersection_update(balls[uw].tolist())
                        if not candidates:
                            break

                    candidates.discard(c1)
                    candidates.discard(c2)

                    if candidates:
                        c3 = random.choice(list(candidates))
                        cover_count[balls[c3]] += 1
                        new_selected = [c for c in selected if c != c1 and c != c2]
                        new_selected.append(c3)
                        selected = new_selected
                        improved = True
                        break
                    else:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
                else:
                    new_selected = [c for c in selected if c != c1 and c != c2]
                    selected = new_selected
                    improved = True
                    break

        return selected, cover_count

    def polish_pass(selected, cover_count, deadline):
        """Final polish: more aggressive removal and swaps."""
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            selected, cover_count = remove_redundant(selected, cover_count)
            selected, cover_count = local_search_1swap(selected, cover_count, deadline)
            selected, cover_count = remove_redundant(selected, cover_count)

        return selected, cover_count

    # Multiple restarts
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1

        # Alternate between deterministic and randomized greedy
        if iteration % 3 == 0:
            selected, cover_count = greedy_construct()
        else:
            selected, cover_count = greedy_construct_random()

        if time.time() > deadline - 10.0:
            continue

        # Local search
        selected, cover_count = remove_redundant(selected, cover_count)
        selected, cover_count = local_search_1swap(selected, cover_count, deadline)
        selected, cover_count = remove_redundant(selected, cover_count)
        selected, cover_count = local_search_2exchange(selected, cover_count, deadline)
        selected, cover_count = polish_pass(selected, cover_count, deadline)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

            if best_size <= 403:
                break

    # Fallback if no valid solution found
    if best_set is None:
        selected, cover_count = greedy_construct()
        best_set = selected

    # Convert to output format
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
