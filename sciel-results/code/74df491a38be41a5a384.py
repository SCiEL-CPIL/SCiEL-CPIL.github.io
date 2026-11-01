import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)

    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode

    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)

    # Precompute balls for all words
    balls = np.zeros((total, ball_size), dtype=np.int32)
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

    # 1. Greedy Construction (vectorized)
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)

    while np.any(current_uncovered):
        if time.time() > deadline - 40.0:
            break

        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)

        if max_count == 0:
            break

        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]

        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)

    # Ensure full coverage
    if np.any(current_uncovered):
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break

    # 2. Aggressive Pruning with multiple passes and varied orderings
    rng = random.Random(42)

    def prune_pass(selected_list, cover_cnt, order='forward'):
        """Remove removable codewords in the given order."""
        changed = False
        if order == 'forward':
            indices = range(len(selected_list))
        elif order == 'reverse':
            indices = range(len(selected_list) - 1, -1, -1)
        else:  # shuffled
            indices = list(range(len(selected_list)))
            rng.shuffle(indices)

        new_selected = []
        for i in indices:
            c = selected_list[i]
            if np.min(cover_cnt[balls[c]]) > 1:
                cover_cnt[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        return new_selected, cover_cnt, changed

    for _ in range(15):
        if time.time() > deadline - 30.0:
            break
        for order in ['forward', 'reverse', 'shuffled']:
            if time.time() > deadline - 30.0:
                break
            selected, cover_count, changed = prune_pass(selected, cover_count, order)
            if not changed:
                break

    # 3. Move-based Local Search
    # When a codeword can't be removed, find a replacement that covers all its
    # critical (count=1) neighbors. This can enable further pruning.
    improved = True
    max_move_iterations = 200
    iter_count = 0
    while improved and time.time() < deadline - 15.0 and iter_count < max_move_iterations:
        improved = False
        iter_count += 1
        selected_set = set(selected)

        for i in range(len(selected)):
            if time.time() > deadline - 15.0:
                break
            c = selected[i]

            # Check if c is removable
            if np.min(cover_count[balls[c]]) > 1:
                # Just remove it
                cover_count[balls[c]] -= 1
                selected.pop(i)
                improved = True
                break

            # Not removable: find critical neighbors (count == 1)
            criticals = balls[c][cover_count[balls[c]] == 1]
            if len(criticals) == 0:
                continue

            # Find a replacement w that covers all criticals
            # and maximizes overlap with already-covered words (to create more redundancy)
            best_w = -1
            best_score = -1

            # Candidate pool: neighbors of c (distance 1) plus some random words
            # For efficiency, check neighbors first
            candidates = list(balls[c])
            # Add some random candidates for diversity
            if time.time() < deadline - 20.0:
                random_sample = rng.sample(range(total), min(50, total))
                candidates.extend(random_sample)

            for w in candidates:
                if w in selected_set:
                    continue

                # Check if w covers all criticals
                w_neighbors = balls[w]
                covers_all = True
                for crit in criticals:
                    if crit not in w_neighbors:
                        covers_all = False
                        break
                if not covers_all:
                    continue

                # Score: number of w's neighbors that are already covered (overlap)
                # Higher overlap means more potential for future removals
                overlap = np.sum(cover_count[w_neighbors] > 0)
                # Also penalize if w creates new uncovered words (shouldn't happen since we keep c)
                score = overlap

                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                # Perform the move: remove c, add best_w
                cover_count[balls[c]] -= 1
                cover_count[balls[best_w]] += 1

                selected[i] = best_w
                selected_set.remove(c)
                selected_set.add(best_w)

                improved = True
                break

    # 4. Final aggressive pruning after moves
    for _ in range(10):
        if time.time() > deadline - 5.0:
            break
        for order in ['forward', 'reverse', 'shuffled']:
            if time.time() > deadline - 5.0:
                break
            selected, cover_count, changed = prune_pass(selected, cover_count, order)
            if not changed:
                break

    # 5. Final verification and conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1

    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
