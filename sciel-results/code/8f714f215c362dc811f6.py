"""Football pool K_3(7,1): minimize codewords covering all of {0,1,2}^7 within Hamming distance 1."""
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    if q != 3 or n != 7 or radius != 1:
        return _direct_sum(q, n, radius)

    # Precompute all 2187 words
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 3 ** 7  # 2187

    # Precompute balls as Python int bitmasks
    balls = [0] * N
    for idx, w in enumerate(all_words):
        mask = 1 << idx
        for pos in range(7):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                neighbor = list(w)
                neighbor[pos] = val
                nidx = word_to_idx[tuple(neighbor)]
                mask |= (1 << nidx)
        balls[idx] = mask

    full_mask = (1 << N) - 1

    # Initial solution: direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            initial_codewords.append(h + list(tail))
    initial_indices = [word_to_idx[tuple(w)] for w in initial_codewords]

    selected = set(initial_indices)
    coverage = 0
    for idx in selected:
        coverage |= balls[idx]

    best_count = len(selected)
    best_selected = set(selected)

    # Phase 1: Greedy removal of redundant codewords
    def is_redundant(idx, selected_set):
        cov_without = 0
        for j in selected_set:
            if j != idx:
                cov_without |= balls[j]
        return (balls[idx] & cov_without) == balls[idx]

    for _pass in range(15):
        if time.time() > deadline:
            break
        removed_any = False
        for idx in list(selected):
            if time.time() > deadline:
                break
            if is_redundant(idx, selected):
                selected.discard(idx)
                coverage = 0
                for j in selected:
                    coverage |= balls[j]
                removed_any = True
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = set(selected)
        if not removed_any:
            break

    # Phase 2: Swap-based local search with simulated annealing
    random.seed(42)

    # Precompute coverage_without for each selected word (for fast swap checks)
    def compute_cov_without(selected_set):
        sel_list = list(selected_set)
        m = len(sel_list)
        cov_without = [0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    cov_without[i] |= balls[sel_list[j]]
        return sel_list, cov_without

    # Temperature for simulated annealing
    T = 0.5
    T_min = 0.01
    cooling = 0.995

    for iteration in range(5000):
        if time.time() > deadline:
            break
        if len(selected) == 0:
            break

        sel_list, cov_without = compute_cov_without(selected)
        m = len(sel_list)

        # Try a swap: remove sel_list[i], add random j
        found_swap = False
        # Pick a random i
        i = random.randint(0, m - 1)
        # Try several random j's
        for _trial in range(30):
            if time.time() > deadline:
                break
            j = random.randint(0, N - 1)
            if j in selected:
                continue
            new_cov = cov_without[i] | balls[j]
            if new_cov == full_mask:
                # Valid swap - accept (maintains coverage)
                selected.discard(sel_list[i])
                selected.add(j)
                coverage = new_cov
                found_swap = True
                break

        if found_swap:
            # Try to remove redundant words after swap
            for idx in list(selected):
                if time.time() > deadline:
                    break
                if is_redundant(idx, selected):
                    selected.discard(idx)
                    coverage = 0
                    for j2 in selected:
                        coverage |= balls[j2]
                    if len(selected) < best_count:
                        best_count = len(selected)
                        best_selected = set(selected)

        else:
            # No valid swap found; try a random removal (may break coverage)
            # Only if we can repair quickly
            i = random.randint(0, m - 1)
            removed_idx = sel_list[i]
            selected.discard(removed_idx)
            # Try to find a replacement that restores coverage
            # Compute current coverage without removed_idx
            cur_cov = 0
            for j2 in selected:
                cur_cov |= balls[j2]
            # Find missing bits
            missing = full_mask & ~cur_cov
            if missing:
                # Try to find a word whose ball covers the most missing bits
                best_j = -1
                best_gain = 0
                for _trial in range(50):
                    if time.time() > deadline:
                        break
                    j = random.randint(0, N - 1)
                    if j in selected:
                        continue
                    gain = (balls[j] & missing).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0:
                    selected.add(best_j)
                    coverage = cur_cov | balls[best_j]
                else:
                    # Revert
                    selected.add(removed_idx)
                    coverage = cur_cov | balls[removed_idx]
            else:
                # Coverage still full without the removed word
                coverage = cur_cov
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = set(selected)

        # Cool down
        T = max(T_min, T * cooling)

    # Final verification
    final_coverage = 0
    for idx in best_selected:
        final_coverage |= balls[idx]
    if final_coverage != full_mask:
        best_selected = set(initial_indices)

    return [list(all_words[idx]) for idx in sorted(best_selected)]


def _direct_sum(q, n, radius):
    """Fallback for non-standard parameters."""
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    return [list(w) for w in itertools.product(range(q), repeat=n)]
