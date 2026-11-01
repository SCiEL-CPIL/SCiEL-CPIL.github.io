"""Iterated local search for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 120.0
_SAFETY = 5.0  # seconds margin before wall-clock kill


def _hamming_dist(a, b):
    return int(np.count_nonzero(np.frombuffer(a, dtype=np.uint8) != np.frombuffer(b, dtype=np.uint8)))


def _covers(codewords, all_words, q, n, radius):
    """Return True if every word in all_words is within radius of some codeword."""
    if not codewords:
        return False
    cw = np.array(codewords, dtype=np.uint8)
    aw = np.array(all_words, dtype=np.uint8)
    # For each all_word, check if any codeword is within radius
    # Use broadcasting in chunks to limit memory
    chunk = 5000
    for start in range(0, len(aw), chunk):
        end = min(start + chunk, len(aw))
        # shape (chunk, 1, n) vs (1, m, n)
        diff = np.abs(aw[start:end, None, :].astype(np.int16) - cw[None, :, :].astype(np.int16))
        dists = diff.sum(axis=2)  # (chunk, m)
        if not np.all(dists.min(axis=1) <= radius):
            return False
    return True


def _greedy_cover(q, n, radius):
    """Greedy set-cover: repeatedly pick the word that covers the most uncovered words."""
    all_words = np.array(np.meshgrid(*[np.arange(q) for _ in range(n)]), indexing='ij').reshape(n, -1).T
    # all_words shape (q**n, n), uint8
    all_words = all_words.astype(np.uint8)
    total = len(all_words)
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    codewords = []
    radius_arr = np.arange(radius + 1)  # not directly used

    # Precompute: for each candidate position, the set of words within radius
    # We'll compute on the fly in chunks
    while uncovered_count > 0:
        best_idx = -1
        best_gain = 0
        # Evaluate candidates in chunks to limit memory
        chunk = 2000
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            cand = all_words[start:end]
            # dist from each candidate to each uncovered word
            # (chunk, 1, n) vs (1, total, n) -> too big; do (chunk, total, n)
            # Instead: for each candidate, count how many uncovered words are within radius
            diff = np.abs(cand[:, None, :].astype(np.int16) - all_words[None, :, :].astype(np.int16))
            dists = diff.sum(axis=2)  # (chunk, total)
            within = dists <= radius  # (chunk, total)
            gains = within[:, uncovered].sum(axis=1)
            local_best = int(np.argmax(gains))
            if gains[local_best] > best_gain:
                best_gain = int(gains[local_best])
                best_idx = start + local_best
        if best_idx < 0 or best_gain == 0:
            break
        codewords.append(all_words[best_idx].tolist())
        # Update uncovered
        diff = np.abs(all_words[best_idx][None, :].astype(np.int16) - all_words.astype(np.int16))
        dists = diff.sum(axis=1)
        newly_covered = dists <= radius
        newly_uncovered = newly_covered & uncovered
        uncovered_count -= int(newly_uncovered.sum())
        uncovered[uncovered & newly_covered] = False
    return codewords


def _reduce(codewords, all_words, q, n, radius):
    """Try to remove redundant codewords."""
    if not codewords:
        return codewords
    result = list(codewords)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            # Try removing result[i]
            candidate = result[:i] + result[i+1:]
            if _covers(candidate, all_words, q, n, radius):
                result = candidate
                changed = True
                break
    return result


def _mutate(codewords, q, n, radius, all_words, num_moves=3):
    """Randomly add/remove/swap codewords."""
    result = list(codewords)
    for _ in range(num_moves):
        op = random.choice(['add', 'remove', 'swap'])
        if op == 'add' and len(result) < 200:
            idx = random.randrange(len(all_words))
            new_cw = all_words[idx].tolist()
            if new_cw not in result:
                result.append(new_cw)
        elif op == 'remove' and len(result) > 10:
            idx = random.randrange(len(result))
            result.pop(idx)
        elif op == 'swap' and len(result) > 10:
            idx = random.randrange(len(result))
            new_idx = random.randrange(len(all_words))
            new_cw = all_words[new_idx].tolist()
            if new_cw not in result:
                result[idx] = new_cw
    return result


def _repair(codewords, all_words, q, n, radius):
    """Ensure coverage by adding words if needed."""
    if not codewords:
        return _greedy_cover(q, n, radius)
    if _covers(codewords, all_words, q, n, radius):
        return codewords
    # Greedily add until covered
    result = list(codewords)
    # Find uncovered words
    uncov = _find_uncovered(result, all_words, radius)
    while uncov:
        # Pick the word that covers the most uncovered
        best_idx = -1
        best_gain = 0
        chunk = 2000
        for start in range(0, len(all_words), chunk):
            end = min(start + chunk, len(all_words))
            cand = all_words[start:end]
            diff = np.abs(cand[:, None, :].astype(np.int16) - all_words[None, :, :].astype(np.int16))
            dists = diff.sum(axis=2)
            within = dists <= radius
            gains = within[:, uncov].sum(axis=1)
            local_best = int(np.argmax(gains))
            if gains[local_best] > best_gain:
                best_gain = int(gains[local_best])
                best_idx = start + local_best
        if best_idx < 0 or best_gain == 0:
            break
        result.append(all_words[best_idx].tolist())
        uncov = _find_uncovered(result, all_words, radius)
    return result


def _find_uncovered(codewords, all_words, radius):
    """Return boolean mask of all_words not covered by codewords."""
    if not codewords:
        return np.ones(len(all_words), dtype=bool)
    cw = np.array(codewords, dtype=np.uint8)
    aw = all_words
    total = len(aw)
    uncovered = np.ones(total, dtype=bool)
    chunk = 5000
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        diff = np.abs(aw[start:end, None, :].astype(np.int16) - cw[None, :, :].astype(np.int16))
        dists = diff.sum(axis=2)
        covered = dists.min(axis=1) <= radius
        uncovered[start:end] = ~covered
    return uncovered


def _local_search(codewords, all_words, q, n, radius, max_iters=200, deadline=None):
    """Iterated local search: mutate, repair, reduce, keep if better."""
    best = list(codewords)
    best_size = len(best)
    current = list(best)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Mutate
        mutated = _mutate(current, q, n, radius, all_words, num_moves=random.choice([1, 2, 3]))
        # Repair to ensure coverage
        repaired = _repair(mutated, all_words, q, n, radius)
        # Reduce
        reduced = _reduce(repaired, all_words, q, n, radius)
        # Accept if better or equal size (for diversity)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(best)
            current = list(best)
        elif len(reduced) == best_size and random.random() < 0.3:
            current = reduced
    return best


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY

    # Precompute all words
    all_words = np.array(np.meshgrid(*[np.arange(q) for _ in range(n)]), indexing='ij').reshape(n, -1).T
    all_words = all_words.astype(np.uint8)

    # Initial greedy solution
    codewords = _greedy_cover(q, n, radius)
    codewords = _reduce(codewords, all_words, q, n, radius)
    best = codewords

    # Iterated local search with multiple restarts
    random.seed(42)
    np.random.seed(42)

    while time.time() < deadline:
        # Try a few iterations of local search
        improved = _local_search(best, all_words, q, n, radius,
                                  max_iters=50, deadline=deadline)
        if len(improved) < len(best):
            best = improved
        # Occasionally do a fresh greedy restart with different randomization
        if time.time() < deadline and random.random() < 0.2:
            # Perturb greedy: start with random subset then greedy complete
            random_subset_size = max(1, len(best) // 3)
            random_indices = random.sample(range(len(all_words)), random_subset_size)
            seed_cw = [all_words[i].tolist() for i in random_indices]
            completed = _repair(seed_cw, all_words, q, n, radius)
            completed = _reduce(completed, all_words, q, n, radius)
            if len(completed) < len(best):
                best = completed

    # Final reduction pass
    best = _reduce(best, all_words, q, n, radius)
    return best
# EVOLVE-BLOCK-END
