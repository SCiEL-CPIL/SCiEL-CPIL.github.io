"""Covering code K_q(n, radius): randomized construction + local improvement."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

_DEADLINE = 140.0  # safety margin below the 150 s wall-clock kill


def _all_words(q, n):
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words


def _build_neighbours(q, n, radius, words):
    """Precompute neighbour lists (indices within Hamming distance <= radius)."""
    total = words.shape[0]
    words_int = words.astype(np.int64)
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    if radius == 1:
        nb = np.empty((total, 1 + n * (q - 1)), dtype=np.int64)
        for idx in range(total):
            digits = words_int[idx]
            pos = 0
            nb[idx, pos] = idx
            pos += 1
            for i in range(n):
                v = digits[i]
                for vp in range(q):
                    if vp == v:
                        continue
                    nb[idx, pos] = idx + (vp - v) * stride[i]
                    pos += 1
        return nb
    else:
        # General radius: enumerate patterns
        patterns = []
        for k in range(radius + 1):
            for subset in itertools.combinations(range(n), k):
                if k == 0:
                    patterns.append([])
                else:
                    for assign in itertools.product(range(q), repeat=k):
                        patterns.append([(subset[j], assign[j]) for j in range(k)])
        nb_list = []
        for idx in range(total):
            digits = words_int[idx]
            nbs = set()
            for pat in patterns:
                new_idx = idx
                for (coord, new_sym) in pat:
                    new_idx += (new_sym - digits[coord]) * stride[coord]
                if 0 <= new_idx < total:
                    nbs.add(new_idx)
            nb_list.append(sorted(nbs))
        return nb_list


def _verify_coverage(codewords, q, n, radius, words, nb):
    """Check if codewords cover all words. Returns (covered_all, uncovered_count)."""
    total = words.shape[0]
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, np.array([q ** (n - 1 - i) for i in range(n)])))
        if isinstance(nb, list):
            nbs = nb[idx]
        else:
            nbs = nb[idx]
        for c in nbs:
            covered[c] = True
    return covered.all(), int((~covered).sum())


def _greedy_cover(q, n, radius, words, nb, seed, time_limit):
    """Greedy set cover with deadline guard."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()

    covered = np.zeros(total, dtype=bool)
    codewords = []

    if isinstance(nb, list):
        uncovered_count = np.array([len(nb[i]) for i in range(total)], dtype=np.int64)
    else:
        uncovered_count = np.full(total, nb.shape[1], dtype=np.int64)

    while True:
        if time.time() - start > time_limit:
            break
        best_idx = int(np.argmax(uncovered_count))
        if uncovered_count[best_idx] == 0:
            break
        codewords.append(words[best_idx].tolist())
        if isinstance(nb, list):
            nbs = nb[best_idx]
        else:
            nbs = nb[best_idx]
        for c in nbs:
            if not covered[c]:
                covered[c] = True
                if isinstance(nb, list):
                    for w in nb[c]:
                        uncovered_count[w] -= 1
                else:
                    for w in nb[c]:
                        uncovered_count[w] -= 1
    return codewords, covered


def _local_improve(codewords, q, n, radius, words, nb, seed, time_limit):
    """Try to remove redundant codewords and swap for better ones."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    codewords = list(codewords)

    # Build index map
    word_to_idx = {}
    for i, w in enumerate(words):
        word_to_idx[tuple(w)] = i

    # Helper: check if a codeword is redundant (its neighbours are all covered by others)
    def is_redundant(idx, codeword_set, covered_by_others):
        # covered_by_others is a set of word indices covered by other codewords
        if isinstance(nb, list):
            nbs = nb[idx]
        else:
            nbs = nb[idx]
        for c in nbs:
            if c not in covered_by_others:
                return False
        return True

    # Iteratively try to remove codewords
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Compute coverage for each codeword
        # For each codeword, find words it uniquely covers
        all_words_covered = np.zeros(total, dtype=np.int32)  # count of codewords covering each word
        for cw in codewords:
            idx = word_to_idx[tuple(cw)]
            if isinstance(nb, list):
                nbs = nb[idx]
            else:
                nbs = nb[idx]
            for c in nbs:
                all_words_covered[c] += 1

        # Try removing each codeword
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            cw = codewords[i]
            idx = word_to_idx[tuple(cw)]
            if isinstance(nb, list):
                nbs = nb[idx]
            else:
                nbs = nb[idx]
            # Check if any word is only covered by this codeword
            redundant = True
            for c in nbs:
                if all_words_covered[c] == 1:
                    redundant = False
                    break
            if redundant:
                codewords.pop(i)
                improved = True
                # Recompute coverage
                all_words_covered = np.zeros(total, dtype=np.int32)
                for cw2 in codewords:
                    idx2 = word_to_idx[tuple(cw2)]
                    if isinstance(nb, list):
                        nbs2 = nb[idx2]
                    else:
                        nbs2 = nb[idx2]
                    for c in nbs2:
                        all_words_covered[c] += 1

    # Try swaps: replace one codeword with another that covers more uncovered words
    # This is more complex; skip for now and rely on removal
    return codewords


def _random_construction(q, n, radius, words, nb, seed, time_limit):
    """Randomized construction: start with random codewords, then improve."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()

    # Start with a reasonable number of random codewords
    # Lower bound: total / (1 + n*(q-1)) for radius 1
    lb = total // (1 + n * (q - 1))
    initial_size = int(lb * 1.3)  # Start with 30% above lower bound

    codewords = []
    covered = np.zeros(total, dtype=bool)
    used = set()

    while len(codewords) < initial_size and time.time() - start < time_limit * 0.3:
        idx = rng.randint(0, total - 1)
        if idx in used:
            continue
        used.add(idx)
        codewords.append(words[idx].tolist())
        if isinstance(nb, list):
            nbs = nb[idx]
        else:
            nbs = nb[idx]
        for c in nbs:
            covered[c] = True

    # Fill gaps with greedy
    if not covered.all():
        uncovered = np.where(~covered)[0]
        for idx in uncovered:
            if time.time() - start > time_limit * 0.5:
                break
            if not covered[idx]:
                codewords.append(words[idx].tolist())
                if isinstance(nb, list):
                    nbs = nb[idx]
                else:
                    nbs = nb[idx]
                for c in nbs:
                    covered[c] = True

    # Local improvement
    codewords = _local_improve(codewords, q, n, radius, words, nb, seed, time_limit * 0.5)

    return codewords


def _hill_climbing(codewords, q, n, radius, words, nb, seed, time_limit):
    """Hill climbing: try adding/removing/swapping codewords to improve coverage."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    codewords = list(codewords)

    word_to_idx = {}
    for i, w in enumerate(words):
        word_to_idx[tuple(w)] = i

    # Compute current coverage
    def compute_coverage(cw_list):
        covered = np.zeros(total, dtype=bool)
        for cw in cw_list:
            idx = word_to_idx[tuple(cw)]
            if isinstance(nb, list):
                nbs = nb[idx]
            else:
                nbs = nb[idx]
            for c in nbs:
                covered[c] = True
        return covered

    covered = compute_coverage(codewords)
    current_uncovered = int((~covered).sum())

    while time.time() - start < time_limit:
        improved = False

        # Try removing redundant codewords
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            cw = codewords[i]
            idx = word_to_idx[tuple(cw)]
            if isinstance(nb, list):
                nbs = nb[idx]
            else:
                nbs = nb[idx]
            # Check if all neighbours are covered by other codewords
            temp_covered = covered.copy()
            for c in nbs:
                temp_covered[c] = False
            # Check if any word becomes uncovered
            still_covered = True
            for c in nbs:
                if not covered[c] and c not in [c2 for c2 in nbs if c2 != c]:
                    # This word is only covered by this codeword
                    still_covered = False
                    break
            # More efficient: check if removing this codeword leaves all words covered
            # A word is uncovered if it was only covered by this codeword
            # We need to know for each word how many codewords cover it
            pass

        # Simpler: just try random swaps
        if len(codewords) > 10:
            i = rng.randint(0, len(codewords) - 1)
            # Try replacing codewords[i] with a random word
            new_idx = rng.randint(0, total - 1)
            new_cw = words[new_idx].tolist()
            # Check if this swap maintains coverage and possibly reduces count
            # For now, just accept if it doesn't increase uncovered
            test_cw = codewords[:]
            test_cw[i] = new_cw
            test_covered = compute_coverage(test_cw)
            test_uncovered = int((~test_covered).sum())
            if test_uncovered <= current_uncovered:
                codewords = test_cw
                covered = test_covered
                current_uncovered = test_uncovered
                improved = True

        if not improved:
            # Try adding a codeword that covers the most uncovered words
            if current_uncovered > 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) > 0:
                    # Find word that covers most uncovered words
                    best_gain = 0
                    best_new_idx = -1
                    for idx in uncovered[:100]:  # Sample
                        if isinstance(nb, list):
                            nbs = nb[idx]
                        else:
                            nbs = nb[idx]
                        gain = sum(1 for c in nbs if not covered[c])
                        if gain > best_gain:
                            best_gain = gain
                            best_new_idx = idx
                    if best_new_idx >= 0:
                        codewords.append(words[best_new_idx].tolist())
                        if isinstance(nb, list):
                            nbs = nb[best_new_idx]
                        else:
                            nbs = nb[best_new_idx]
                        for c in nbs:
                            covered[c] = True
                        current_uncovered = int((~covered).sum())
                        improved = True

    return codewords


def construct(q, n, radius):
    start = time.time()

    # Precompute words and neighbours
    words = _all_words(q, n)
    nb = _build_neighbours(q, n, radius, words)
    total = words.shape[0]

    best_codewords = None
    best_count = float('inf')

    # Multiple strategies with different seeds
    strategies = [
        ('greedy', 42),
        ('greedy', 123),
        ('greedy', 456),
        ('random', 789),
        ('random', 1011),
        ('random', 2022),
    ]

    time_per_strategy = (time.time() - start + 0)
    remaining = _DEADLINE - time.time()
    time_budget = remaining / max(1, len(strategies))

    for strategy, seed in strategies:
        if time.time() > _DEADLINE - 5:
            break
        try:
            if strategy == 'greedy':
                codewords, covered = _greedy_cover(q, n, radius, words, nb, seed, time_budget * 0.8)
            else:
                codewords = _random_construction(q, n, radius, words, nb, seed, time_budget * 0.6)

            # Verify
            if isinstance(nb, list):
                cov_all, unc = _verify_coverage(codewords, q, n, radius, words, nb)
            else:
                cov_all, unc = _verify_coverage(codewords, q, n, radius, words, nb)

            if cov_all and len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords[:]

            # Local improvement on the best so far
            if best_codewords and time.time() < _DEADLINE - 10:
                improved = _local_improve(best_codewords, q, n, radius, words, nb, seed, 5.0)
                if isinstance(nb, list):
                    cov_all2, unc2 = _verify_coverage(improved, q, n, radius, words, nb)
                else:
                    cov_all2, unc2 = _verify_coverage(improved, q, n, radius, words, nb)
                if cov_all2 and len(improved) < best_count:
                    best_count = len(improved)
                    best_codewords = improved[:]

        except Exception:
            continue

    if best_codewords is not None:
        return best_codewords

    # Fallback: direct sum construction
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(3), repeat=n)]
# EVOLVE-BLOCK-END
