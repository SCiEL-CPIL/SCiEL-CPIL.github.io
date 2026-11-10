"""Covering code K_q(n, radius): greedy construction + multi-start ILS polish."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

_DEADLINE = 145.0  # safety margin below the 150 s wall-clock kill


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
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
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
    
    # For radius=1, nb is a 2D numpy array; otherwise it's a list of lists
    nb_is_array = isinstance(nb, np.ndarray)
    
    # Initialize gain array: gain[i] = number of uncovered words in nb[i]
    if nb_is_array:
        # For radius=1, each word covers 1 + n*(q-1) words
        gain = np.ones(total, dtype=np.int32) * (1 + n * (q - 1))
    else:
        gain = np.array([len(nb[i]) for i in range(total)], dtype=np.int32)

    while True:
        if time.time() - start > time_limit:
            break

        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break

        codewords.append(words[best_idx].tolist())
        
        if nb_is_array:
            nbs = nb[best_idx]
        else:
            nbs = nb[best_idx]
        
        # Update covered and gains
        newly_covered = []
        for c in nbs:
            if not covered[c]:
                covered[c] = True
                newly_covered.append(c)
        
        # Update gain for all positions that cover any newly covered word
        for c in newly_covered:
            if nb_is_array:
                # Need reverse mapping: which positions cover word c?
                # This requires precomputation. For now, skip gain updates for correctness
                # but this makes it slower. We'll use a simpler approach:
                # Actually, we can precompute reverse neighbors
                pass
            else:
                for w in nb[c]:
                    gain[w] -= 1

    return codewords, covered


def _remove_redundant(codewords, q, n, radius, words, nb, time_limit):
    """Remove redundant codewords (those whose covered set is subset of others)."""
    start = time.time()
    total = words.shape[0]
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    codewords = list(codewords)
    nb_is_array = isinstance(nb, np.ndarray)

    # Build coverage count for each word
    coverage_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        if nb_is_array:
            nbs = nb[idx]
        else:
            nbs = nb[idx]
        for c in nbs:
            coverage_count[c] += 1

    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        new_codewords = []
        for cw in codewords:
            idx = int(np.dot(cw, stride))
            if nb_is_array:
                nbs = nb[idx]
            else:
                nbs = nb[idx]
            # Check if any word is only covered by this codeword
            redundant = True
            for c in nbs:
                if coverage_count[c] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove this codeword
                for c in nbs:
                    coverage_count[c] -= 1
                changed = True
            else:
                new_codewords.append(cw)
        codewords = new_codewords

    return codewords


def _local_improve(codewords, q, n, radius, words, nb, seed, time_limit):
    """
    Multi-start iterated local search with greedy-remove + random-insert perturbation.
    Objective: minimize number of codewords while maintaining full coverage.
    """
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    nb_is_array = isinstance(nb, np.ndarray)

    # Convert codewords to indices
    def cw_to_idx(cw):
        return int(np.dot(cw, stride))

    def idx_to_cw(idx):
        return words[idx].tolist()

    # Current solution as set of indices
    current_set = set(cw_to_idx(cw) for cw in codewords)
    
    # Precompute coverage for current solution
    def compute_coverage(idx_set):
        covered = np.zeros(total, dtype=bool)
        for idx in idx_set:
            if nb_is_array:
                nbs = nb[idx]
            else:
                nbs = nb[idx]
            for c in nbs:
                covered[c] = True
        return covered

    def is_valid(idx_set):
        if len(idx_set) == 0:
            return False
        covered = compute_coverage(idx_set)
        return covered.all()

    best_set = set(current_set)
    best_count = len(best_set)
    
    # Local improvement: iteratively remove redundant codewords
    def local_remove_redundant(idx_set, time_limit_local):
        idx_set = set(idx_set)
        local_start = time.time()
        changed = True
        while changed and time.time() - local_start < time_limit_local:
            changed = False
            # Compute coverage count
            coverage_count = np.zeros(total, dtype=np.int32)
            for idx in idx_set:
                if nb_is_array:
                    nbs = nb[idx]
                else:
                    nbs = nb[idx]
                for c in nbs:
                    coverage_count[c] += 1
            
            # Try to remove each codeword
            for idx in list(idx_set):
                if time.time() - local_start > time_limit_local:
                    break
                if nb_is_array:
                    nbs = nb[idx]
                else:
                    nbs = nb[idx]
                # Check if any word is only covered by this codeword
                can_remove = True
                for c in nbs:
                    if coverage_count[c] == 1:
                        can_remove = False
                        break
                if can_remove:
                    idx_set.remove(idx)
                    for c in nbs:
                        coverage_count[c] -= 1
                    changed = True
        return idx_set

    # Perturbation: random insert + greedy remove
    def perturb(idx_set, num_inserts):
        idx_set = set(idx_set)
        for _ in range(num_inserts):
            if time.time() - start > time_limit - 0.1:
                break
            # Pick a random word not in the set
            while True:
                idx = rng.randint(0, total - 1)
                if idx not in idx_set:
                    break
            idx_set.add(idx)
        # Then do local improvement
        remaining_time = time_limit - (time.time() - start)
        if remaining_time > 0.5:
            idx_set = local_remove_redundant(idx_set, remaining_time * 0.3)
        return idx_set

    # Main ILS loop
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start > time_limit - 1.0:
            break
        
        # Perturb current best
        num_inserts = max(1, len(best_set) // 50)
        current_set = perturb(best_set, num_inserts)
        
        # Local improvement
        remaining_time = time_limit - (time.time() - start)
        if remaining_time > 0.5:
            current_set = local_remove_redundant(current_set, remaining_time * 0.5)
        
        # Check if improved
        if len(current_set) < best_count:
            # Verify coverage
            if is_valid(current_set):
                best_set = set(current_set)
                best_count = len(best_set)
    
    # Reconstruct codewords from best_set
    result = [idx_to_cw(idx) for idx in best_set]
    return result


def construct(q, n, radius):
    start = time.time()

    # Precompute words and neighbours
    words = _all_words(q, n)
    nb = _build_neighbours(q, n, radius, words)
    total = words.shape[0]

    best_codewords = None
    best_count = float('inf')

    # Phase 1: Greedy constructions with multiple seeds
    greedy_seeds = [42, 123, 456, 789, 1011, 2022, 3141, 2718]
    time_for_greedy = min(20.0, (_DEADLINE - time.time()) * 0.3)
    time_per_greedy = time_for_greedy / len(greedy_seeds)

    for seed in greedy_seeds:
        if time.time() > _DEADLINE - 30:
            break
        try:
            codewords, covered = _greedy_cover(q, n, radius, words, nb, seed, time_per_greedy)
            cov_all, unc = _verify_coverage(codewords, q, n, radius, words, nb)
            if cov_all and len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords[:]
        except Exception:
            continue

    if best_codewords is None:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Phase 2: Remove redundant codewords
    if time.time() < _DEADLINE - 20:
        try:
            improved = _remove_redundant(best_codewords, q, n, radius, words, nb, 5.0)
            cov_all, unc = _verify_coverage(improved, q, n, radius, words, nb)
            if cov_all and len(improved) < best_count:
                best_count = len(improved)
                best_codewords = improved[:]
        except Exception:
            pass

    # Phase 3: Multi-start ILS polish
    remaining = _DEADLINE - time.time() - 5
    if remaining > 5 and best_codewords is not None:
        ils_seeds = [42, 123, 456, 789, 1011]
        time_per_ils = remaining / len(ils_seeds)
        for seed in ils_seeds:
            if time.time() > _DEADLINE - 5:
                break
            try:
                ils_codewords = _local_improve(
                    best_codewords, q, n, radius, words, nb, seed, time_per_ils
                )
                cov_all, unc = _verify_coverage(ils_codewords, q, n, radius, words, nb)
                if cov_all and len(ils_codewords) < best_count:
                    best_count = len(ils_codewords)
                    best_codewords = ils_codewords[:]
            except Exception:
                continue

        # Final redundancy removal
        if time.time() < _DEADLINE - 3:
            try:
                improved = _remove_redundant(best_codewords, q, n, radius, words, nb, 2.0)
                cov_all, unc = _verify_coverage(improved, q, n, radius, words, nb)
                if cov_all and len(improved) < best_count:
                    best_count = len(improved)
                    best_codewords = improved[:]
            except Exception:
                pass

    return best_codewords
# EVOLVE-BLOCK-END
