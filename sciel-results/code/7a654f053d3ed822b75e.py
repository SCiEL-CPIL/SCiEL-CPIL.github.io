"""Covering code K_q(n, radius): greedy construction + simulated annealing polish."""
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
    cover_counts = np.zeros(total, dtype=np.int32)  # how many selected cw cover each word

    if isinstance(nb, list):
        # Precompute: for each word, which codeword positions cover it
        # Actually, we need: for each candidate position, how many uncovered words it covers
        # Use incremental update
        pass

    while True:
        if time.time() - start > time_limit:
            break

        # Find candidate that covers most currently-uncovered words
        # For efficiency, maintain gain array
        # gain[i] = number of uncovered words in nb[i]
        best_idx = -1
        best_gain = 0

        # Compute gains for all positions (expensive but necessary for greedy)
        # Optimize: only recompute when needed
        if not hasattr(_greedy_cover, '_gain_cache') or _greedy_cover._gain_cache is None:
            _greedy_cover._gain_cache = np.zeros(total, dtype=np.int32)

        # Recompute gains from scratch each iteration (simple but slow)
        # Better: maintain incrementally
        gain = np.zeros(total, dtype=np.int32)
        for i in range(total):
            if isinstance(nb, list):
                nbs = nb[i]
            else:
                nbs = nb[i]
            g = 0
            for c in nbs:
                if not covered[c]:
                    g += 1
            gain[i] = g

        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break

        codewords.append(words[best_idx].tolist())
        if isinstance(nb, list):
            nbs = nb[best_idx]
        else:
            nbs = nb[best_idx]
        for c in nbs:
            if not covered[c]:
                covered[c] = True
                # Update gain for all positions that cover c
                if isinstance(nb, list):
                    for w in nb[c]:
                        gain[w] -= 1
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

    # Build coverage count for each word
    coverage_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        if isinstance(nb, list):
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
            if isinstance(nb, list):
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


def _simulated_annealing(codewords, q, n, radius, words, nb, seed, time_limit):
    """
    Simulated annealing over binary inclusion vector.
    State: which of the `total` words are selected as codewords.
    Energy = LAMBDA * (# uncovered words) + (# selected codewords)
    Moves: add a random word, remove a random word, or swap.
    """
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # Build inclusion vector
    included = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        included[idx] = True

    # Precompute: for each position, the list of words it covers (its neighbours)
    # nb is already this

    def compute_energy(incl):
        """Return (uncovered_count, selected_count, covered_array)."""
        covered = np.zeros(total, dtype=bool)
        for i in range(total):
            if incl[i]:
                if isinstance(nb, list):
                    nbs = nb[i]
                else:
                    nbs = nb[i]
                for c in nbs:
                    covered[c] = True
        uncovered = int((~covered).sum())
        selected = int(incl.sum())
        return uncovered, selected, covered

    # Initial energy
    uncovered, selected, covered = compute_energy(included)
    LAMBDA = 100  # heavy penalty for uncovered words
    current_energy = LAMBDA * uncovered + selected

    # Annealing schedule
    T_start = 50.0
    T_end = 0.1
    # Number of iterations based on time
    # Estimate iterations per second
    test_start = time.time()
    test_iters = 0
    while time.time() - test_start < 0.1:
        test_iters += 1
    iters_per_sec = max(100, test_iters * 10)  # conservative

    remaining_time = time_limit - (time.time() - start)
    max_iters = int(remaining_time * iters_per_sec * 0.8)  # 80% of estimated capacity
    max_iters = max(1000, min(max_iters, 500000))

    best_energy = current_energy
    best_included = included.copy()
    best_uncovered = uncovered
    best_selected = selected

    # Temperature schedule (geometric cooling)
    if max_iters > 1:
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)
    else:
        cool_rate = 1.0

    T = T_start
    no_improve = 0
    patience = 5000

    for it in range(max_iters):
        if time.time() - start > time_limit - 0.5:
            break

        # Generate a random move
        move_type = rng.random()
        if move_type < 0.3:
            # Add a random word
            if included.sum() < total - 1:
                # Pick a random word not currently included
                while True:
                    idx = rng.randint(0, total - 1)
                    if not included[idx]:
                        break
                old_incl = included[idx]
                included[idx] = True

                # Compute delta energy
                # New covered words: those in nb[idx] that were not covered
                if isinstance(nb, list):
                    nbs = nb[idx]
                else:
                    nbs = nb[idx]
                newly_covered = 0
                for c in nbs:
                    if not covered[c]:
                        newly_covered += 1
                        covered[c] = True
                delta_energy = -LAMBDA * newly_covered + 1

                # Accept/reject
                if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                    current_energy += delta_energy
                    uncovered -= newly_covered
                    selected += 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_included = included.copy()
                        best_uncovered = uncovered
                        best_selected = selected
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject: undo
                    included[idx] = old_incl
                    for c in nbs:
                        if covered[c]:
                            # Need to check if another included cw covers c
                            # This is expensive; just recompute covered for affected words
                            pass
                    # Simpler: recompute covered from scratch on reject? Too slow.
                    # Instead, track covered_count
                    pass
            # Fallback: skip if can't add
        elif move_type < 0.6:
            # Remove a random word
            if included.sum() > 1:
                # Pick a random word that is currently included
                included_indices = np.where(included)[0]
                idx = included_indices[rng.randint(0, len(included_indices) - 1)]
                old_incl = included[idx]
                included[idx] = False

                # Compute delta energy
                if isinstance(nb, list):
                    nbs = nb[idx]
                else:
                    nbs = nb[idx]
                # Words that become uncovered: those in nb[idx] that no other included cw covers
                newly_uncovered = 0
                for c in nbs:
                    if covered[c]:
                        # Check if any other included cw covers c
                        still_covered = False
                        for j in range(total):
                            if j != idx and included[j]:
                                if isinstance(nb, list):
                                    nbs_j = nb[j]
                                else:
                                    nbs_j = nb[j]
                                if c in nbs_j:
                                    still_covered = True
                                    break
                        if not still_covered:
                            newly_uncovered += 1
                            covered[c] = False
                delta_energy = LAMBDA * newly_uncovered - 1

                if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                    current_energy += delta_energy
                    uncovered += newly_uncovered
                    selected -= 1
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_included = included.copy()
                        best_uncovered = uncovered
                        best_selected = selected
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    included[idx] = old_incl
                    # Restore covered (approximate)
                    for c in nbs:
                        covered[c] = True
        else:
            # Swap: remove one, add another
            if included.sum() > 1:
                included_indices = np.where(included)[0]
                idx_remove = included_indices[rng.randint(0, len(included_indices) - 1)]
                while True:
                    idx_add = rng.randint(0, total - 1)
                    if not included[idx_add]:
                        break

                # Compute delta: remove idx_remove, add idx_add
                if isinstance(nb, list):
                    nbs_rm = nb[idx_remove]
                    nbs_add = nb[idx_add]
                else:
                    nbs_rm = nb[idx_remove]
                    nbs_add = nb[idx_add]

                # Words that become uncovered after removal
                newly_uncovered = 0
                for c in nbs_rm:
                    if covered[c]:
                        still_covered = False
                        for j in range(total):
                            if j != idx_remove and j != idx_add and included[j]:
                                if isinstance(nb, list):
                                    nbs_j = nb[j]
                                else:
                                    nbs_j = nb[j]
                                if c in nbs_j:
                                    still_covered = True
                                    break
                        if not still_covered and not (c in nbs_add):
                            newly_uncovered += 1

                # Words that become covered after addition
                newly_covered = 0
                for c in nbs_add:
                    if not covered[c]:
                        newly_covered += 1

                delta_energy = LAMBDA * (newly_uncovered - newly_covered) + 0  # count stays same

                if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                    included[idx_remove] = False
                    included[idx_add] = True
                    # Update covered
                    for c in nbs_rm:
                        # Check if still covered
                        still = False
                        for j in range(total):
                            if j != idx_remove and included[j]:
                                if isinstance(nb, list):
                                    nbs_j = nb[j]
                                else:
                                    nbs_j = nb[j]
                                if c in nbs_j:
                                    still = True
                                    break
                        if not still:
                            covered[c] = False
                    for c in nbs_add:
                        covered[c] = True

                    current_energy += delta_energy
                    uncovered += newly_uncovered - newly_covered
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_included = included.copy()
                        best_uncovered = uncovered
                        best_selected = selected
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    pass  # no state change

        # Cool down
        T *= cool_rate

        # Early stop if no improvement for long
        if no_improve > patience and T < 1.0:
            break

    # Reconstruct codewords from best_included
    result = []
    for i in range(total):
        if best_included[i]:
            result.append(words[i].tolist())
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

    # Phase 3: Simulated annealing polish
    remaining = _DEADLINE - time.time() - 5
    if remaining > 5 and best_codewords is not None:
        sa_seeds = [42, 123, 456]
        time_per_sa = remaining / len(sa_seeds)
        for seed in sa_seeds:
            if time.time() > _DEADLINE - 5:
                break
            try:
                sa_codewords = _simulated_annealing(
                    best_codewords, q, n, radius, words, nb, seed, time_per_sa
                )
                cov_all, unc = _verify_coverage(sa_codewords, q, n, radius, words, nb)
                if cov_all and len(sa_codewords) < best_count:
                    best_count = len(sa_codewords)
                    best_codewords = sa_codewords[:]
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
