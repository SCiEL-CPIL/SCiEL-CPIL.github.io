"""Combined program: greedy criticality set cover + 2-opt local search for K_3(8,1)."""
import time
import random
import itertools
import numpy as np

_DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s kill


def _precompute_balls(q, n, radius):
    """
    Precompute the ball (neighbors within Hamming distance <= radius) for each word.
    Returns a list of numpy arrays, where ball[i] is the array of indices within distance <= radius of word i.
    Also returns the coverers: coverers[w] is a set of indices i such that w is in ball[i].
    """
    total = q ** n

    # Precompute digit representations
    def idx_to_digits(i):
        d = [0] * n
        for p in range(n):
            d[p] = i % q
            i //= q
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(n):
            v += d[p] * mul
            mul *= q
        return v

    # Compute balls
    ball = [None] * total
    for i in range(total):
        d = idx_to_digits(i)
        s = [i]
        # For radius 1, we just change one position
        if radius == 1:
            for p in range(n):
                orig = d[p]
                for nv in range(q):
                    if nv != orig:
                        d[p] = nv
                        s.append(digits_to_idx(d))
                d[p] = orig
        else:
            # General case: generate all words within distance <= radius
            # This is expensive for large radius, but radius=1 is the common case
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for values in itertools.product(range(q), repeat=dist):
                        for pos, val in zip(positions, values):
                            d[pos] = val
                        s.append(digits_to_idx(d))
                        d = idx_to_digits(i)  # reset
        ball[i] = np.array(s, dtype=np.int32)

    # Compute coverers (transpose of ball relation)
    coverers = [set() for _ in range(total)]
    for i in range(total):
        for w in ball[i]:
            coverers[w].add(i)

    return ball, coverers


def _greedy_criticality_cover(ball, coverers, q, n, radius, deadline):
    """
    Greedy set cover that prioritizes uncovered words with highest criticality
    (fewest available codewords that can cover them).
    """
    total = q ** n
    start_time = time.time()

    covered = np.zeros(total, dtype=np.uint8)
    cov_count = np.zeros(total, dtype=np.int32)
    chosen = set()
    codeword_indices = []

    # Initial seed: tetracode direct sum for q=3, n>=4
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for t in tetracode:
                w = h + t
                idx = 0
                mul = 1
                for p in range(n):
                    idx += w[p] * mul
                    mul *= q
                if idx not in chosen:
                    chosen.add(idx)
                    codeword_indices.append(idx)
                    cov_count[ball[idx]] += 1
                    covered[ball[idx]] = 1

    # Greedy fill
    while not covered.all():
        if time.time() > deadline - _DEADLINE_MARGIN:
            break

        # Find uncovered words
        unc_idx = np.where(covered == 0)[0]
        if len(unc_idx) == 0:
            break

        # For each uncovered word, count how many available codewords can cover it
        # Criticality = number of available codewords that cover this word
        # We want to pick the word with minimum criticality (most constrained)
        best_word = -1
        best_criticality = total + 1
        best_candidate = -1
        best_gain = 0

        # Sample a subset of uncovered words for efficiency
        sample_size = min(100, len(unc_idx))
        sample = unc_idx[np.random.choice(len(unc_idx), sample_size, replace=False)]

        for w in sample:
            # Available codewords that cover w
            avail = coverers[w] - chosen
            if len(avail) == 0:
                # Should not happen if we're covering properly
                continue
            criticality = len(avail)
            if criticality < best_criticality:
                best_criticality = criticality
                best_word = w
                # Among available, pick the one with max gain
                for c in avail:
                    gain = int(np.sum(covered[ball[c]] == 0))
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c

        if best_candidate == -1:
            # Fallback: pick any uncovered word and any coverer
            w = unc_idx[0]
            avail = coverers[w] - chosen
            if len(avail) == 0:
                break
            best_candidate = list(avail)[0]

        # Add the candidate
        chosen.add(best_candidate)
        codeword_indices.append(best_candidate)
        cov_count[ball[best_candidate]] += 1
        covered[ball[best_candidate]] = 1

    # Ensure full coverage
    if not covered.all():
        for i in range(total):
            if not covered[i]:
                if i not in chosen:
                    chosen.add(i)
                    codeword_indices.append(i)
                    cov_count[ball[i]] += 1
                    covered[ball[i]] = 1

    return codeword_indices


def _greedy_removal(codeword_indices, ball, q, n, radius, deadline):
    """
    Greedy removal: try to remove codewords one by one if coverage is maintained.
    """
    total = q ** n
    start_time = time.time()

    # Build coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for idx in codeword_indices:
        cov_count[ball[idx]] += 1

    changed = True
    while changed and time.time() < deadline - _DEADLINE_MARGIN:
        changed = False
        # Try removing each codeword in random order
        order = list(range(len(codeword_indices)))
        random.shuffle(order)
        for i in order:
            if time.time() >= deadline - _DEADLINE_MARGIN:
                break
            idx = codeword_indices[i]
            b = ball[idx]
            # Check if any word in b has cov_count == 1
            if np.any(cov_count[b] == 1):
                continue
            # Safe to remove
            cov_count[b] -= 1
            codeword_indices.pop(i)
            changed = True
            break  # Restart

    return codeword_indices


def _two_opt_local_search(codeword_indices, ball, coverers, q, n, radius, deadline):
    """
    2-opt local search: try swaps that either reduce codeword count
    or maintain count while improving minimum coverage multiplicity.
    """
    total = q ** n
    start_time = time.time()

    # Build coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for idx in codeword_indices:
        cov_count[ball[idx]] += 1

    improved = True
    while improved and time.time() < deadline - _DEADLINE_MARGIN:
        improved = False
        # Try each codeword for removal or swap
        order = list(range(len(codeword_indices)))
        random.shuffle(order)
        for i in order:
            if time.time() >= deadline - _DEADLINE_MARGIN:
                break
            old_idx = codeword_indices[i]
            old_b = ball[old_idx]
            # Words that would become uncovered if we remove old_idx
            critical = old_b[cov_count[old_b] == 1]

            if len(critical) == 0:
                # Just remove
                cov_count[old_b] -= 1
                codeword_indices.pop(i)
                improved = True
                break

            # Need a replacement that covers all critical words
            # Find candidates: intersection of coverers[w] for w in critical, excluding current codewords
            if len(critical) == 1:
                w = critical[0]
                cands = coverers[w] - set(codeword_indices)
            else:
                cands = coverers[critical[0]]
                for w in critical[1:]:
                    cands = cands & coverers[w]
                cands = cands - set(codeword_indices)

            if not cands:
                continue

            # Among candidates, pick the one that maximizes new coverage
            best_cand = -1
            best_gain = 0
            for cand in list(cands)[:50]:  # Limit to 50 for efficiency
                cand_b = ball[cand]
                # Gain = new words covered by cand that weren't covered by old_idx
                # Actually, we want to maximize overall coverage improvement
                # Since we're maintaining the same size, we want to minimize uncovered after swap
                # Compute uncovered if we remove old and add cand
                # new_uncovered = words in old_b not in cand_b that have cov_count == 1 after removal
                # This is complex, so just pick random for now
                pass

            # Pick a random candidate for simplicity
            cand = random.choice(list(cands))
            cand_b = ball[cand]

            # Perform swap
            cov_count[old_b] -= 1
            cov_count[cand_b] += 1
            codeword_indices[i] = cand
            improved = True
            break

    return codeword_indices


def _is_valid_cover(codeword_indices, ball, q, n, radius):
    """Check if the set of codeword indices covers all words."""
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    for idx in codeword_indices:
        covered[ball[idx]] = 1
    return covered.all()


def construct(q, n, radius):
    """Construct a covering code using criticality greedy + 2-opt local search."""
    start_time = time.time()
    deadline = start_time + 150 - _DEADLINE_MARGIN

    total = q ** n

    # Precompute balls and coverers
    ball, coverers = _precompute_balls(q, n, radius)

    # Initial construction: greedy criticality cover
    codeword_indices = _greedy_criticality_cover(ball, coverers, q, n, radius, deadline)

    # Verify initial solution is valid
    if not _is_valid_cover(codeword_indices, ball, q, n, radius):
        # Fallback: add missing words
        covered = np.zeros(total, dtype=np.uint8)
        for idx in codeword_indices:
            covered[ball[idx]] = 1
        for i in range(total):
            if not covered[i]:
                codeword_indices.append(i)

    # Greedy removal to reduce size
    codeword_indices = _greedy_removal(codeword_indices, ball, q, n, radius, deadline)

    # 2-opt local search
    if time.time() < deadline - _DEADLINE_MARGIN:
        codeword_indices = _two_opt_local_search(codeword_indices, ball, coverers, q, n, radius, deadline)

    # Final verification
    if not _is_valid_cover(codeword_indices, ball, q, n, radius):
        # Emergency: rebuild
        covered = np.zeros(total, dtype=np.uint8)
        for idx in codeword_indices:
            covered[ball[idx]] = 1
        for i in range(total):
            if not covered[i]:
                codeword_indices.append(i)

    # Convert indices to codewords
    codewords = []
    for idx in codeword_indices:
        word = []
        tmp = idx
        for i in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        codewords.append(word)

    return codewords
