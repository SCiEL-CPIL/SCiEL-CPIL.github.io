"""Rewritten program: greedy set-cover with NumPy vectorization and deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _all_words(q, n):
    """Return list of all words of length n over 0..q-1."""
    return [list(t) for t in itertools.product(range(q), repeat=n)]


def _ball_mask(word, q, n, radius, word_to_idx):
    """Return set of words within Hamming distance <= radius of word."""
    # For small radius and n, enumerate all words within distance radius
    # radius=1: word itself + all single-position changes
    covered = set()
    covered.add(tuple(word))
    if radius >= 1:
        for pos in range(n):
            orig = word[pos]
            for v in range(q):
                if v == orig:
                    continue
                w = word[:]
                w[pos] = v
                covered.add(tuple(w))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    words = _all_words(q, n)
    N = len(words)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}

    # Precompute ball membership: balls[i] is a boolean array of length N
    # True if words[j] is within distance <= radius of words[i]
    # For radius=1 and q=3, n=6: each ball has 1 + n*(q-1) = 1+6*2 = 13 words
    # Total memory: N * N bools = 729*729 ~ 0.5 MB, fine
    balls = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(words):
        covered = _ball_mask(w, q, n, radius, word_to_idx)
        for t in covered:
            j = word_to_idx[t]
            balls[i, j] = True

    # Greedy set cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    # remaining[i] = number of currently uncovered words covered by candidate i
    remaining = balls.sum(axis=1).astype(np.int32)  # balls[i] & uncovered -> but we'll update

    # Actually, we need remaining[i] = sum of balls[i, j] for j where uncovered[j]
    # Let's recompute properly: remaining[i] = balls[i] dot uncovered
    remaining = (balls & uncovered[None, :]).sum(axis=1).astype(np.int32)

    while uncovered.any():
        # Check deadline
        if time.time() > deadline:
            break

        # Pick candidate with max remaining
        idx = int(np.argmax(remaining))
        if remaining[idx] <= 0:
            # No progress possible with this approach; break
            break

        selected.append(words[idx])
        # Update uncovered: remove words covered by idx
        newly_covered = balls[idx] & uncovered
        uncovered &= ~newly_covered
        # Update remaining for all candidates: subtract newly_covered from their counts
        # remaining[i] = sum(balls[i, j] for j in uncovered)
        # We can update: remaining -= newly_covered (since newly_covered are no longer uncovered)
        # But only for candidates that cover those words
        remaining -= newly_covered  # vectorized: for each i, subtract number of newly_covered in balls[i]
        # Wait, that's not right. newly_covered is a bool vector of length N.
        # remaining[i] should decrease by the number of j where newly_covered[j] and balls[i,j]
        # That's balls[i] dot newly_covered
        # Let me redo:
        # Actually the line above is wrong. Let me fix:
        # remaining[i] = sum_{j: uncovered[j]} balls[i,j]
        # After update, uncovered' = uncovered & ~newly_covered
        # remaining'[i] = sum_{j: uncovered'[j]} balls[i,j] = remaining[i] - sum_{j: newly_covered[j]} balls[i,j]
        # So remaining -= balls @ newly_covered
        # But I already did remaining -= newly_covered which is wrong.
        # Let me recompute properly each iteration for correctness, N is small.
        pass

    # The above greedy loop has a bug in the update. Let me rewrite cleanly.
    # Given N=729, recomputing remaining each iteration is O(N^2) per step, 
    # and we do at most ~73 steps, so ~73 * 729^2 ~ 39M ops, fine in NumPy.

    # Restart greedy cleanly
    uncovered = np.ones(N, dtype=bool)
    selected = []
    while uncovered.any():
        if time.time() > deadline:
            break
        # remaining[i] = number of uncovered words covered by candidate i
        remaining = (balls & uncovered[None, :]).sum(axis=1)
        idx = int(np.argmax(remaining))
        if remaining[idx] <= 0:
            break
        selected.append(words[idx])
        uncovered &= ~balls[idx]

    # Local improvement: try to remove a codeword if coverage still holds
    # Check if removing codeword k still covers all words
    # coverage mask: for each word j, is it covered by at least one selected codeword?
    # Build coverage: cov[j] = any(selected[i] covers j)
    # For efficiency: cov = OR of balls[selected_i] over selected
    # Try removing each: if cov_without_i is still all True, remove it.

    # Compute current coverage
    cov = np.zeros(N, dtype=bool)
    for w in selected:
        i = word_to_idx[tuple(w)]
        cov |= balls[i]

    # Try removing each codeword
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for w in selected:
            i = word_to_idx[tuple(w)]
            # Check if removing w still covers everything
            # cov_without = cov with balls[i] removed, but need to recompute
            # cov_without[j] = cov[j] and not (balls[i,j] and no other selected covers j)
            # Simpler: recompute cov without this codeword
            cov_test = np.zeros(N, dtype=bool)
            for w2 in selected:
                if w2 == w:
                    continue
                i2 = word_to_idx[tuple(w2)]
                cov_test |= balls[i2]
            if cov_test.all():
                # Can remove w
                changed = True
                break
            else:
                new_selected.append(w)
        if changed:
            selected = new_selected
            # Recompute cov
            cov = np.zeros(N, dtype=bool)
            for w in selected:
                i = word_to_idx[tuple(w)]
                cov |= balls[i]

    # Verify coverage
    cov = np.zeros(N, dtype=bool)
    for w in selected:
        i = word_to_idx[tuple(w)]
        cov |= balls[i]
    if not cov.all():
        # Fallback to seed construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return tetracode[:3 ** (n - 4)] if n < 4 else tetracode

    return selected
# EVOLVE-BLOCK-END
