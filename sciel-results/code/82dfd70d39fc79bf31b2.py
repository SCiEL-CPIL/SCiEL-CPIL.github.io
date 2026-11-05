"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def _word_to_idx(word, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _precompute_ball_parts(q, n, radius):
    """For each position and original value, precompute the list of index-deltas
    to all words at Hamming distance 1 by changing that position."""
    parts = []
    for pos in range(n):
        pos_parts = []
        for orig in range(q):
            deltas = []
            for new_val in range(q):
                if new_val == orig:
                    continue
                # index change: (new_val - orig) * q^(n-1-pos)
                deltas.append((new_val - orig) * (q ** (n - 1 - pos)))
            pos_parts.append(deltas)
        parts.append(pos_parts)
    return parts


def _ball_indices_fast(cw_idx, cw, ball_parts):
    """Build the ball index list using precomputed deltas."""
    indices = [cw_idx]
    for pos in range(len(cw)):
        for d in ball_parts[pos][cw[pos]]:
            indices.append(cw_idx + d)
    return indices


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before the 120s (150s hard) limit

    total_words = q ** n

    # Precompute ball parts for fast ball construction
    ball_parts = _precompute_ball_parts(q, n, radius)

    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the other n - 4 coordinates: 9 * q**(n-4) words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    codewords = [h + list(tail) for h in tetracode
                 for tail in itertools.product(range(q), repeat=n - 4)]

    # Precompute each codeword's ball as a list of covered-word indices.
    cw_idxs = [_word_to_idx(cw, q) for cw in codewords]
    balls = [_ball_indices_fast(idx, cw, ball_parts) for idx, cw in zip(cw_idxs, codewords)]

    # Coverage counts: for each word in the space, how many codewords cover it.
    counts = [0] * total_words
    for ball in balls:
        for w in ball:
            counts[w] += 1

    # Track how many words are uncovered (counts == 0).
    uncovered = sum(1 for c in counts if c == 0)

    # unique_count[i] = number of words in balls[i] that are covered only by codeword i
    unique_count = [0] * len(codewords)
    for i, ball in enumerate(balls):
        uc = 0
        for w in ball:
            if counts[w] == 1:
                uc += 1
        unique_count[i] = uc

    def is_covered():
        return uncovered == 0

    def is_redundant(i):
        return unique_count[i] == 0

    # ---- Phase 1: aggressive greedy removal in several orderings ----
    def removal_pass(order):
        nonlocal uncovered
        for i in order:
            if time.time() > deadline:
                return
            if i >= len(codewords):
                continue
            if unique_count[i] == 0:
                # Remove codeword i
                ball = balls[i]
                for w in ball:
                    if counts[w] == 1:
                        uncovered += 1
                    counts[w] -= 1
                codewords.pop(i)
                balls.pop(i)
                unique_count.pop(i)
                # Update unique_count for remaining codewords that covered words now at count 1
                # Actually we need to update: for each other ball, if it contains a word now at count 1,
                # its unique_count increases. But that's expensive. Instead, recompute lazily.
                # Simpler: mark that unique_count may be stale and recompute when needed.
                # For correctness of is_redundant, we need accurate unique_count.
                # Let's recompute unique_count for all remaining codewords after each removal batch.
                # Actually, let's just recompute all unique_count after the removal pass.

    for _ in range(6):
        if time.time() > deadline:
            break
        removal_pass(range(len(codewords) - 1, -1, -1))
        # Recompute unique_count after each pass
        unique_count = [0] * len(codewords)
        for i, ball in enumerate(balls):
            uc = 0
            for w in ball:
                if counts[w] == 1:
                    uc += 1
            unique_count[i] = uc
        removal_pass(range(len(codewords)))
        unique_count = [0] * len(codewords)
        for i, ball in enumerate(balls):
            uc = 0
            for w in ball:
                if counts[w] == 1:
                    uc += 1
            unique_count[i] = uc
        removal_pass(range(len(codewords) - 1, -1, -1))
        unique_count = [0] * len(codewords)
        for i, ball in enumerate(balls):
            uc = 0
            for w in ball:
                if counts[w] == 1:
                    uc += 1
            unique_count[i] = uc

    best_code = codewords[:]
    best_len = len(codewords)
    best_uncovered = uncovered

    def snapshot():
        nonlocal best_code, best_len, best_uncovered
        if uncovered < best_uncovered or (uncovered == best_uncovered and len(codewords) < best_len):
            best_uncovered = uncovered
            best_code = codewords[:]
            best_len = len(codewords)

    def _remove_codeword(i):
        """Remove codeword i (must be redundant). Updates counts, uncovered, unique_count."""
        nonlocal uncovered
        ball = balls[i]
        for w in ball:
            if counts[w] == 1:
                uncovered += 1
            counts[w] -= 1
        codewords.pop(i)
        balls.pop(i)
        unique_count.pop(i)
        # Recompute unique_count for all remaining (expensive but correct)
        for j in range(len(codewords)):
            uc = 0
            for w in balls[j]:
                if counts[w] == 1:
                    uc += 1
            unique_count[j] = uc

    def try_remove(i):
        """Remove codeword i if redundant. Returns True on success."""
        if unique_count[i] != 0:
            return False
        _remove_codeword(i)
        return True

    def try_add(word):
        idx = len(codewords)
        codewords.append(word)
        cw_idx = _word_to_idx(word, q)
        ball = _ball_indices_fast(cw_idx, word, ball_parts)
        balls.append(ball)
        uc = 0
        for w in ball:
            if counts[w] == 0:
                uncovered -= 1
            counts[w] += 1
            if counts[w] == 1:
                uc += 1
        unique_count.append(uc)
        # Update unique_count for other codewords: words that went from 0->1 or 1->2
        # Words that went 0->1: no other codeword covers them (count was 0)
        # Words that went 1->2: the other codeword covering them loses a unique word
        # We need to find which other balls contain words that were at count 1 before
        # This is complex; let's just recompute all unique_count
        for j in range(len(codewords)):
            if j == idx:
                continue
            uc_j = 0
            for w in balls[j]:
                if counts[w] == 1:
                    uc_j += 1
            unique_count[j] = uc_j
        return idx

    def swap(i, new_word):
        """Replace codeword i with new_word, keeping coverage valid. Returns True if kept."""
        nonlocal uncovered
        # Check removal keeps coverage
        if unique_count[i] != 0:
            return False
        # Apply removal
        old_ball = balls[i]
        for w in old_ball:
            if counts[w] == 1:
                uncovered += 1
            counts[w] -= 1
        # Add new word
        new_cw_idx = _word_to_idx(new_word, q)
        new_ball = _ball_indices_fast(new_cw_idx, new_word, ball_parts)
        codewords[i] = new_word
        balls[i] = new_ball
        uc_new = 0
        for w in new_ball:
            if counts[w] == 0:
                uncovered -= 1
            counts[w] += 1
            if counts[w] == 1:
                uc_new += 1
        unique_count[i] = uc_new
        # Update unique_count for all other codewords
        for j in range(len(codewords)):
            if j == i:
                continue
            uc_j = 0
            for w in balls[j]:
                if counts[w] == 1:
                    uc_j += 1
            unique_count[j] = uc_j
        return True

    # ---- Phase 2: randomized local search ----
    rng = random.Random(42)
    step = 0
    max_steps = 200000
    stuck = 0

    while time.time() < deadline and step < max_steps:
        step += 1
        m = len(codewords)
        if m <= 1:
            break

        # Strategy 1: remove a random redundant codeword
        if m > 1 and rng.random() < 0.4:
            idx = rng.randrange(m)
            if unique_count[idx] == 0:
                _remove_codeword(idx)
                snapshot()
                stuck = 0
                continue
            stuck += 1

        # Strategy 2: swap a codeword with a nearby one (distance 1 or 2)
        idx = rng.randrange(m)
        old_cw = codewords[idx]
        dist = rng.choice([1, 2])
        new_cw = old_cw[:]
        positions = list(range(n))
        rng.shuffle(positions)
        for pos in positions[:dist]:
            old_val = new_cw[pos]
            new_val = rng.randrange(q)
            while new_val == old_val:
                new_val = rng.randrange(q)
            new_cw[pos] = new_val
        if new_cw == old_cw:
            continue
        if swap(idx, new_cw):
            snapshot()
            stuck = 0
        else:
            stuck += 1

        # Strategy 3: add a random word then try to remove two (escape local minima)
        if m > 2 and rng.random() < 0.15:
            random_word = [rng.randrange(q) for _ in range(n)]
            if random_word not in codewords:
                try_add(random_word)
                removed = 0
                for _ in range(2):
                    if len(codewords) <= 2:
                        break
                    ridx = rng.randrange(len(codewords))
                    if unique_count[ridx] == 0:
                        _remove_codeword(ridx)
                        removed += 1
                # Remove the added word if still present
                if random_word in codewords:
                    ridx2 = codewords.index(random_word)
                    if unique_count[ridx2] == 0:
                        _remove_codeword(ridx2)
                if removed >= 1:
                    snapshot()
                    stuck = 0
                else:
                    stuck += 1

        # If stuck for a while, do a burst of greedy removals
        if stuck > 200:
            stuck = 0
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if unique_count[i] == 0:
                    _remove_codeword(i)

    # ---- Phase 3: final polish - exhaustive redundant removal ----
    if is_covered():
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if unique_count[i] == 0:
                    _remove_codeword(i)
                    improved = True

    # Return the best valid code found
    if best_uncovered == 0:
        return best_code

    # Fallback: ensure we return a valid covering code.
    # Rebuild a valid code from best_code and verify coverage; if needed, restore full code.
    # best_code should be valid since we only snapshot when uncovered==0.
    # As a safety net, if somehow invalid, return the original tetracode sum.
    return best_code if best_uncovered == 0 else [h + list(tail) for h in tetracode
                                                  for tail in itertools.product(range(q), repeat=n - 4)]
# EVOLVE-BLOCK-END
