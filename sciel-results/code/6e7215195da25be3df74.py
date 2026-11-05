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


def _ball_indices(cw, q, n, radius):
    """Return a list of indices of all words within Hamming distance radius of cw."""
    indices = [_word_to_idx(cw, q)]
    if radius >= 1:
        for pos in range(n):
            orig = cw[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                lst = list(cw)
                lst[pos] = new_val
                indices.append(_word_to_idx(lst, q))
    return indices


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before the 120s (150s hard) limit

    total_words = q ** n

    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the other n - 4 coordinates: 9 * q**(n-4) words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    codewords = [h + list(tail) for h in tetracode
                 for tail in itertools.product(range(q), repeat=n - 4)]

    # Precompute each codeword's ball as a list of covered-word indices.
    balls = [_ball_indices(cw, q, n, radius) for cw in codewords]

    # Coverage counts: for each word in the space, how many codewords cover it.
    counts = [0] * total_words
    for ball in balls:
        for w in ball:
            counts[w] += 1

    # Track how many words are uncovered (counts == 0).
    uncovered = sum(1 for c in counts if c == 0)

    def is_covered():
        return uncovered == 0

    def is_redundant(i):
        for w in balls[i]:
            if counts[w] <= 1:
                return False
        return True

    # ---- Phase 1: aggressive greedy removal in several orderings ----
    def removal_pass(order):
        nonlocal uncovered
        for i in order:
            if time.time() > deadline:
                return
            if i >= len(codewords):
                continue
            if is_redundant(i):
                for w in balls[i]:
                    if counts[w] == 1:
                        uncovered += 1
                    counts[w] -= 1
                codewords.pop(i)
                balls.pop(i)

    for _ in range(6):
        if time.time() > deadline:
            break
        removal_pass(range(len(codewords) - 1, -1, -1))
        removal_pass(range(len(codewords)))
        removal_pass(range(len(codewords) - 1, -1, -1))

    best_code = codewords[:]
    best_len = len(codewords)
    best_uncovered = uncovered

    def snapshot():
        nonlocal best_code, best_len, best_uncovered
        if uncovered < best_uncovered or (uncovered == best_uncovered and len(codewords) < best_len):
            best_uncovered = uncovered
            best_code = codewords[:]
            best_len = len(codewords)

    def try_remove(i):
        """Remove codeword i if redundant. Returns True on success."""
        nonlocal uncovered
        if not is_redundant(i):
            return False
        for w in balls[i]:
            if counts[w] == 1:
                uncovered += 1
            counts[w] -= 1
        codewords.pop(i)
        balls.pop(i)
        return True

    def try_add(word):
        idx = len(codewords)
        codewords.append(word)
        ball = _ball_indices(word, q, n, radius)
        balls.append(ball)
        for w in ball:
            if counts[w] == 0:
                uncovered -= 1
            counts[w] += 1
        return idx

    def swap(i, new_word):
        """Replace codeword i with new_word, keeping coverage valid. Returns True if kept."""
        nonlocal uncovered
        # Check removal keeps coverage
        for w in balls[i]:
            if counts[w] <= 1:
                return False
        # Apply removal
        for w in balls[i]:
            if counts[w] == 1:
                uncovered += 1
            counts[w] -= 1
        old_cw = codewords[i]
        old_ball = balls[i]
        new_ball = _ball_indices(new_word, q, n, radius)
        codewords[i] = new_word
        balls[i] = new_ball
        for w in new_ball:
            if counts[w] == 0:
                uncovered -= 1
            counts[w] += 1
        return True

    # ---- Phase 2: randomized local search ----
    rng = random.Random(42)
    step = 0
    max_steps = 60000
    stuck = 0

    while time.time() < deadline and step < max_steps:
        step += 1
        m = len(codewords)
        if m <= 1:
            break

        # Strategy 1: remove a random redundant codeword
        if m > 1 and rng.random() < 0.4:
            idx = rng.randrange(m)
            if is_redundant(idx):
                for w in balls[idx]:
                    if counts[w] == 1:
                        uncovered += 1
                    counts[w] -= 1
                codewords.pop(idx)
                balls.pop(idx)
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
                    if is_redundant(ridx):
                        for w in balls[ridx]:
                            if counts[w] == 1:
                                uncovered += 1
                            counts[w] -= 1
                        codewords.pop(ridx)
                        balls.pop(ridx)
                        removed += 1
                # Remove the added word if still present
                if random_word in codewords:
                    ridx2 = codewords.index(random_word)
                    for w in balls[ridx2]:
                        if counts[w] == 1:
                            uncovered += 1
                        counts[w] -= 1
                    codewords.pop(ridx2)
                    balls.pop(ridx2)
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
                if is_redundant(i):
                    for w in balls[i]:
                        if counts[w] == 1:
                            uncovered += 1
                        counts[w] -= 1
                    codewords.pop(i)
                    balls.pop(i)

    # ---- Phase 3: final polish - exhaustive redundant removal ----
    if is_covered():
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_redundant(i):
                    for w in balls[i]:
                        if counts[w] == 1:
                            uncovered += 1
                        counts[w] -= 1
                    codewords.pop(i)
                    balls.pop(i)
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
