"""New design: seed with tetracode x all-tails, then prune (remove redundant) + repair (greedy cover) loop."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_TIME_LIMIT = 115.0  # safety margin under the 120s wall-clock


def _balls_all(q, n, radius):
    """Precompute, for every word in {0..q-1}^n, the list of words within Hamming distance radius.
    Returns a list of length q**n where each entry is a list of ints (encoded word ids)."""
    total = q ** n
    # Precompute neighbors for each coordinate position
    # For a given word (as a tuple of digits) and a position i, the words at distance 1 in that
    # coordinate are obtained by changing digit i to any other value.
    # We'll build neighbors by iterating over all words and all positions.
    # For radius=1: ball(word) = {word} U {word with one digit changed}
    # For general radius we'd need combinations; we keep it general but the task uses radius=1.
    if radius == 1:
        balls = []
        for idx in range(total):
            # decode
            digits = []
            x = idx
            for _ in range(n):
                digits.append(x % q)
                x //= q
            digits = digits[::-1]  # most significant first
            ball = [idx]
            for i in range(n):
                orig = digits[i]
                for v in range(q):
                    if v == orig:
                        continue
                    # re-encode with digit i = v
                    new_digits = digits[:]
                    new_digits[i] = v
                    new_idx = 0
                    for d in new_digits:
                        new_idx = new_idx * q + d
                    ball.append(new_idx)
            balls.append(ball)
        return balls
    # General radius fallback (slower)
    balls = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits = digits[::-1]
        ball = set()
        # iterate over all subsets of positions up to radius
        from itertools import combinations, product
        for r in range(radius + 1):
            for pos in combinations(range(n), r):
                # for each position, choose a new digit (including original)
                new_digits = digits[:]
                for p in pos:
                    new_digits[p] = -1  # placeholder
                # product over changed positions
                ranges = []
                for p in pos:
                    ranges.append(range(q))
                if not ranges:
                    new_idx = idx
                    ball.add(new_idx)
                    continue
                for combo in product(*ranges):
                    nd = digits[:]
                    for p, v in zip(pos, combo):
                        nd[p] = v
                    new_idx = 0
                    for d in nd:
                        new_idx = new_idx * q + d
                    ball.add(new_idx)
        balls.append(list(ball))
    return balls


def construct(q, n, radius):
    start = time.time()
    total = q ** n
    balls = _balls_all(q, n, radius)
    ball_size = len(balls[0])

    # Seed: tetracode x all-tails (9 * 3**(n-4) words) for q=3, n=6 -> 81 words
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_words = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                seed_words.append(h + list(tail))
    else:
        # Fallback: all-zero seed
        seed_words = [[0] * n]

    # Encode seed words
    def encode(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    selected = set(encode(w) for w in seed_words)

    # counts[word] = number of selected codewords whose ball covers word
    counts = np.zeros(total, dtype=np.int32)
    for cw in selected:
        counts[balls[cw]] += 1

    def add(cw):
        selected.add(cw)
        counts[balls[cw]] += 1

    def remove(cw):
        selected.remove(cw)
        counts[balls[cw]] -= 1

    # Greedy pick: choose the codeword that covers the most currently-uncovered words
    # We track uncovered = (counts == 0)
    def greedy_pick():
        # For each candidate cw (not yet selected), gain = number of uncovered words in its ball
        # We'll vectorize: build a matrix? That's 729 * 19 ~ 13.8k, fine.
        # Actually we can compute gain per candidate by summing over its ball where counts==0.
        best_gain = -1
        best_cw = None
        # Iterate candidates; to speed up, precompute ball arrays
        ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
        for cw in range(total):
            if cw in selected:
                continue
            b = ball_arrays[cw]
            gain = np.count_nonzero(counts[b] == 0)
            if gain > best_gain:
                best_gain = gain
                best_cw = cw
                if gain == ball_size:
                    break
        return best_cw

    def prune_pass():
        """Remove any selected codeword whose ball is fully covered by the others.
        Returns number removed."""
        removed = 0
        # Iterate in a fixed order; re-check after each removal
        # To be efficient, we can do a sweep: for each selected cw, temporarily check if removing it
        # would leave any word with count 0.
        # counts is current. For cw, the words that would become uncovered are those where counts==1
        # and cw is the only cover. So check: are there any words in ball(cw) with counts==1?
        # If not, cw is redundant.
        ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
        # We'll iterate over a snapshot of selected
        snapshot = list(selected)
        for cw in snapshot:
            if cw not in selected:
                continue
            b = ball_arrays[cw]
            # If any word in b has count==1, removing cw would uncover it -> not redundant
            if np.any(counts[b] == 1):
                continue
            # Redundant
            remove(cw)
            removed += 1
        return removed

    def repair_pass():
        """Add codewords until all words are covered. Returns number added."""
        added = 0
        while True:
            uncovered = np.count_nonzero(counts == 0)
            if uncovered == 0:
                break
            cw = greedy_pick()
            if cw is None:
                break
            add(cw)
            added += 1
            if time.time() - start > _TIME_LIMIT:
                break
        return added

    # Initial: ensure coverage
    repair_pass()

    # Alternating prune/repair until no improvement or time runs out
    best_size = len(selected)
    no_improve = 0
    while time.time() - start < _TIME_LIMIT:
        before = len(selected)
        prune_pass()
        repair_pass()
        after = len(selected)
        if after < best_size:
            best_size = after
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                break
        if after == before:
            break

    # Decode selected to list of words
    result = []
    for cw in selected:
        word = []
        x = cw
        for _ in range(n):
            word.append(x % q)
            x //= q
        word.reverse()
        result.append(word)
    return result
# EVOLVE-BLOCK-END
