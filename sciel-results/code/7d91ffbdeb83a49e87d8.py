"""Rewritten program: seed direct-sum code + greedy minimalization (set-cover pruning)."""
# EVOLVE-BLOCK-START
import itertools
import time


def _ball(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of word."""
    # radius is 1 for this task, but handle generally and cheaply.
    yield tuple(word)
    if radius >= 1:
        for i in range(n):
            orig = word[i]
            for v in range(q):
                if v != orig:
                    yield (word[:i] + (v,) + word[i + 1:])
    # (radius > 1 not needed for this task; kept minimal for clarity)


def _covers(ball_set, word, q, n, radius):
    """True if every word in ball_set is within radius of word."""
    for w in ball_set:
        d = 0
        for i in range(n):
            if w[i] != word[i]:
                d += 1
                if d > radius:
                    break
        if d > radius:
            return False
    return True


def construct(q, n, radius):
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock

    # Base construction: direct sum of ternary tetracode (perfect radius-1 code, len 4)
    # with all words on the remaining coordinates.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    code = [h + list(tail)
            for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]

    # Precompute each codeword's ball as a frozenset of tuples.
    balls = [frozenset(_ball(tuple(c), q, n, radius)) for c in code]

    # Greedy minimalization: remove a codeword if its ball is covered by the rest.
    active = list(range(len(code)))
    improved = True
    while improved:
        improved = False
        new_active = []
        for idx in active:
            if time.time() > deadline:
                # Fall back to the current active set (still valid: removals only).
                break
            ball = balls[idx]
            # Is every word in this ball covered by some OTHER active codeword?
            removable = True
            for w in ball:
                found = False
                for j in new_active:
                    if j == idx:
                        continue
                    if _covers(ball, None, q, n, 0) if False else _word_in_ball(w, code[j], q, n, radius):
                        found = True
                        break
                if not found:
                    # check remaining active that haven't been considered yet
                    for j in active:
                        if j == idx:
                            continue
                        if _word_in_ball(w, code[j], q, n, radius):
                            found = True
                            break
                    if not found:
                        removable = False
                        break
            if removable:
                improved = True
            else:
                new_active.append(idx)
        else:
            active = new_active
            continue
        # deadline hit mid-pass: keep the partially-built new_active merged with rest? 
        # Safer: keep the previous active (which is valid).
        break

    # Ensure we always return a valid code (the current active set is valid by construction).
    result = [code[i] for i in active]

    # Final safety: if deadline was hit mid-pass, active may be stale-but-valid; that's fine.
    return result


def _word_in_ball(w, center, q, n, radius):
    d = 0
    for i in range(n):
        if w[i] != center[i]:
            d += 1
            if d > radius:
                return False
    return True
# EVOLVE-BLOCK-END
