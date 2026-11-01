"""Greedy set-cover + local search for K_3(6,1)."""
import time
import itertools

# Precompute for the fixed task parameters (q=3, n=6, radius=1)
Q = 3
N = 6
R = 1
SPACE = Q ** N  # 729

def _idx(word):
    v = 0
    for x in word:
        v = v * Q + x
    return v

def _unidx(i):
    w = [0] * N
    for pos in range(N - 1, -1, -1):
        w[pos] = i % Q
        i //= Q
    return w

# Precompute all words and their balls (bit masks over the space)
_ALL_WORDS = [_unidx(i) for i in range(SPACE)]
_BALLS = []
for i in range(SPACE):
    w = _ALL_WORDS[i]
    mask = 1 << i
    for pos in range(N):
        orig = w[pos]
        for d in range(Q):
            if d == orig:
                continue
            w[pos] = d
            mask |= 1 << _idx(w)
        w[pos] = orig
    _BALLS.append(mask)

# Precompute the seed (tetracode direct sum) as a list of indices
def _seed_indices():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    idxs = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=N - 4):
            idxs.append(_idx(h + list(tail)))
    return idxs

def construct(q, n, radius):
    # Fallback: if parameters differ from the specialized precompute, use a generic greedy.
    if q != Q or n != N or radius != R:
        return _generic_construct(q, n, radius)

    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # Start from the seed (81 words). Convert to a set of indices.
    chosen = set(_seed_indices())

    # Compute covered mask
    covered = 0
    for c in chosen:
        covered |= _BALLS[c]

    # Greedy: while not fully covered, add the word that covers most uncovered words.
    full_mask = (1 << SPACE) - 1
    while covered != full_mask:
        if time.time() > deadline - 5.0:
            break
        uncovered = full_mask ^ covered
        best_c = -1
        best_gain = 0
        for c in range(SPACE):
            if c in chosen:
                continue
            gain = _BALLS[c] & uncovered
            gc = bin(gain).count('1')
            if gc > best_gain:
                best_gain = gc
                best_c = c
                if gc == best_gain and gc > 0:
                    pass
        if best_c < 0:
            break
        chosen.add(best_c)
        covered |= _BALLS[best_c]

    # Local search: try to reduce the number of codewords.
    # Strategy A: for each chosen word, try replacing it with a word that covers more
    #             uncovered-by-it words (i.e., the new word's ball minus old ball adds coverage
    #             that no other chosen word provides). This is a 1-for-1 exchange that keeps
    #             size same but improves coverage; not reducing size.
    # Strategy B: find a word w such that removing some chosen c and adding w keeps coverage
    #             and reduces size? That's not possible with 1-for-1. Instead:
    # Strategy C: find two chosen words c1, c2 and one word w such that {chosen - c1 - c2 + w}
    #             still covers everything -> reduces size by 1.
    # This is expensive. Let's do a simpler and effective approach:
    #   Repeatedly try: for each chosen c, compute what coverage is lost if we remove c
    #   (i.e., words only covered by c). If that lost set is nonempty, we must find a new
    #   word that covers all of it. Try to find a single word w (not in chosen) whose ball
    #   contains the lost set. If found, replace c with w (size unchanged, but may enable
    #   further reductions). This alone doesn't reduce size.
    #
    # Better: do a "dominance" reduction. If a chosen word c's ball is entirely contained
    # in the union of other chosen balls, then c is redundant -> remove it. Repeat.
    def _try_remove_redundant():
        nonlocal chosen, covered
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                return
            for c in list(chosen):
                others = 0
                for o in chosen:
                    if o == c:
                        continue
                    others |= _BALLS[o]
                if (_BALLS[c] & ~others) == 0:
                    chosen.discard(c)
                    covered = 0
                    for o in chosen:
                        covered |= _BALLS[o]
                    changed = True
                    break

    _try_remove_redundant()

    # Now do 2-for-1 exchanges: remove c1, c2 and add w, reducing size by 1.
    # For efficiency, iterate over pairs and check if there's a word covering the union of
    # their exclusive coverage.
    def _exclusive(c, others_set):
        # words covered by c but not by any other in others_set
        others = 0
        for o in others_set:
            if o == c:
                continue
            others |= _BALLS[o]
        return _BALLS[c] & ~others

    # Precompute exclusive coverage for each chosen word
    def _recompute_exclusive():
        excl = {}
        for c in chosen:
            others = 0
            for o in chosen:
                if o == c:
                    continue
                others |= _BALLS[o]
            excl[c] = _BALLS[c] & ~others
        return excl

    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 2.0:
            break
        excl = _recompute_exclusive()
        # Try to find c1, c2 such that excl[c1] | excl[c2] is covered by a single word w.
        c_list = list(chosen)
        m = len(c_list)
        found = False
        for i in range(m):
            if found:
                break
            for j in range(i + 1, m):
                if time.time() > deadline - 2.0:
                    found = True
                    break
                need = excl[c_list[i]] | excl[c_list[j]]
                if need == 0:
                    # both redundant, handled by _try_remove_redundant
                    continue
                # find a word w not in chosen whose ball contains need
                for w in range(SPACE):
                    if w in chosen:
                        continue
                    if (need & ~_BALLS[w]) == 0:
                        # replace
                        chosen.discard(c_list[i])
                        chosen.discard(c_list[j])
                        chosen.add(w)
                        improved = True
                        found = True
                        break
        # After any change, re-run redundancy removal
        _try_remove_redundant()

    # Convert back to list of words
    return [_unidx(c) for c in chosen]


def _generic_construct(q, n, radius):
    """Generic greedy set-cover for arbitrary q, n, radius. Slower but correct."""
    space = q ** n
    # Build all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]

    def hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Precompute balls as sets of indices
    balls = []
    for i, w in enumerate(words):
        ball = set()
        # all words within radius
        # brute force for small spaces
        for j, v in enumerate(words):
            if hamming(w, v) <= radius:
                ball.add(j)
        balls.append(ball)

    full = set(range(space))
    covered = set()
    chosen = []
    # Greedy
    while covered != full:
        best_i = -1
        best_gain = 0
        for i in range(space):
            if i in set(chosen):
                continue
            gain = len(balls[i] - covered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        chosen.append(best_i)
        covered |= balls[best_i]

    # Remove redundants
    changed = True
    while changed:
        changed = False
        for c in list(chosen):
            others = set()
            for o in chosen:
                if o != c:
                    others |= balls[o]
            if balls[c] <= others:
                chosen.remove(c)
                covered = set()
                for o in chosen:
                    covered |= balls[o]
                changed = True
                break

    return [words[c] for c in chosen]
