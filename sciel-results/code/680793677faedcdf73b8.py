"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

# Precompute the 729 words of {0,1,2}^6 as integer indices 0..728.
_Q = 3
_N = 6
_TOTAL = _Q ** _N  # 729
_POS = _N


def _word_to_idx(word):
    v = 0
    for a in word:
        v = v * _Q + a
    return v


def _idx_to_word(idx):
    w = [0] * _N
    for i in range(_N - 1, -1, -1):
        w[i] = idx % _Q
        idx //= _Q
    return w


def _hamming(a, b):
    return sum(1 for i in range(_N) if a[i] != b[i])


# Build the "ball of radius 1" (20 words) for each of the 729 words, as index lists.
# For a word w, its ball is {w} union {w with position i changed to v, for each i, v != w[i]}.
_BALLS = []
for _i in range(_TOTAL):
    _w = _idx_to_word(_i)
    _ball = [_i]
    for _p in range(_N):
        _orig = _w[_p]
        for _v in range(_Q):
            if _v == _orig:
                continue
            _w2 = _w[:]
            _w2[_p] = _v
            _ball.append(_word_to_idx(_w2))
    _BALLS.append(tuple(_ball))


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_code():
    # Direct sum: tetracode (9 words length 4) x all words on other 2 coords => 81 words.
    tet = _tetracode()
    code = []
    for h in tet:
        for tail in itertools.product(range(_Q), repeat=_N - 4):
            code.append(h + list(tail))
    return code


def construct(q, n, radius):
    # The contract's fixed task is q=3, n=6, radius=1. General fallback for other params.
    if (q, n, radius) != (_Q, _N, 1):
        # Generic greedy covering code (fallback).
        start = time.time()
        deadline = start + 110.0
        all_words = list(itertools.product(range(q), repeat=n))
        remaining = set(all_words)
        code = []
        while remaining:
            if time.time() > deadline:
                break
            best = None
            best_cover = -1
            for w in all_words:
                c = sum(1 for x in remaining if sum(a != b for a, b in zip(w, x)) <= radius)
                if c > best_cover:
                    best_cover = c
                    best = w
                if time.time() > deadline:
                    break
            if best is None:
                break
            code.append(list(best))
            for x in list(remaining):
                if sum(a != b for a, b in zip(best, x)) <= radius:
                    remaining.discard(x)
        return code

    start = time.time()
    deadline = start + 110.0  # safety margin before the 150s kill

    code = _initial_code()  # 81 words, a valid covering code
    best = code[:]
    best_size = len(code)

    # Track coverage: for each of the 729 words, whether it is within distance 1 of some codeword.
    covered = [False] * _TOTAL
    for w in code:
        for j in _BALLS[_word_to_idx(w)]:
            covered[j] = True

    # Codewords as a list of index tuples; keep a set for dedup.
    cw_idx = [_word_to_idx(w) for w in code]
    cw_set = set(cw_idx)

    def _remove(idx):
        nonlocal covered
        # Recompute coverage contribution of this codeword by checking others.
        # Efficient approach: decrement a per-word count of how many codewords cover it.
        pass

    # Use a coverage counter for O(ball) removals.
    cov_count = [0] * _TOTAL
    for idx in cw_idx:
        for j in _BALLS[idx]:
            cov_count[j] += 1
    for j in range(_TOTAL):
        covered[j] = cov_count[j] > 0

    def _try_remove(idx):
        # Returns True if removing idx keeps a valid cover.
        for j in _BALLS[idx]:
            if cov_count[j] == 1:
                return False
        return True

    def _do_remove(idx):
        nonlocal covered
        for j in _BALLS[idx]:
            cov_count[j] -= 1
            if cov_count[j] == 0:
                covered[j] = False
        cw_idx.remove(idx)
        cw_set.discard(idx)

    # Greedy removal: repeatedly drop any codeword whose removal preserves coverage.
    # Iterate until no removable codeword remains (fixed point).
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(cw_idx):
            if _try_remove(idx):
                _do_remove(idx)
                changed = True
                if len(cw_idx) < best_size:
                    best_size = len(cw_idx)
                    best = [_idx_to_word(i) for i in cw_idx]

    # Local improvement: try to find a codeword to add that lets us drop more than one.
    # Candidate additions: uncovered-adjacent words are none (cover is valid), so we look
    # for words whose addition increases redundancy enough to enable a net removal.
    # We do a limited randomized search to find a better configuration.
    import random
    rng = random.Random(12345)

    # Precompute, for each word, the set of codewords that cover it (for quick analysis).
    # Instead, do a simple hill-climb: try adding a random word, then greedily remove.
    for _iter in range(3000):
        if time.time() > deadline - 15.0:
            break
        cand = rng.randrange(_TOTAL)
        if cand in cw_set:
            continue
        # Tentatively add cand.
        for j in _BALLS[cand]:
            cov_count[j] += 1
        cw_idx.append(cand)
        cw_set.add(cand)

        # Greedily remove while possible.
        changed = True
        while changed:
            changed = False
            for idx in list(cw_idx):
                if _try_remove(idx):
                    _do_remove(idx)
                    changed = True

        if len(cw_idx) < best_size:
            best_size = len(cw_idx)
            best = [_idx_to_word(i) for i in cw_idx]
        else:
            # Revert: remove cand.
            for j in _BALLS[cand]:
                cov_count[j] -= 1
            cw_idx.remove(cand)
            cw_set.discard(cand)

        if len(cw_idx) == best_size:
            # At or below best; continue searching but don't need to revert differently.
            pass

    # Final: ensure best is a valid cover (it is, by construction).
    return best
# EVOLVE-BLOCK-END
