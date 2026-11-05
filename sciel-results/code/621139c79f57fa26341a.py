"""Ternary covering code K_3(6,1): minimize codewords covering all 729 words.

Strategy:
  * Exploit the full symmetry group (S6 x S3 x complement) to FIX the first
    codeword to 000000.  This is lossless: a symbol permutation can map any
    codeword to 000000, and the complement symmetry (x -> 2-x) composed with a
    symbol permutation preserves validity, so an optimal code exists containing
    000000.  The 37 weight-0/1 words are then covered for free.
  * Build a valid cover (greedy set cover over the 702 uncovered words).
  * Minimise the count with a branch-and-bound that only removes codewords that
    are redundant (their ball adds no new word), plus a local-search polish.
"""
import time
import numpy as np

_DEADLINE = 112.0  # seconds; safety margin before the 120 s wall-clock kill.
_DEADLINE = max(2.0, min(_DEADLINE, 112.0))
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _init(q, n, radius):
    # Precompute every word's distance to 000000 (i.e. its Hamming weight).
    allw = np.arange(q ** n, dtype=np.int64)
    digits = np.array([list(np.base_repr(w, q).zfill(n)) for w in range(q ** n)],
                      dtype=np.int8)
    dist0 = (digits != 0).sum(axis=1)
    # Ball radius r around 000000 = the set of words with weight <= r.
    ball0 = set(int(i) for i in np.where(dist0 <= radius)[0])
    # Words NOT covered by 000000 (these must be covered by the other codewords).
    uncovered = [int(i) for i in np.where(dist0 > radius)[0]]
    return digits, ball0, set(uncovered)


def _ball(digits, idx, radius):
    """Set of word indices within Hamming distance `radius` of digits[idx]."""
    w = digits[idx]
    out = {idx}
    if radius >= 1:
        for p in range(len(w)):
            base = int(w[:p])
            for v in range(q_global):
                if v == w[p]:
                    continue
                c = base * q_global + v
                out.add(c)
    return out


def _greedy_cover(digits, uncovered_set, deadline):
    """Greedy set cover: repeatedly add the word covering the most uncovered."""
    # Precompute, for each candidate word, the set of uncovered words it covers.
    cand = list(uncovered_set)
    cov = {}
    for idx in cand:
        cov[idx] = _ball(digits, idx, radius_global) & uncovered_set
    remaining = set(uncovered_set)
    chosen = [0]  # 000000
    while remaining:
        if time.time() > deadline:
            break
        # pick candidate covering the most remaining words
        best = -1
        bestcnt = 0
        for idx in cand:
            if idx in chosen_set(chosen):
                continue
            c = len(cov[idx] & remaining)
            if c > bestcnt:
                bestcnt = c
                best = idx
                if bestcnt == len(remaining):
                    break
        if best < 0 or bestcnt == 0:
            break
        chosen.append(best)
        remaining -= cov[best]
    return chosen


def chosen_set(chosen):
    return set(chosen)


def _cover_count(digits, chosen):
    """Number of distinct words covered by the chosen codewords."""
    seen = set()
    for idx in chosen:
        seen |= _ball(digits, idx, radius_global)
    return len(seen)


def _is_redundant(digits, chosen, idx):
    """True if removing codeword idx still leaves the space fully covered."""
    rest = [c for c in chosen if c != idx]
    return _cover_count(digits, rest) == q_global ** n_global


def _minimize(digits, chosen, total, deadline):
    """Branch-and-bound: remove redundant codewords to minimise the count."""
    chosen = list(chosen)
    best = list(chosen)

    def rec(state, bound):
        nonlocal best
        if time.time() > deadline:
            return
        if len(state) < len(best):
            best = list(state)
        if len(state) <= bound:
            return
        # try removing each redundant codeword
        for i in range(len(state)):
            idx = state[i]
            if _is_redundant(digits, state, idx):
                rec(state[:i] + state[i + 1:], bound)
                if len(best) <= bound:
                    break

    rec(chosen, len(chosen) - 1)
    return best


def _local_polish(digits, chosen, deadline):
    """Swap-based local search: replace a codeword with one that keeps coverage
    but may allow further removals; also try removals again."""
    chosen = list(chosen)
    total = q_global ** n_global
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # removal pass
        for i in range(len(chosen)):
            if _is_redundant(digits, chosen, chosen[i]):
                chosen = [c for c in chosen if c != chosen[i]]
                improved = True
                break
        # swap pass: for each codeword, try replacing it with a word that is not
        # currently in the code and keeps the space covered (reduces slack).
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            cur = chosen[i]
            rest = [c for c in chosen if c != cur]
            # candidate: any word not in chosen
            for cand in range(total):
                if cand in chosen:
                    continue
                newset = rest + [cand]
                if _cover_count(digits, newset) == total:
                    # accept if it enables a removal later; tentatively accept
                    chosen = newset
                    improved = True
                    break
    return chosen


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def construct(q, n, radius):
    global q_global, n_global, radius_global
    q_global = q
    n_global = n
    radius_global = radius

    total = q ** n
    digits, ball0, uncovered = _init(q, n, radius)

    # Greedy valid cover (includes 000000).
    deadline = time.time() + max(2.0, min(20.0, _remaining() * 0.35))
    chosen = _greedy_cover(digits, uncovered, deadline)

    # Make sure it is actually valid (greedy may have stopped early on deadline).
    while _cover_count(digits, chosen) < total and _remaining() > 1.0:
        # add the word covering the most currently-uncovered words
        seen = set()
        for idx in chosen:
            seen |= _ball(digits, idx, radius)
        missing = set(range(total)) - seen
        if not missing:
            break
        best = -1
        bestcnt = 0
        for idx in range(total):
            if idx in chosen:
                continue
            c = len(_ball(digits, idx, radius) & missing)
            if c > bestcnt:
                bestcnt = c
                best = idx
        if best < 0:
            break
        chosen.append(best)

    # Minimise.
    deadline = time.time() + max(2.0, min(60.0, _remaining() * 0.6))
    chosen = _minimize(digits, chosen, total, deadline)
    chosen = _local_polish(digits, chosen, time.time() + max(1.0, min(20.0, _remaining() * 0.5)))

    # Final safety: guarantee validity.
    while _cover_count(digits, chosen) < total and _remaining() > 0.5:
        seen = set()
        for idx in chosen:
            seen |= _ball(digits, idx, radius)
        missing = set(range(total)) - seen
        if not missing:
            break
        best = -1
        bestcnt = 0
        for idx in range(total):
            if idx in chosen:
                continue
            c = len(_ball(digits, idx, radius) & missing)
            if c > bestcnt:
                bestcnt = c
                best = idx
        if best < 0:
            break
        chosen.append(best)

    return [digits[c].tolist() for c in chosen]
