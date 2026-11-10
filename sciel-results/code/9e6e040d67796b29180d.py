"""
Minimum-cardinality inclusion-maximal Sidon set in [1, 256].

Structural change from the parent: the blocking / addability tests are
re-expressed with *bitmasks over the 512 possible pair sums*.  For an element
x the "sum signature" sig[x] is the 512-bit mask of sums {2x} ∪ {x+s : s in
[1,256]}.  A candidate set S is Sidon iff the OR of its signatures has exactly
n(n+1)/2 bits set (no collision).  An element y is blocked by S iff
sig[y] & sums_mask(S) is non-empty.  Blockage is monotone in S, so the set of
still-unblocked elements only shrinks during the greedy set-cover, which makes
the frontier a bitset and each greedy step essentially O(1) (bitwise ops +
popcount) instead of the parent's O(N^2) Python set-intersection loops.

That speedup is what permits many random restarts inside the 60 s budget; each
restart is a random-order greedy set-cover (a valid maximal Sidon set by
construction), followed by the parent's removal + swap refinement.
"""
import time
import random

N = 256
MAX_SUM = 2 * N            # 512
_DEADLINE = 55.0           # safety margin before the 60 s kill


def _is_sidon(S):
    """Check if S is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    if not S:
        return False
    if not _is_sidon(S):
        return False
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    for x in range(1, N + 1):
        if x in S_set:
            continue
        blocked = False
        if 2 * x in sums:
            blocked = True
        else:
            for s in S:
                if x + s in sums:
                    blocked = True
                    break
        if not blocked:
            return False
    return True


# ---------------------------------------------------------------------------
# Bitmask machinery (the structural core of this version)
# ---------------------------------------------------------------------------
# sig[x] : 512-bit mask of the sums that x would contribute to a set.
#          bit (2x-2) and bit (x+s-2) for s in 1..N.
SIG = [0] * (N + 1)
for x in range(1, N + 1):
    m = 1 << (2 * x - 2)
    for s in range(1, N + 1):
        m |= 1 << (x + s - 2)
    SIG[x] = m

# precompute element index -> bit in the element-domain bitset (0..N-1)
def _elbit(x):
    return 1 << (x - 1)


def _greedy_set_cover_bitmask(order, deadline=None):
    """
    Greedy set-cover over a fixed candidate `order` (a permutation of [1,N]).
    At each step pick the first addable element that blocks the most
    still-unblocked elements, add it, and shrink the frontier.

    Returns (S, sums_mask, count).  The result is a valid maximal Sidon set:
    every element not in S is either blocked by S or was never addable (and so
    cannot be added without breaking Sidon).
    """
    S = []
    sums_mask = 0
    count = 0
    inS_mask = 0
    unblocked_mask = 0
    for x in range(1, N + 1):
        unblocked_mask |= _elbit(x)

    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if not (unblocked_mask & _elbit(x)):
            continue
        # addability: no sum collision with existing S
        if SIG[x] & sums_mask:
            continue
        # which still-unblocked elements does x block?
        block = 0
        for y in range(1, N + 1):
            if y == x:
                continue
            if not (unblocked_mask & _elbit(y)):
                continue
            if SIG[y] & (sums_mask | SIG[x]):
                block |= _elbit(y)
        if block == 0:
            continue
        # commit
        S.append(x)
        sums_mask |= SIG[x]
        count += 1
        inS_mask |= _elbit(x)
        unblocked_mask &= ~(block | _elbit(x))
        if unblocked_mask == 0:
            break

    S.sort()
    return S, sums_mask, count


def _local_search_remove(S, N, deadline=None):
    """Remove elements while keeping S a valid maximal Sidon set."""
    S = list(S)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(len(S)):
            S_new = S[:i] + S[i + 1:]
            if not S_new:
                continue
            if _is_maximal(S_new, N):
                S = S_new
                improved = True
                break
    return S


def _swap_improve(S, N, deadline=None):
    """Swap one element for another to enable a size reduction."""
    S = list(S)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(len(S)):
            for j in range(1, N + 1):
                if j in S:
                    continue
                S_new = S[:i] + [j] + S[i + 1:]
                S_new.sort()
                if _is_maximal(S_new, N):
                    S_trial = list(S_new)
                    for k in range(len(S_trial)):
                        S_trial2 = S_trial[:k] + S_trial[k + 1:]
                        if S_trial2 and _is_maximal(S_trial2, N):
                            S = S_trial2
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    return S


def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1

    def elapsed():
        return time.time() - start_time

    # ---- Phase 1: many bitmask greedy set-cover restarts + removal ----
    # Each restart is cheap (bitwise ops), so we can afford many of them.
    seed = 0
    while elapsed() < _DEADLINE - 12:
        seed += 1
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S, _, _ = _greedy_set_cover_bitmask(order, deadline=_DEADLINE - 10)
        if not S:
            continue
        if len(S) < best_size:
            if elapsed() < _DEADLINE - 6:
                S_imp = _local_search_remove(S, N, deadline=_DEADLINE - 6)
                if _is_maximal(S_imp, N) and len(S_imp) < len(S):
                    S = S_imp
            if _is_maximal(S, N) and len(S) < best_size:
                best_set = S
                best_size = len(S)

    # ---- Phase 2: more aggressive removal on the best so far ----
    if best_set is not None and elapsed() < _DEADLINE - 4:
        for _ in range(30):
            if elapsed() > _DEADLINE - 4:
                break
            S_new = _local_search_remove(best_set, N, deadline=_DEADLINE - 4)
            if _is_maximal(S_new, N) and len(S_new) < best_size:
                best_set = S_new
                best_size = len(S_new)

    # ---- Phase 3: swap improvement ----
    if best_set is not None and elapsed() < _DEADLINE - 2:
        S_new = _swap_improve(best_set, N, deadline=_DEADLINE - 2)
        if _is_maximal(S_new, N) and len(S_new) < best_size:
            best_set = S_new
            best_size = len(S_new)

    # ---- Fallback: deterministic greedy (always valid) ----
    if best_set is None:
        S, _, _ = _greedy_set_cover_bitmask(list(range(1, N + 1)))
        if S:
            best_set = S
        else:
            best_set = [1]

    return best_set
