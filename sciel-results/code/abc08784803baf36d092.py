"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

MASK128 = (1 << 128) - 1


def _is_valid(S):
    """Return True iff no three distinct members a,b,c have all three
    pair sums a+b, a+c, b+c also in S."""
    Sset = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j + 1, n):
                c = S[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True


def _can_add(mask, x):
    """Exact feasibility: can x be added to the set represented by `mask`
    without creating a forbidden triple?

    A newly-forbidden triple must contain x. For distinct a,b already in S:
      sums are a+b, a+x, b+x.
    Since a,b >= 1, a+x > x and b+x > x, so a+x and b+x must be in the OLD set.
      a+b must be in the OLD set, OR a+b == x.
    So x is addable iff there is NO pair a<b in S with
        (a+x in S) and (b+x in S) and (a+b in S or a+b == x).
    """
    max_v = 128 - x
    if max_v < 1:
        return True
    for a in range(1, max_v + 1):
        if not ((mask >> a) & 1):
            continue
        if not ((mask >> (x + a)) & 1):
            continue
        for b in range(a + 1, max_v + 1):
            if not ((mask >> b) & 1):
                continue
            if not ((mask >> (x + b)) & 1):
                continue
            ab = a + b
            if ab == x or ((mask >> ab) & 1):
                return False
    return True


def _popcount(mask):
    return bin(mask).count('1')


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before the 90s kill

    # Known-valid seed of size 67: S = {1} U {62} U {64, ..., 128}
    S = [1, 62] + list(range(64, 129))
    mask = 0
    for v in S:
        mask |= (1 << v)
    best_mask = mask
    best_size = len(S)

    def time_left():
        return deadline - time.time()

    def elems_of(m):
        out = []
        for i in range(1, 129):
            if (m >> i) & 1:
                out.append(i)
        return out

    def greedy_add(m, deadline):
        """Repeatedly add any feasible element until none remain. Returns (m, size)."""
        size = _popcount(m)
        progress = True
        while progress and time_left() > 2.0:
            progress = False
            for x in range(1, 129):
                if (m >> x) & 1:
                    continue
                if _can_add(m, x):
                    m |= (1 << x)
                    size += 1
                    progress = True
                    break
        return m, size

    # ---- Phase 1: fast greedy add from the seed ----
    mask, sz = greedy_add(mask, deadline)
    if sz > best_size:
        best_size = sz
        best_mask = mask

    # ---- Phase 2: local search via swaps (remove one, then greedily add) ----
    # Escapes the greedy maximal independent set by sacrificing an element to
    # enable a cascade of additions.  Uses the exact bitmask feasibility oracle.
    no_improve = 0
    while time_left() > 5.0 and no_improve < 1500:
        if time_left() < 2.0:
            break
        elems = elems_of(mask)
        made_progress = False
        for r in elems:
            if time_left() < 2.0:
                break
            without = mask & ~(1 << r)
            nm, nsz = greedy_add(without, deadline)
            if nsz > best_size:
                mask = nm
                best_size = nsz
                best_mask = nm
                no_improve = 0
                made_progress = True
                break
            elif nsz > _popcount(mask):
                # Not a global improvement but grows past the current plateau;
                # accept to keep exploring (best_mask still guards the return).
                mask = nm
                no_improve = 0
                made_progress = True
                break
        if not made_progress:
            no_improve += 1

    # ---- Phase 3: final safety validation + fallback ----
    if best_size < 67:
        return [1, 62] + list(range(64, 129))
    result = elems_of(best_mask)
    if not _is_valid(result):
        return [1, 62] + list(range(64, 129))
    return [int(x) for x in sorted(result)]

# EVOLVE-BLOCK-END
