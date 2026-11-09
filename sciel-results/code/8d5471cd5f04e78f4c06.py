"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

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


def _can_add(x, S, Sset):
    """Return True iff adding x to S (x not in S) keeps S valid.
    Only triples containing x can become newly forbidden, so check those."""
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            if (x + a) in Sset and (x + b) in Sset and (a + b) in Sset:
                return False
    return True


def _greedy_add(S, Sset, candidates, deadline, step):
    """Greedily add candidates that keep the set valid. Returns (S, Sset)."""
    for x in candidates:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(x, S, Sset):
            Sset.add(x)
            S.append(x)
    return S, Sset


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before the 90s kill

    # Known-valid seed of size 67:
    # S = {1} U {62} U {64, ..., 128}
    # - Triples not involving 1 lie in {62, 64..128}; pair sums there are
    #   only 126, 127, 128 and no triple has all three sums present.
    # - Triples involving 1: {1, b, c}, b,c >= 64. Then b+c >= 129, which is
    #   outside S, so the triple is never forbidden.
    S = [1, 62] + list(range(64, 129))
    Sset = set(S)
    best_S = list(S)
    best_size = len(S)

    if time.time() > deadline:
        return list(best_S)

    # Bit-parallel "forbidden pairs" for a candidate x: for each a in S,
    # the set of b in S such that {x, a, b} is a forbidden triple.
    # Represent S as a 128-bit mask; forbidden pairs = OR over a of
    #   (mask & shift(mask, x + a))  where bit i of mask = (i in S).
    # Adding x is safe iff x has no forbidden partner (i.e. the OR is 0).
    def _mask_of(sset):
        m = 0
        for v in sset:
            m |= (1 << v)
        return m

    def _addable(x, mask):
        forb = 0
        # iterate over set bits a in S
        m = mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            shift = x + a
            if shift <= 127:
                forb |= (mask << shift) & ((1 << 128) - 1)
            m ^= lb
        return forb == 0

    # Precompute masks for the fixed large block {64..128} and the full seed.
    # But S changes as we add, so recompute the mask after each accepted add.
    # Candidates: the 61 integers in 2..63 (everything not in the seed).
    candidates = [v for v in range(2, 64)]

    # Order candidates by "cost" (how many forbidden triples they create),
    # cheapest first, to pack more in. Cost = number of (a,b) pairs in S
    # making {x,a,b} forbidden, computed via the bit-parallel OR popcount.
    def _cost(x, mask):
        forb = 0
        m = mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            shift = x + a
            if shift <= 127:
                forb |= (mask << shift) & ((1 << 128) - 1)
            m ^= lb
        return forb.bit_count()

    mask = _mask_of(Sset)
    remaining = list(candidates)

    # Iteratively pick the lowest-cost addable candidate, add it, update mask.
    # Repeat until no candidate is addable or time runs out.
    while True:
        if time.time() > deadline:
            break
        if not remaining:
            break
        # Compute cost for each remaining candidate; pick min-cost addable one.
        best_x = None
        best_c = None
        for x in remaining:
            if x in Sset:
                continue
            c = _cost(x, mask)
            if c == 0:
                # Addable immediately (no forbidden pair).
                best_x = x
                best_c = 0
                break
            if best_c is None or c < best_c:
                best_c = c
                best_x = x
        if best_x is None or best_c is None:
            break
        if best_c > 0:
            # No zero-cost candidate left; adding any would create a forbidden
            # triple, so stop (greedy maximal independent set reached).
            break
        # Add best_x (cost 0, guaranteed safe).
        Sset.add(best_x)
        S.append(best_x)
        mask |= (1 << best_x)
        remaining.remove(best_x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    # Fallback safety: if somehow the greedy result is invalid, return the seed.
    if len(best_S) > best_size or not _is_valid(best_S):
        # _is_valid on best_S is O(n^3) ~ 67^3 ~ 300k, cheap.
        if not _is_valid(best_S):
            return [1, 62] + list(range(64, 129))

    result = [int(x) for x in sorted(best_S)]
    return result

# EVOLVE-BLOCK-END
