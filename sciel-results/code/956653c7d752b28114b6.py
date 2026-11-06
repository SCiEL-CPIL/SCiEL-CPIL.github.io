"""Find maximum subset of [1,180] with no three distinct a,b,c where 1/a=1/b+1/c.

Structural approach: model as Maximum Independent Set on a 3-uniform hypergraph
(vertices = elements 1..180, hyperedges = forbidden triples).  Solve with a
branch-and-bound that uses a strong upper bound (greedy set-cover lower bound on
the number of further removals still required).  This explores many removal
orderings instead of committing to a single greedy order, and is then polished
with a repair (add-back / swap) local search.
"""
import time
import sys

N = 180
DEADLINE = 82.0  # safety margin under the 90s wall-clock kill


def _find_triples(N=N):
    """All unordered triples of DISTINCT elements in [1,N] with 1/a=1/b+1/c."""
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            den = a + b
            if (a * b) % den != 0:
                continue
            c = (a * b) // den
            if c < 1 or c > N or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                triples.append(t)
    return triples


def _greedy_seed(triples, N=N):
    """Greedy removal to break every triple; return a valid set (as bitmask)."""
    S = set(range(1, N + 1))
    active = list(range(len(triples)))
    while active:
        cnt = {}
        for ti in active:
            for x in triples[ti]:
                if x in S:
                    cnt[x] = cnt.get(x, 0) + 1
        if not cnt:
            break
        worst = max(cnt, key=lambda x: (cnt[x], -x))
        S.remove(worst)
        active = [ti for ti in active if all(x in S for x in triples[ti])]
    mask = 0
    for x in S:
        mask |= 1 << x
    return mask


def _repair(mask, triples, N=N):
    """Try to add back / swap in elements to grow the valid set.  Returns mask."""
    FULL = (1 << (N + 1)) - 2  # bits 1..N
    S = mask
    removed = FULL ^ S
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if (S >> x) & 1:
                continue
            # adding x valid?
            ok = True
            for t in triples:
                if x in t and ((S >> t[0]) & 1) and ((S >> t[1]) & 1) and ((S >> t[2]) & 1):
                    ok = False
                    break
            if ok:
                S |= (1 << x)
                improved = True
    # swap phase: drop one, add one (net zero size) to enable later growth
    for _ in range(60):
        found = False
        for x in range(1, N + 1):
            if not ((S >> x) & 1):
                continue
            for y in range(1, N + 1):
                if (S >> y) & 1:
                    continue
                cand = S & ~(1 << x) | (1 << y)
                ok = True
                for t in triples:
                    if ((cand >> t[0]) & 1) and ((cand >> t[1]) & 1) and ((cand >> t[2]) & 1):
                        ok = False
                        break
                if ok:
                    S = cand
                    found = True
                    break
            if found:
                break
        if not found:
            break
    # final add-back pass after swaps
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if (S >> x) & 1:
                continue
            ok = True
            for t in triples:
                if x in t and ((S >> t[0]) & 1) and ((S >> t[1]) & 1) and ((S >> t[2]) & 1):
                    ok = False
                    break
            if ok:
                S |= (1 << x)
                improved = True
    return S


def _bnb(triples, N=N, deadline=None, best=[0, 0]):
    """Branch-and-bound maximum independent set.  Updates best[1] (mask) in place."""
    m = len(triples)
    # element -> list of triple indices
    elem_triples = [[] for _ in range(N + 1)]
    for ti in range(m):
        for x in triples[ti]:
            elem_triples[x].append(ti)

    sys.setrecursionlimit(100000)
    nodes = [0]
    FULL = (1 << (N + 1)) - 2

    def solve(S, R):
        if deadline is not None and time.time() > deadline:
            return
        nodes[0] += 1
        # R: bitmask of triple indices still "alive" (all 3 elements in S)
        if R == 0:
            sz = bin(S).count("1")
            if sz > best[0]:
                best[0] = sz
                best[1] = S
            return
        # upper bound: |S| - (greedy set-cover lower bound on removals needed)
        # compute greedy cover count
        cnt = [0] * (N + 1)
        r = R
        while r:
            lb = r & (-r)
            ti = lb.bit_length() - 1
            r ^= lb
            for x in triples[ti]:
                cnt[x] += 1
        # greedy set cover
        cover = 0
        rr = R
        while rr:
            mx = 0
            mxel = 0
            for x in range(1, N + 1):
                if cnt[x] > mx:
                    mx = cnt[x]
                    mxel = x
            if mx == 0:
                break
            cover += 1
            for ti in elem_triples[mxel]:
                if (rr >> ti) & 1:
                    rr ^= (1 << ti)
                    for x in triples[ti]:
                        cnt[x] -= 1
        ub = bin(S).count("1") - cover
        if ub <= best[0]:
            return
        # pick branching element: in most alive triples
        mx = 0
        mxel = 0
        for x in range(1, N + 1):
            if cnt[x] > mx:
                mx = cnt[x]
                mxel = x
        if mx == 0:
            sz = bin(S).count("1")
            if sz > best[0]:
                best[0] = sz
                best[1] = S
            return
        # Branch A: remove mxel
        S1 = S & ~(1 << mxel)
        R1 = R
        for ti in elem_triples[mxel]:
            R1 &= ~(1 << ti)
        solve(S1, R1)
        # Branch B: keep mxel -> must remove all its triple-mates
        S2 = S
        R2 = R
        for ti in elem_triples[mxel]:
            if (R2 >> ti) & 1:
                R2 &= ~(1 << ti)
                for x in triples[ti]:
                    if x != mxel:
                        S2 &= ~(1 << x)
        solve(S2, R2)

    solve(FULL, (1 << m) - 1)
    return best[1]


def construct():
    start = time.time()
    deadline = start + DEADLINE

    triples = _find_triples(N)
    m = len(triples)

    # 1) greedy seed
    seed = _greedy_seed(triples, N)
    seed = _repair(seed, triples, N)
    best = [bin(seed).count("1"), seed]

    # 2) branch-and-bound to improve
    if time.time() < deadline - 3.0:
        bnb_out = _bnb(triples, N, deadline=deadline - 3.0, best=best)
        best[1] = bnb_out

    # 3) final polish
    best[1] = _repair(best[1], triples, N)
    best[0] = bin(best[1]).count("1")

    result = [x for x in range(1, N + 1) if (best[1] >> x) & 1]
    if not result:
        result = [1]
    return result


if __name__ == "__main__":
    res = construct()
    print(len(res))
