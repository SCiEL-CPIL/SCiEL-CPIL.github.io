"""Covering design C(24,7,3): minimize number of 7-blocks covering all 3-subsets."""
import itertools
import random
import time


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0          # safety margin under the 150s kill

    # All t-subsets, indexed for fast coverage counting
    triples = list(itertools.combinations(range(v), t))
    idx = {tr: i for i, tr in enumerate(triples)}
    ntr = len(triples)

    def build_greedy(rng, use_random_tie):
        """One randomized greedy covering; returns list of blocks (sorted tuples)."""
        uncovered = set(range(ntr))
        cov = {}                       # block(tuple) -> frozenset of triple indices
        blocks = []
        while uncovered:
            first = rng.choice(list(uncovered))
            block = [triples[first][0]]
            while len(block) < k:
                bset = frozenset(block)
                best_x = None
                best_gain = -1
                best_rand = -1.0
                for x in range(v):
                    if x in bset:
                        continue
                    gain = 0
                    for comb in itertools.combinations(block, t - 1):
                        tr = comb + (x,)
                        i = idx[tr]
                        if i in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_rand = rng.random() if use_random_tie else 0.0
                    elif gain == best_gain and use_random_tie:
                        r = rng.random()
                        if r > best_rand:
                            best_rand = r
                            best_x = x
                block.append(best_x)
            b = tuple(sorted(block))
            cset = frozenset(idx[tr] for tr in itertools.combinations(b, t))
            blocks.append(b)
            cov[b] = cset
            uncovered.difference_update(cset)
        return blocks

    def remove_redundant(blocks):
        """Drop blocks that are not needed to cover all triples."""
        cov = {}
        for b in blocks:
            cov[b] = frozenset(idx[tr] for tr in itertools.combinations(b, t))
        count = [0] * ntr
        for cset in cov.values():
            for i in cset:
                count[i] += 1
        cur = list(blocks)
        changed = True
        while changed:
            changed = False
            for pos in range(len(cur)):
                b = cur[pos]
                cset = cov[b]
                if any(count[i] <= 1 for i in cset):
                    continue          # removing b would uncover some triple
                for i in cset:
                    count[i] -= 1
                del cov[b]
                cur.pop(pos)
                changed = True
                break
        return cur

    # Phase 1: randomized greedy restarts, keep the best (fewest blocks)
    best = None
    rng = random.Random(20240607)
    nrestart = 0
    while True:
        if best is None:
            cand = build_greedy(rng, use_random_tie=False)
        else:
            cand = build_greedy(rng, use_random_tie=True)
        nrestart += 1
        cand = remove_redundant(cand)
        if best is None or len(cand) < len(best):
            best = cand
        if time.time() > deadline - 15.0:
            break
        if nrestart >= 24:            # cap: greedy is cheap, but don't over-spend
            break
        if len(best) <= 70:           # near lower bound; extra restarts unlikely to help
            break

    # Phase 2: final removal polish on the best (already applied; re-run is a no-op
    # safety net in case best came from a non-polished path)
    best = remove_redundant(best)

    return [list(b) for b in best]
