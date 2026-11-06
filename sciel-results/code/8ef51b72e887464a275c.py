import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 145.0

    n = 16

    def remaining_budget():
        return DEADLINE - (time.time() - start)

    # ---------- fast linearity check (bitmasks) ----------
    def build_linearity_masks(triples):
        pair_mask = [0] * (n * (n - 1) // 2)
        for (a, b, c) in triples:
            m = 0
            if a < b:
                m |= 1 << (a * (n - 1) - b * (b - 1) // 2)
            if a < c:
                m |= 1 << (a * (n - 1) - c * (c - 1) // 2)
            if b < c:
                m |= 1 << (b * (n - 1) - c * (c - 1) // 2)
            pair_mask[m] = 1
        return pair_mask

    def is_linear_add(triples, new, pair_mask):
        a, b, c = new
        if a == b or a == c or b == c:
            return False
        m = 0
        if a < b:
            m |= 1 << (a * (n - 1) - b * (b - 1) // 2)
        if a < c:
            m |= 1 << (a * (n - 1) - c * (c - 1) // 2)
        if b < c:
            m |= 1 << (b * (n - 1) - c * (c - 1) // 2)
        return pair_mask[m] == 0

    # ---------- independence number via bitmask subset enumeration ----------
    def independence_number(triples):
        # triple_masks: list of 16-bit masks; a set S is independent iff
        # no triple mask is a subset of S.
        tm = [0] * len(triples)
        for i, (a, b, c) in enumerate(triples):
            tm[i] = (1 << a) | (1 << b) | (1 << c)
        # Precompute for each subset S (0..2^n-1) whether it contains a triple.
        # We only need this for subsets of size >= 2 (any triple has 3 points).
        full = (1 << n) - 1
        # has_triple[S] = True iff exists t with tm[i] & ~S == 0
        has = [False] * (1 << n)
        # For each triple, mark all supersets.
        for mask in tm:
            # iterate over supersets of mask within [0, full]
            comp = full ^ mask
            sub = comp
            while True:
                has[mask | sub] = True
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        # Now find largest k with an independent k-subset
        # Enumerate subsets by popcount descending.
        # Precompute popcount for speed
        popc = [0] * (1 << n)
        for i in range(1, 1 << n):
            popc[i] = popc[i >> 1] + (i & 1)
        # group indices by popcount (only up to n)
        by_size = [[] for _ in range(n + 1)]
        for i in range(1 << n):
            by_size[popc[i]].append(i)
        for k in range(n, 2, -1):
            for S in by_size[k]:
                if not has[S]:
                    return k
        return 2

    def independence_number_fast(triples, cache=None):
        # Same as above; kept separate for clarity.
        return independence_number(triples)

    # ---------- initial system: cyclic STS(15) on points 0..14 + a few triples with 15 ----------
    def make_sts15():
        sts = set()
        base_blocks = [(0, 1, 4), (0, 2, 7), (0, 3, 9)]
        for bb in base_blocks:
            for d in range(15):
                b = tuple(sorted(((bb[0] + d) % 15, (bb[1] + d) % 15, (bb[2] + d) % 15)))
                sts.add(b)
        return [list(t) for t in sorted(sts)]

    def verify_linear(triples):
        # O(T^2) check for debugging / safety
        for i in range(len(triples)):
            si = set(triples[i])
            for j in range(i + 1, len(triples)):
                if len(si & set(triples[j])) > 1:
                    return False
        return True

    sts15 = make_sts15()
    if not verify_linear(sts15):
        # Fallback: disjoint triples (guaranteed linear)
        sts15 = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    # Add a few triples containing 15 to give point 15 some structure.
    # Choose pairs from 0..14 that are not already a pair in sts15 (to keep linearity easy).
    existing_pairs = set()
    for (a, b, c) in sts15:
        existing_pairs.add((min(a, b), max(a, b)))
        existing_pairs.add((min(a, c), max(a, c)))
        existing_pairs.add((min(b, c), max(b, c)))

    seed_15 = []
    # Add triples [15, a, b] where {a,b} is not an existing pair (so no shared pair with sts15)
    # and the two points a,b are not both already paired with 15 in seed_15.
    used_with_15 = set()
    for a in range(15):
        for b in range(a + 1, 15):
            if (a, b) in existing_pairs:
                continue
            if a in used_with_15 or b in used_with_15:
                continue
            seed_15.append((15, a, b))
            used_with_15.add(a)
            used_with_15.add(b)
            if len(seed_15) >= 5:
                break
        if len(seed_15) >= 5:
            break

    triples = sts15 + [list(t) for t in seed_15]

    # Safety: verify linearity of the initial system
    if not verify_linear(triples):
        # Drop any offending triples with 15
        triples = sts15 + [t for t in seed_15 if is_linear_add(sts15, t, build_linearity_masks(sts15))]

    pair_mask = build_linearity_masks(triples)
    best_indep = independence_number(triples)

    # ---------- greedy: add any feasible triple that lowers the independence number ----------
    # Precompute all candidate triples (a,b,c) with a<b<c
    all_candidates = list(combinations(range(n), 3))
    random.shuffle(all_candidates)

    improved = True
    while improved and remaining_budget() > 10.0:
        improved = False
        for cand in all_candidates:
            if remaining_budget() < 8.0:
                break
            if not is_linear_add(triples, cand, pair_mask):
                continue
            # Try adding
            new_triples = triples + [list(cand)]
            new_indep = independence_number(new_triples)
            if new_indep < best_indep:
                triples = new_triples
                best_indep = new_indep
                pair_mask = build_linearity_masks(triples)
                improved = True
                # Restart scan from beginning (order matters; reshuffle occasionally)
                if remaining_budget() < 8.0:
                    break
                # Occasionally reshuffle to escape local order bias
                if random.random() < 0.15:
                    random.shuffle(all_candidates)

    # ---------- final safety: verify linearity ----------
    if not verify_linear(triples):
        # Rebuild a guaranteed-linear system: keep only a maximal linear subset
        kept = []
        pm = [0] * (n * (n - 1) // 2)
        for t in triples:
            if is_linear_add(kept, t, pm):
                kept.append(t)
                a, b, c = t
                m = 0
                if a < b:
                    m |= 1 << (a * (n - 1) - b * (b - 1) // 2)
                if a < c:
                    m |= 1 << (a * (n - 1) - c * (c - 1) // 2)
                if b < c:
                    m |= 1 << (b * (n - 1) - c * (c - 1) // 2)
                pm[m] = 1
        triples = kept

    return triples
