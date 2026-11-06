"""Rewritten program: B&B MIS on 3-uniform hypergraph with matching upper bound."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

_START = None
_DEADLINE = 80.0  # seconds wall-clock safety margin

def _lcm(a, b):
    return a // gcd(a, b) * b

def _is_bad_triple(a, b, c):
    l_ab = _lcm(a, b)
    if l_ab != _lcm(a, c):
        return False
    if l_ab != _lcm(b, c):
        return False
    return True

def construct():
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + _DEADLINE

    # Precompute all bad triples
    bad_triples = []
    for a in range(1, 129):
        for b in range(a + 1, 129):
            for c in range(b + 1, 129):
                if _is_bad_triple(a, b, c):
                    bad_triples.append((a, b, c))

    # Per-vertex index of triples containing it
    vertex_triples = defaultdict(list)
    for i, (a, b, c) in enumerate(bad_triples):
        for v in (a, b, c):
            vertex_triples[v].append(i)

    # Triple bitmasks (vertex v -> bit v-1)
    triple_masks = [
        (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        for (a, b, c) in bad_triples
    ]

    # Per-vertex list of "other two" masks for include-feasibility
    other_two_masks = defaultdict(list)
    for (a, b, c) in bad_triples:
        for v in (a, b, c):
            others = [x for x in (a, b, c) if x != v]
            other_two_masks[v].append((1 << (others[0] - 1)) | (1 << (others[1] - 1)))

    def popcount(x):
        return bin(x).count('1')

    # Greedy initial solution (strong lower bound)
    def greedy_init():
        S = set()
        deg = {v: len(vertex_triples[v]) for v in range(1, 129)}
        order = sorted(range(1, 129), key=lambda v: -deg[v])
        for v in order:
            ok = True
            for om in other_two_masks[v]:
                # need both others in S
                o0 = (om >> 0) & 1
                # decode the two vertices
                # simpler: recompute from om bits
                bits = []
                m = om
                while m:
                    low = m & (-m)
                    bits.append(low.bit_length())
                    m ^= low
                if len(bits) == 2 and bits[0] in S and bits[1] in S:
                    ok = False
                    break
            if ok:
                S.add(v)
        return S

    best_set = greedy_init()
    best_size = len(best_set)

    # B&B
    all_mask = (1 << 128) - 1

    def matching_bound(avail_mask):
        count = 0
        cur = avail_mask
        for tm in triple_masks:
            if (tm & cur) == tm:
                count += 1
                cur &= ~tm
        return count

    def can_include(v, included_mask):
        for om in other_two_masks[v]:
            if (om & included_mask) == om:
                return False
        return True

    def count_triples_containing(v, avail_mask):
        cnt = 0
        for ti in vertex_triples[v]:
            if (triple_masks[ti] & avail_mask) == triple_masks[ti]:
                cnt += 1
        return cnt

    best = [best_size]
    initial_included_mask = 0
    for v in best_set:
        initial_included_mask |= (1 << (v - 1))
    best_mask = [initial_included_mask]

    import sys
    sys.setrecursionlimit(100000)

    def dfs(avail_mask, included_mask, inc_count):
        if time.time() > deadline:
            return
        if inc_count + popcount(avail_mask) <= best[0]:
            return
        if avail_mask == 0:
            if inc_count > best[0]:
                best[0] = inc_count
                best_mask[0] = included_mask
            return
        mb = matching_bound(avail_mask)
        if inc_count + popcount(avail_mask) - mb <= best[0]:
            return
        # collect available vertices
        m = avail_mask
        avail_list = []
        while m:
            low = m & (-m)
            avail_list.append(low.bit_length())
            m ^= low
        # pick branching vertex: max fully-available triple count
        bv = avail_list[0]
        best_cnt = -1
        for v in avail_list:
            c = count_triples_containing(v, avail_mask)
            if c > best_cnt:
                best_cnt = c
                bv = v
            if time.time() > deadline:
                return
        bit = 1 << (bv - 1)
        # include branch first
        if can_include(bv, included_mask):
            dfs(avail_mask & ~bit, included_mask | bit, inc_count + 1)
        # exclude branch
        dfs(avail_mask & ~bit, included_mask, inc_count)

    dfs(all_mask, 0, 0)

    result = []
    bm = best_mask[0]
    for v in range(1, 129):
        if bm & (1 << (v - 1)):
            result.append(v)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
