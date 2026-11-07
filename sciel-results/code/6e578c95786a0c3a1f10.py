"""Cochromatic partition of a pinned random graph via maximal-set B&B."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    t_start = time.time()
    deadline = t_start + 50.0  # safety margin before 60s kill

    # Build adjacency as bitmasks (graph + complement)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    full = (1 << n) - 1
    comp = [full & ~adj[i] & ~(1 << i) for i in range(n)]

    # Bron-Kerbosch (pivot) to enumerate ALL maximal cliques of a bitmask graph
    def maximal_cliques(g):
        result = []
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if P == 0 and X == 0:
                result.append(R)
                return
            # choose pivot u from P | X maximizing |P & N(u)|
            PX = P | X
            best_u = -1
            best_cnt = -1
            m = PX
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                c = (P & g[u]).bit_count()
                if c > best_cnt:
                    best_cnt = c
                    best_u = u
                m ^= lsb
            if best_u < 0:
                return
            # iterate over P \ N(u)
            cand = P & ~g[best_u]
            m = cand
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                bk(R | (1 << v), P & g[v], X & g[v])
                P &= ~(1 << v)
                X |= (1 << v)
        bk(0, full, 0)
        return result

    max_cliques = maximal_cliques(adj)
    max_indep = maximal_cliques(comp)

    # Candidate masks = maximal cliques + maximal independent sets, dedup
    cands = list(dict.fromkeys(max_cliques + max_indep))
    cands.sort(key=lambda m: -m.bit_count())

    # Greedy upper bound (largest-first)
    def greedy():
        parts = []
        rem = full
        while rem:
            best = 0
            for m in cands:
                s = m & rem
                if s.bit_count() > best.bit_count():
                    best = s
                    if best == rem:
                        break
            parts.append(best)
            rem &= ~best
        return parts

    greedy_parts = greedy()
    best_count = len(greedy_parts)
    best_parts = greedy_parts[:]

    # Lower bound: ceil(#remaining / max candidate size)
    max_size = max(m.bit_count() for m in cands) if cands else 1
    memo = {}

    def lb(rem):
        if rem == 0:
            return 0
        return (rem.bit_count() + max_size - 1) // max_size

    # Iterative deepening DFS: find a partition using <= limit parts
    def dfs(rem, parts, limit):
        nonlocal best_count, best_parts
        if time.time() > deadline:
            return False
        if rem == 0:
            if len(parts) < best_count:
                best_count = len(parts)
                best_parts = parts[:]
            return True
        if len(parts) + lb(rem) > limit:
            return False
        # Try candidates that intersect rem, largest coverage first
        seen = set()
        for m in cands:
            s = m & rem
            if s == 0 or s in seen:
                continue
            seen.add(s)
            parts.append(s)
            if dfs(rem & ~s, parts, limit):
                parts.pop()
                return True
            parts.pop()
        return False

    # Try to improve below the greedy bound
    for limit in range(best_count - 1, 0, -1):
        if time.time() > deadline:
            break
        if dfs(full, [], limit):
            break

    # Build result from best_parts (bitmasks -> lists of ints)
    result = []
    for m in best_parts:
        result.append([int(v) for v in range(n) if (m >> v) & 1])
    return result
# EVOLVE-BLOCK-END
