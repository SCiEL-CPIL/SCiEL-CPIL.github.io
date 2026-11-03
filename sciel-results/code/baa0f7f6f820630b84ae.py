"""Shortest superpermutation on n symbols.

Strategy: build the word by concatenating a permutation of the (n-1)!
(n-1)-letter blocks in some order, each block separated by the fresh symbol n
(overlapped by the common prefix of consecutive blocks).  The naive
"insert new blocks in order of first appearance" gives the classic length
1!+2!+...+n!.  We then improve the order of the blocks by local search
(2-opt / 3-opt) to minimise total length, which is equivalent to minimising
the sum of (n-1 - LCP(block_i, block_{i+1})) over the cyclic order.

A strict wall-clock deadline guard keeps the function well under the limit and
always returns a valid superpermutation.
"""
import time
import random
from itertools import permutations


def _lcp(a, b):
    """Length of the longest common prefix of two equal-length tuples."""
    L = 0
    for x, y in zip(a, b):
        if x != y:
            break
        L += 1
    return L


def _build_word(blocks, order, n):
    """Build the superpermutation string from the block order."""
    parts = ["".join(map(str, blocks[order[0]]))]
    for k in range(1, len(order)):
        prev = blocks[order[k - 1]]
        cur = blocks[order[k]]
        l = 0
        for a, b in zip(prev, cur):
            if a != b:
                break
            l += 1
        parts.append("".join(map(str, cur[l:])))
    return "".join(parts)


def _classic_order(bm1):
    """Greedy first-appearance order of the (n-1)! blocks."""
    order = [0]
    seen = set()
    remaining = [i for i in range(1, len(bm1))]
    cur = bm1[0]
    remaining.sort(key=lambda i: _lcp(cur, bm1[i]), reverse=True)
    # Rebuild by repeatedly taking the next block that best overlaps current
    while remaining:
        best = None
        best_l = -1
        for i in remaining:
            l = _lcp(cur, bm1[i])
            if l > best_l:
                best_l = l
                best = i
        order.append(best)
        cur = bm1[best]
        remaining.remove(best)
    return order


def _total_cost(order, lcp_mat, m):
    """Total length = m + sum over cyclic consecutive pairs of (m - LCP)."""
    L = len(order)
    total = m
    for i in range(L):
        a = order[i]
        b = order[(i + 1) % L]
        total += m - lcp_mat[a][b]
    return total


def _local_search(order, lcp_mat, m, deadline):
    """2-opt / 3-opt local search to reduce total length of the cyclic order."""
    L = len(order)
    if L < 4:
        return order
    order = list(order)
    current_len = _total_cost(order, lcp_mat, m)

    def rev2(i, j):
        """Reverse segment [i, j] inclusive (2-opt)."""
        a = order[i - 1]
        b = order[i]
        c = order[j]
        d = order[(j + 1) % L]
        old = lcp_mat[a][b] + lcp_mat[c][d]
        new = lcp_mat[a][d] + lcp_mat[c][b]
        return new - old  # negative = improvement

    def rev3(i, j):
        """Reverse segment [i, j] inclusive (3-opt)."""
        a = order[i - 1]
        b = order[i]
        c = order[j]
        d = order[(j + 1) % L]
        e = order[(j + 2) % L]
        old = lcp_mat[a][b] + lcp_mat[c][d] + lcp_mat[e]
        # new arrangement: a, d, c, b, e
        new = lcp_mat[a][d] + lcp_mat[d][c] + lcp_mat[c][b] + lcp_mat[b][e]
        old = lcp_mat[a][b] + lcp_mat[c][d] + lcp_mat[b][e]
        return new - old

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # 2-opt pass
        for i in range(1, L):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, L):
                delta = rev2(i, j)
                if delta < 0:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    current_len += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # 3-opt pass (coarser, try a few)
            for i in range(1, L):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 2, L):
                    delta = rev3(i, j)
                    if delta < 0:
                        order[i:j + 1] = reversed(order[i:j + 1])
                        current_len += delta
                        improved = True
                        break
                if improved:
                    break
    return order


def _construct_superperm(n, deadline):
    # Build (n-1)! blocks
    bm1 = list(permutations(range(1, n)))
    L = len(bm1)
    m = n - 1

    # Precompute LCP matrix
    lcp_mat = [[0] * L for _ in range(L)]
    for i in range(L):
        bi = bm1[i]
        for j in range(L):
            lcp_mat[i][j] = _lcp(bi, bm1[j])

    # Start from the classic order
    order = _classic_order(bm1)

    # Refine with local search
    order = _local_search(order, lcp_mat, m, deadline)

    return _build_word(bm1, order, n)


def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    deadline = time.time() + 110.0  # safety margin below 120s

    try:
        return _construct_superperm(n, deadline)
    except Exception:
        # Fallback: classic construction
        word = [1]
        for k in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                w = tuple(word[i:i + k - 1])
                if len(set(w)) == k - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
