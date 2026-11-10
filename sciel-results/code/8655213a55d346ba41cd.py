import time
import random
import math
from collections import defaultdict
from itertools import permutations

def _overlap(a, b):
    """Max k in [0, n-1] with a[n-k:] == b[:k]."""
    n = len(a)
    for k in range(n - 1, -1, -1):
        if k == 0:
            return 0
        if a[n - k:] == b[:k]:
            return k
    return 0


def _edge_cost(a, b):
    """Cost to append b after a = n - overlap."""
    return len(a) - _overlap(a, b)


def _build_sparse_graph(n):
    """Return (perm_list, perm_index, neighbors) where neighbors[i] is a
    list of (j, cost) with cost <= n-2 (i.e. overlap >= 2)."""
    perm_list = list(permutations(range(1, n + 1)))
    perm_index = {p: i for i, p in enumerate(perm_list)}
    neighbors = [[] for _ in range(len(perm_list))]

    def add_edge(i, j, c):
        if j != i and c <= n - 2:
            neighbors[i].append((j, c))

    for i, p in enumerate(perm_list):
        # suffix variants: drop the last k symbols (k = 1..n-1)
        for k in range(1, n):
            q = p[:-k]
            add_edge(i, perm_index[q], k)
        # prefix variants: drop the first k symbols (k = 1..n-1)
        for k in range(1, n):
            q = p[k:]
            add_edge(i, perm_index[q], k)

    return perm_list, perm_index, neighbors


def _cost_between(i, j, perm_list, neighbors, n):
    for (k, c) in neighbors[i]:
        if k == j:
            return c
    return n - _overlap(perm_list[i], perm_list[j])


def _greedy_path(perm_list, perm_index, neighbors, n):
    """Greedy nearest-neighbour Hamiltonian path."""
    N = len(perm_list)
    used = bytearray(N)
    path = []
    cur = 0
    for _ in range(N):
        used[cur] = 1
        path.append(cur)
        best = -1
        bestc = n + 1
        for (j, c) in neighbors[cur]:
            if not used[j] and c < bestc:
                bestc = c
                best = j
        if best < 0:
            # fallback: cheapest unused (rare)
            for j in range(N):
                if not used[j]:
                    c = _cost_between(cur, j, perm_list, neighbors, n)
                    if c < bestc:
                        bestc = c
                        best = j
        cur = best
    return path


def _recompute_path_cost(path, perm_list, neighbors, n):
    total = 0
    for a, b in zip(path, path[1:]):
        total += _cost_between(a, b, perm_list, neighbors, n)
    return total


def _two_opt_improve(path, perm_list, neighbors, n, deadline):
    """2-opt: reverse a segment to reduce total cost."""
    N = len(path)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(N - 2):
            if time.time() > deadline:
                break
            for j in range(i + 2, N):
                # delta cost of reversing path[i..j]
                a = path[i - 1] if i > 0 else None
                b = path[i]
                c = path[j]
                d = path[j + 1] if j + 1 < N else None
                cost_b_c = _cost_between(b, c, perm_list, neighbors, n)
                if a is None and d is None:
                    new = _cost_between(c, b, perm_list, neighbors, n)
                    if new < cost_b_c:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True
                elif a is None:
                    new = _cost_between(c, b, perm_list, neighbors, n) + \
                          _cost_between(c, d, perm_list, neighbors, n)
                    if new < cost_b_c:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True
                elif d is None:
                    new = _cost_between(a, c, perm_list, neighbors, n) + \
                          _cost_between(c, b, perm_list, neighbors, n)
                    if new < cost_b_c:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True
                else:
                    new = _cost_between(a, c, perm_list, neighbors, n) + \
                          _cost_between(c, b, perm_list, neighbors, n) + \
                          _cost_between(c, d, perm_list, neighbors, n)
                    if new < cost_b_c:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True
                if improved:
                    break
    return path


def _build_superperm(n):
    """Standard constructive superpermutation (fallback / n<=4)."""
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            if not new:
                new = block
            else:
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, 0, -1):
                    if new[-o:] == block[:o]:
                        ov = o
                        break
                new.extend(block[ov:])
        word = new
    return word


def _greedy_delete(word, n, deadline):
    """Greedy deletion local search."""
    L = len(word)
    if L < n:
        return word
    perm_counts = defaultdict(int)
    for i in range(L - n + 1):
        perm_counts[tuple(word[i:i + n])] += 1

    def can_remove(i):
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        for start in range(start_min, start_max + 1):
            if perm_counts[tuple(word[start:start + n])] == 1:
                return False
        return True

    def remove_at(i):
        nonlocal L
        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)
        for start in range(start_min, start_max + 1):
            perm_counts[tuple(word[start:start + n])] -= 1
        word.pop(i)
        L -= 1
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n)
        for start in range(new_start_min, new_start_max + 1):
            perm_counts[tuple(word[start:start + n])] += 1

    while True:
        if time.time() > deadline:
            break
        removed_any = False
        i = 0
        while i < L:
            if can_remove(i):
                remove_at(i)
                removed_any = True
                break
            else:
                i += 1
        if not removed_any:
            break
    return word


def _verify(word, n):
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
    return len(seen) == math.factorial(n)


def _construct_via_graph(n, deadline):
    perm_list, perm_index, neighbors = _build_sparse_graph(n)
    N = len(perm_list)

    # Greedy start
    path = _greedy_path(perm_list, perm_index, neighbors, n)

    # 2-opt improvement (time-bounded)
    two_opt_deadline = min(deadline - 5.0, time.time() + 30.0)
    if time.time() < two_opt_deadline:
        _two_opt_improve(path, perm_list, neighbors, n, two_opt_deadline)

    # Reconstruct word
    word = list(perm_list[path[0]])
    for k in range(1, N):
        ov = _overlap(tuple(word[-n:]), perm_list[path[k]])
        word.extend(list(perm_list[path[k]][ov:]))

    return word


def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"

    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s

    # For small n the standard construction is already optimal/short enough.
    if n <= 4:
        word = _build_superperm(n)
    else:
        # Use the sparse-graph minimum-weight Hamiltonian path approach.
        try:
            word = _construct_via_graph(n, deadline)
        except Exception:
            word = _build_superperm(n)

    # Final greedy deletion pass (time-bounded).
    if time.time() < deadline - 3.0:
        word = _greedy_delete(word, n, deadline - 3.0)

    # Safety: if somehow invalid, fall back to standard construction.
    if not _verify(word, n):
        word = _build_superperm(n)
        if time.time() < deadline - 3.0:
            word = _greedy_delete(word, n, deadline - 3.0)

    return "".join(map(str, word))
