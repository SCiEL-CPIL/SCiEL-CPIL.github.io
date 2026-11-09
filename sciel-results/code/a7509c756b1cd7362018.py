"""
Four-colourings of K_n with no monochromatic C6 (maximise n).

Structural approach: depth-first backtracking that assigns the edges of a new
vertex v (to vertices 0..v-1) one by one.  Before assigning, we precompute for
the current subgraph on {0..v-1} a table Path4[c][i][d] meaning "there exists a
simple path of length 4 between i and d in colour c using only vertices < v".
Adding edge (i, v) in colour c is forbidden exactly when some earlier neighbour
d of v already has colour c on (d, v) AND Path4[c][i][d] holds, because that
path of length 4 plus edges (i, v) and (d, v) forms a 6-cycle.  This makes each
colour check O(deg) rather than O(n^4), letting the search reach much larger n
within the same wall-clock budget.
"""
import time
import random
import sys

sys.setrecursionlimit(100000)

DEADLINE = 118.0
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _precompute_path4(v, adj_list):
    """Path4[c][i][d] = simple path of length 4 between i,d in colour c,
    using only vertices in {0..v-1}."""
    Path4 = [[[False] * v for _ in range(v)] for _ in range(4)]
    for c in range(4):
        for i in range(v):
            if not adj_list[c][i]:
                continue
            row = Path4[c][i]
            for a in adj_list[c][i]:
                for b in adj_list[c][a]:
                    if b == i or b == a:
                        continue
                    for e in adj_list[c][b]:
                        if e == i or e == a or e == b:
                            continue
                        for d in adj_list[c][e]:
                            if d == i or d == a or d == b or d == e:
                                continue
                            row[d] = True
    return Path4


def _backtrack(n_target, deadline, seed):
    """Find a valid 4-colouring of K_{n_target} or return None."""
    random.seed(seed)
    adj = [[-1] * n_target for _ in range(n_target)]
    adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
    node_count = [0]
    deadline_local = [deadline]

    def _search(v):
        if time.time() > deadline_local[0]:
            return False
        if v == n_target:
            return True
        if v == 0:
            return _search(v + 1)

        Path4 = _precompute_path4(v, adj_list)

        # Build assignment list: edges (i, v) for i in 0..v-1, in random order.
        order = list(range(v))
        random.shuffle(order)
        assign_idx = [0]
        v_neigh_c = [[0, 0, 0, 0] for _ in range(v)]  # count of colour c neighbours of v

        def _assign(pos):
            if time.time() > deadline_local[0]:
                return False
            node_count[0] += 1
            if pos == v:
                return _search(v + 1)

            i = order[pos]
            for c in range(4):
                # forbidden if some earlier neighbour d of v already has colour c
                # and there is a length-4 path i..d in colour c
                valid = True
                for d in range(i):
                    if adj[d][v] == c and Path4[c][i][d]:
                        valid = False
                        break
                if not valid:
                    continue

                adj[i][v] = c
                adj[v][i] = c
                adj_list[c][i].add(v)
                adj_list[c][v].add(i)
                v_neigh_c[i][c] += 1

                if _assign(pos + 1):
                    return True

                adj[i][v] = -1
                adj[v][i] = -1
                adj_list[c][i].remove(v)
                adj_list[c][v].remove(i)
                v_neigh_c[i][c] -= 1

            return False

        return _assign(0)

    if _search(0):
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(adj[i][j])
        return colors
    return None


def _verify(colors, n):
    """Direct check: no colour class contains a C6."""
    if n < 6:
        return True
    adj = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1

    adj_list = [[set() for _ in range(n)] for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = adj[i][j]
            adj_list[c][i].add(j)
            adj_list[c][j].add(i)

    # A C6 exists iff there is an edge (u, w) in colour c and a simple path of
    # length 5 between u and w in colour c avoiding the edge (u, w).
    # Equivalently: for some vertex, a length-4 path between two of its
    # neighbours in the same colour.
    for c in range(4):
        for x in range(n):
            nbrs = adj_list[c][x]
            if len(nbrs) < 2:
                continue
            # enumerate pairs (a, b) of neighbours, check length-4 path a..b
            # using only vertices other than x.
            lst = list(nbrs)
            for ii in range(len(lst)):
                a = lst[ii]
                for jj in range(ii + 1, len(lst)):
                    b = lst[jj]
                    # path a - p - q - r - b, all distinct, none == x
                    found = False
                    for p in adj_list[c][a]:
                        if p == x or p == b:
                            continue
                        for q in adj_list[c][p]:
                            if q == x or q == a or q == b or q == p:
                                continue
                            for r in adj_list[c][q]:
                                if r == x or r == a or r == b or r == p or r == q:
                                    continue
                                if b in adj_list[c][r]:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        return False
    return True


def construct():
    global START_TIME
    START_TIME = time.time()

    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    # Try to reach larger n via backtracking.
    for target_n in range(10, 21):
        if _time_left() < 12.0:
            break

        found = False
        seed = 12345 + target_n * 7919
        while not found and _time_left() > 8.0:
            colors = _backtrack(target_n, time.time() + _time_left() - 8.0, seed)
            if colors is not None:
                if _verify(colors, target_n):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
                else:
                    # shouldn't happen; retry with a different seed
                    seed += 1
                    continue
            # backtracking exhausted or timed out for this seed
            seed += 1

    # Final safety validation.
    if best_colors is not None and best_n >= 1 and not _verify(best_colors, best_n):
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    return {'n': best_n, 'colors': best_colors}
