"""Structural redesign: randomized multi-start + 2-opt + two-ended extension."""
import random
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

DEADLINE = time.time() + 50.0  # safety margin under 60s


def two_opt_extend(path, rng):
    """Iteratively try 2-opt flips that keep the path valid, then extend both ends.
    A 2-opt flip reverses a middle segment: path = A + B, becomes A + reverse(B).
    Validity requires edge(A[-1], B[-1]) and edge(A[0], B[0])... actually for a path
    A=[a0..ak], B=[b0..bm], current edges ak-b0 and a0-bm. After reverse: A + B_rev,
    new edges ak-bm (must exist) and a0-b0 (must exist)."""
    improved = True
    while improved:
        improved = False
        n = len(path)
        if n < 4:
            break
        # Try all pairs (i, j) with 1 <= i < j <= n-1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # path[i..j] is the segment to reverse
                # Before: ... path[i-1] - path[i] ... path[j] - path[j+1] ...
                # After:  ... path[i-1] - path[j] ... path[i] - path[j+1] ...
                a = path[i - 1]
                b = path[i]
                c = path[j]
                d = path[j + 1] if j + 1 < n else None
                # Need edge(a, c) and (if d exists) edge(b, d)
                if c not in ADJ[a]:
                    continue
                if d is not None and d not in ADJ[b]:
                    continue
                # Also need the reversed segment to be internally valid (it is, since it was valid)
                # Apply the flip
                path[i:j + 1] = path[i:j + 1][::-1]
                improved = True
                # After flip, try to extend both ends
                _extend_both_ends(path)
                if time.time() > DEADLINE:
                    return path
    return path


def _extend_both_ends(path):
    """Extend path from both ends greedily, picking the more constrained end first."""
    changed = True
    while changed:
        changed = False
        # Compute available extensions at each end
        head_avail = ADJ[path[0]] - set(path)
        tail_avail = ADJ[path[-1]] - set(path)
        # Pick the end with fewer options (more constrained) to reduce dead-end risk
        if len(head_avail) == 0 and len(tail_avail) == 0:
            break
        if len(head_avail) <= len(tail_avail) and len(head_avail) > 0:
            # Extend from head
            v = min(head_avail, key=lambda w: (len(ADJ[w] - set(path)), w))
            path.insert(0, v)
            changed = True
        elif len(tail_avail) > 0:
            v = min(tail_avail, key=lambda w: (len(ADJ[w] - set(path)), w))
            path.append(v)
            changed = True
    return path


def greedy_path(start, rng):
    """Greedy path from start with random tie-breaking."""
    path = [start]
    seen = {start}
    while True:
        avail = ADJ[path[-1]] - seen
        if not avail:
            break
        # Weight by degree (prefer low-degree to avoid dead-ends) with randomness
        candidates = list(avail)
        # Sort by (degree, vertex) but add randomness
        rng.shuffle(candidates)
        candidates.sort(key=lambda w: (len(ADJ[w] - seen), w))
        v = candidates[0]
        path.append(v)
        seen.add(v)
    return path


def construct():
    t0 = time.time()
    rng = random.Random(748)
    best = []

    # Phase 1: Deterministic greedy from every start (like seed, but also try reverse)
    for start in range(N):
        if time.time() > DEADLINE:
            break
        path = greedy_path(start, rng)
        _extend_both_ends(path)
        if len(path) > len(best):
            best = list(path)

    # Phase 2: Multi-restart randomized greedy + 2-opt + extension
    for iteration in range(500):
        if time.time() > DEADLINE:
            break
        start = rng.randint(0, N - 1)
        path = greedy_path(start, rng)
        # Apply 2-opt improvement
        path = two_opt_extend(path, rng)
        # Extend both ends
        _extend_both_ends(path)
        if len(path) > len(best):
            best = list(path)

    # Phase 3: Try to extend the best path further with random restarts on the complement
    if len(best) < N:
        if time.time() > DEADLINE:
            return best
        # Try inserting vertices into gaps of the best path
        used = set(best)
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            # Pick a random unused vertex and try to insert it
            unused = [v for v in range(N) if v not in used]
            if not unused:
                break
            v = rng.choice(unused)
            # Try to insert v between consecutive pairs in best
            inserted = False
            for i in range(len(best) - 1):
                a, b = best[i], best[i + 1]
                if v in ADJ[a] and v in ADJ[b]:
                    new_path = best[:i + 1] + [v] + best[i + 1:]
                    if len(new_path) > len(best):
                        best = new_path
                        used.add(v)
                        inserted = True
                        break
            if not inserted:
                # Try to prepend or append
                if v in ADJ[best[0]]:
                    best = [v] + best
                    used.add(v)
                elif v in ADJ[best[-1]]:
                    best = best + [v]
                    used.add(v)

    # Final extension
    _extend_both_ends(best)

    if not best:
        # Fallback: return a single vertex
        best = [0]

    return best
