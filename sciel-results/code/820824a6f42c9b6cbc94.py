import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # safety margin before 330s kill
    all_points = [(x, y) for y in range(N) for x in range(N)]
    best = []

    def is_valid_add(selected, dists, q):
        # dists: dict point -> set of squared distances to other selected points
        qx, qy = q
        # Check q's distances to all selected are distinct
        seen = set()
        for p in selected:
            d2 = (qx - p[0]) ** 2 + (qy - p[1]) ** 2
            if d2 in seen:
                return False
            seen.add(d2)
            # Check p: would q create a duplicate distance for p?
            if d2 in dists[p]:
                return False
        return True

    def greedy_build(order):
        selected = []
        dists = {}
        for q in order:
            if is_valid_add(selected, dists, q):
                qx, qy = q
                selected.append(q)
                dists[q] = set()
                for p in selected[:-1]:
                    d2 = (qx - p[0]) ** 2 + (qy - p[1]) ** 2
                    dists[q].add(d2)
                    dists[p].add(d2)
        return selected

    # Multiple restarts with different random orderings
    for seed in range(20000):
        if time.time() > deadline:
            break
        random.seed(seed)
        candidates = all_points[:]
        random.shuffle(candidates)
        res = greedy_build(candidates)
        if len(res) > len(best):
            best = res

    # Also try a few structured orderings
    for ox in range(N):
        if time.time() > deadline:
            break
        # spiral-ish / row-major with offset
        order = [(x, y) for y in range(N) for x in range(N)]
        # rotate starting point
        start = (ox, 0)
        # just try row-major and column-major
        res = greedy_build(order)
        if len(res) > len(best):
            best = res
        order2 = [(x, y) for x in range(N) for y in range(N)]
        res2 = greedy_build(order2)
        if len(res2) > len(best):
            best = res2

    return best
# EVOLVE-BLOCK-END
