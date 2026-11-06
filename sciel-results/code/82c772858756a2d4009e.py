# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start = time.time()
    TIME_LIMIT = 85.0

    # Precompute gcd table
    gt = [[0]*129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            gt[i][j] = gcd(i, j)
            gt[j][i] = gt[i][j]

    # conflict[x] = frozenset of pairs (y,z) with y<z, y!=x, z!=x, such that (x,y,z) forbidden
    conflict = [set() for _ in range(129)]
    for a in range(1, 129):
        for b in range(a+1, 129):
            gab = gt[a][b]
            for c in range(b+1, 129):
                if gt[a][c] == gab and gt[b][c] == gab:
                    conflict[a].add((b, c))
                    conflict[b].add((a, c))
                    conflict[c].add((a, b))

    def can_add(x, s):
        for y, z in conflict[x]:
            if y in s and z in s:
                return False
        return True

    def greedy(order=None):
        s = set()
        if order is None:
            order = list(range(1, 129))
            random.shuffle(order)
        for x in order:
            if can_add(x, s):
                s.add(x)
        return s

    best = set()
    best_size = 0

    # Phase 1: Many greedy restarts
    for r in range(200):
        if time.time() - start > TIME_LIMIT - 30:
            break
        s = greedy()
        if len(s) > best_size:
            best = s
            best_size = len(s)

    # Phase 2: Local search (add + swap)
    s = set(best)
    stuck_count = 0
    for it in range(200000):
        if time.time() - start > TIME_LIMIT:
            break

        improved = False
        # Try to add
        cands = [x for x in range(1, 129) if x not in s]
        random.shuffle(cands)
        for x in cands:
            if can_add(x, s):
                s.add(x)
                improved = True
                break

        if not improved:
            # Try swap
            sl = list(s)
            random.shuffle(sl)
            for xrm in sl:
                if time.time() - start > TIME_LIMIT:
                    break
                st = s - {xrm}
                for xad in cands:
                    if xad in s:
                        continue
                    if can_add(xad, st):
                        s = st | {xad}
                        improved = True
                        break
                if improved:
                    break

        if len(s) > best_size:
            best = set(s)
            best_size = len(s)
            stuck_count = 0
        elif not improved:
            stuck_count += 1
            if stuck_count > 200:
                s = greedy()
                stuck_count = 0

    result = list(best)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
