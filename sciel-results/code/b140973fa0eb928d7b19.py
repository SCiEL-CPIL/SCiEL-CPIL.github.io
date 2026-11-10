import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin
    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        lcm_table[i][i] = i
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    num_bad = len(bad_triples)

    # For each element, list of triple indices it participates in
    elem_to_triples = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(i)
        elem_to_triples[b].append(i)
        elem_to_triples[c].append(i)

    # triple_elems: 1-indexed elements for each triple
    triple_elems = [(a, b, c) for (a, b, c) in bad_triples]

    # Helper: count new violations if we add element x (x not currently selected)
    def add_violations(sel, x):
        cnt = 0
        for ti in elem_to_triples[x]:
            a, b, c = triple_elems[ti]
            others = [v for v in (a, b, c) if v != x]
            if sel[others[0]] and sel[others[1]]:
                cnt += 1
        return cnt

    # ============================================================
    # PHASE 1: Set-covering greedy to get a valid solution
    # ============================================================
    selected = [False] * (n + 1)
    for x in range(1, n + 1):
        selected[x] = True

    # Active triples: all initially active
    active = set(range(num_bad))

    # Iteratively remove the element that hits the most active triples
    while active:
        if time.time() > deadline - 40:
            break
        best_x = None
        best_cov = 0
        for x in range(1, n + 1):
            if not selected[x]:
                continue
            cov = 0
            for ti in elem_to_triples[x]:
                if ti in active:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x is None or best_cov == 0:
            break
        selected[best_x] = False
        for ti in elem_to_triples[best_x]:
            active.discard(ti)

    # Now greedily add back elements that don't create violations
    changed = True
    while changed and time.time() < deadline - 35:
        changed = False
        for x in range(1, n + 1):
            if selected[x]:
                continue
            if add_violations(selected, x) == 0:
                selected[x] = True
                changed = True

    cur_size = sum(selected[1:])
    best_size = cur_size
    best_sel = selected[:]

    # ============================================================
    # PHASE 2: Local search - 1-remove, greedy-add
    # ============================================================
    def local_search_1remove(sel, cur_sz, max_iters=500):
        """Try removing one element and greedily adding back. Return improved sel and size."""
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            best_gain = 0
            best_move = None
            for r in range(1, n + 1):
                if not sel[r]:
                    continue
                # Simulate removing r
                sel[r] = False
                added = []
                for x in range(1, n + 1):
                    if not sel[x] and add_violations(sel, x) == 0:
                        sel[x] = True
                        added.append(x)
                gain = len(added) - 1
                if gain > best_gain:
                    best_gain = gain
                    best_move = (r, added)
                # Revert
                for x in added:
                    sel[x] = False
                sel[r] = True
            if best_move is not None and best_gain > 0:
                r, added = best_move
                sel[r] = False
                for x in added:
                    sel[x] = True
                cur_sz += best_gain
                improved = True
            else:
                break
        return sel, cur_sz, improved

    # Run local search
    sel, cur_size, improved = local_search_1remove(selected, cur_size, max_iters=200)
    if cur_size > best_size:
        best_size = cur_size
        best_sel = sel[:]

    # ============================================================
    # PHASE 3: 2-remove, greedy-add local search
    # ============================================================
    def local_search_2remove(sel, cur_sz, max_iters=50):
        """Try removing two elements and greedily adding back."""
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 8:
                break
            best_gain = 0
            best_move = None
            for r1 in range(1, n + 1):
                if not sel[r1]:
                    continue
                for r2 in range(r1 + 1, n + 1):
                    if not sel[r2]:
                        continue
                    sel[r1] = False
                    sel[r2] = False
                    added = []
                    for x in range(1, n + 1):
                        if not sel[x] and add_violations(sel, x) == 0:
                            sel[x] = True
                            added.append(x)
                    gain = len(added) - 2
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (r1, r2, added)
                    for x in added:
                        sel[x] = False
                    sel[r1] = True
                    sel[r2] = True
            if best_move is not None and best_gain > 0:
                r1, r2, added = best_move
                sel[r1] = False
                sel[r2] = False
                for x in added:
                    sel[x] = True
                cur_sz += best_gain
                improved = True
            else:
                break
        return sel, cur_sz, improved

    sel, cur_size, improved = local_search_2remove(selected, cur_size, max_iters=30)
    if cur_size > best_size:
        best_size = cur_size
        best_sel = sel[:]

    # ============================================================
    # PHASE 4: Random restarts with different orderings
    # ============================================================
    rng = random.Random(42)

    for restart in range(30):
        if time.time() > deadline - 10:
            break

        # Random ordering
        order = list(range(1, n + 1))
        rng.shuffle(order)

        sel = [False] * (n + 1)
        for x in order:
            if add_violations(sel, x) == 0:
                sel[x] = True

        # Greedy addition
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for x in range(1, n + 1):
                if not sel[x] and add_violations(sel, x) == 0:
                    sel[x] = True
                    changed = True

        sz = sum(sel[1:])
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]

        # Quick local search
        sel, sz, _ = local_search_1remove(sel, sz, max_iters=50)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]

        sel, sz, _ = local_search_2remove(sel, sz, max_iters=10)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]

    # ============================================================
    # PHASE 5: Simulated annealing on valid states
    # ============================================================
    if time.time() < deadline - 5:
        cur_sel = best_sel[:]
        cur_size = best_size
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        sa_rng = random.Random(999)

        for it in range(50000):
            if time.time() > deadline - 3:
                break
            if T < T_min:
                break

            # Random move: pick an element, try to flip
            x = sa_rng.randint(1, n)
            if cur_sel[x]:
                # Remove x (always valid)
                cur_sel[x] = False
                new_size = cur_size - 1
                # Try to add back elements
                added = []
                for y in range(1, n + 1):
                    if not cur_sel[y] and add_violations(cur_sel, y) == 0:
                        cur_sel[y] = True
                        added.append(y)
                new_size += len(added)
                delta = new_size - cur_size
                if delta >= 0 or sa_rng.random() < pow(2.718281828, delta / T):
                    cur_size = new_size
                    if cur_size > best_size:
                        best_size = cur_size
                        best_sel = cur_sel[:]
                else:
                    # Revert
                    cur_sel[x] = True
                    for y in added:
                        cur_sel[y] = False
            else:
                # Try adding x
                viol = add_violations(cur_sel, x)
                if viol == 0:
                    cur_sel[x] = True
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_sel = cur_sel[:]
                else:
                    # Would create violations - skip (can't add in invalid state)
                    pass

            T *= cooling

    # ============================================================
    # PHASE 6: Final polish - try to add any remaining elements
    # ============================================================
    for x in range(1, n + 1):
        if not best_sel[x] and add_violations(best_sel, x) == 0:
            best_sel[x] = True

    best_size = sum(best_sel[1:])

    # Convert to list of 1-indexed elements
    result = [x for x in range(1, n + 1) if best_sel[x]]
    if not result:
        result = [1]

    return result
