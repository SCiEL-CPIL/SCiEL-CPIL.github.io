"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 50  # safety margin

    # Precompute GCD table for O(1) lookups
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute all bad triples: bad_triple[a][b] = set of c such that {a,b,c} is bad
    bad_triple = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[b][c] == gab and gcd_table[c][a] == gab:
                    bad_triple[a][b].add(c)
                    bad_triple[b][a].add(c)
                    bad_triple[a][c].add(b)
                    bad_triple[c][a].add(b)
                    bad_triple[b][c].add(a)
                    bad_triple[c][b].add(a)

    # Count conflicts for each number
    conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            cnt = len(bad_triple[a][b])
            conflict_count[a] += cnt
            conflict_count[b] += cnt

    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a bad triple."""
        sel = list(selected_set)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                if x in bad_triple[a][b]:
                    return False
        return True

    def can_add_fast(selected_list, x, bad_triple_local):
        """Faster can_add using precomputed list."""
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            row_a = bad_triple_local[a]
            for j in range(i + 1, n):
                b = selected_list[j]
                if x in row_a[b]:
                    return False
        return True

    def greedy_build(order, selected_set=None):
        """Greedy build following given order."""
        if selected_set is None:
            selected_set = set()
        selected_list = list(selected_set)
        for x in order:
            if x in selected_set:
                continue
            # Check if x can be added
            ok = True
            for i in range(len(selected_list)):
                a = selected_list[i]
                row_a = bad_triple[a]
                for j in range(i + 1, len(selected_list)):
                    b = selected_list[j]
                    if x in row_a[b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected_set.add(x)
                selected_list.append(x)
        return selected_set

    def local_search(selected_set, time_limit=5.0):
        """Local search with add, 1-swap, and 2-swap moves."""
        selected = set(selected_set)
        selected_list = list(selected)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element not in selected
            for x in range(1, N + 1):
                if x in selected:
                    continue
                ok = True
                for i in range(len(selected_list)):
                    a = selected_list[i]
                    row_a = bad_triple[a]
                    for j in range(i + 1, len(selected_list)):
                        b = selected_list[j]
                        if x in row_a[b]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    selected.add(x)
                    selected_list.append(x)
                    improved = True
                    break
            # If we can't add, try 1-swap: remove one, add another
            if not improved:
                for idx in range(len(selected_list)):
                    x = selected_list[idx]
                    # Build remaining list without x
                    remaining_list = selected_list[:idx] + selected_list[idx+1:]
                    found_swap = False
                    for y in range(1, N + 1):
                        if y in selected:
                            continue
                        ok = True
                        for i in range(len(remaining_list)):
                            a = remaining_list[i]
                            row_a = bad_triple[a]
                            for j in range(i + 1, len(remaining_list)):
                                b = remaining_list[j]
                                if y in row_a[b]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            selected.discard(x)
                            selected.add(y)
                            selected_list = remaining_list + [y]
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            # If still no improvement, try 2-swap: remove two, add two
            if not improved and time.time() - start < time_limit - 0.5:
                n_sel = len(selected_list)
                found_2swap = False
                for i in range(n_sel):
                    if time.time() - start >= time_limit - 0.5:
                        break
                    for j in range(i + 1, n_sel):
                        if time.time() - start >= time_limit - 0.5:
                            break
                        x1, x2 = selected_list[i], selected_list[j]
                        remaining_list = [v for k, v in enumerate(selected_list) if k != i and k != j]
                        # Try to find two elements to add
                        for y1 in range(1, N + 1):
                            if y1 in selected or found_2swap:
                                continue
                            # Check y1 can be added to remaining
                            ok1 = True
                            for ii in range(len(remaining_list)):
                                a = remaining_list[ii]
                                row_a = bad_triple[a]
                                for jj in range(ii + 1, len(remaining_list)):
                                    b = remaining_list[jj]
                                    if y1 in row_a[b]:
                                        ok1 = False
                                        break
                                if not ok1:
                                    break
                            if not ok1:
                                continue
                            temp_list = remaining_list + [y1]
                            for y2 in range(y1 + 1, N + 1):
                                if y2 in selected or found_2swap:
                                    continue
                                ok2 = True
                                for ii in range(len(temp_list)):
                                    a = temp_list[ii]
                                    row_a = bad_triple[a]
                                    for jj in range(ii + 1, len(temp_list)):
                                        b = temp_list[jj]
                                        if y2 in row_a[b]:
                                            ok2 = False
                                            break
                                    if not ok2:
                                        break
                                if ok2:
                                    selected.discard(x1)
                                    selected.discard(x2)
                                    selected.add(y1)
                                    selected.add(y2)
                                    selected_list = temp_list + [y2]
                                    improved = True
                                    found_2swap = True
                                    break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
        return selected

    best = set()

    # Multiple restarts with different orderings
    base_order = list(range(1, N + 1))

    # Try natural order first
    result = greedy_build(base_order)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)

    # Try conflict-ordered (fewest conflicts first)
    conflict_ordered = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
    result = greedy_build(conflict_ordered)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)

    # Try descending order
    desc_order = list(range(N, 0, -1))
    result = greedy_build(desc_order)
    result = local_search(result, time_limit=2.0)
    if len(result) > len(best):
        best = set(result)

    # Random restarts
    restart_count = 0
    no_improve_count = 0
    while time.time() < deadline:
        if random.random() < 0.4:
            # Perturb: start from best and randomly remove some, then rebuild
            start_set = set(best)
            for _ in range(random.randint(1, 4)):
                if start_set:
                    start_set.remove(random.choice(list(start_set)))
            remaining_order = [x for x in range(1, N + 1) if x not in start_set]
            random.shuffle(remaining_order)
            order = list(start_set) + remaining_order
            result = greedy_build(order, start_set)
        elif random.random() < 0.5:
            # Conflict-ordered with random perturbation
            order = list(conflict_ordered)
            for _ in range(random.randint(5, 15)):
                i, j = random.randint(0, N-1), random.randint(0, N-1)
                order[i], order[j] = order[j], order[i]
            result = greedy_build(order)
        else:
            random.shuffle(base_order)
            result = greedy_build(base_order)
        result = local_search(result, time_limit=2.0)
        if len(result) > len(best):
            best = set(result)
            no_improve_count = 0
        else:
            no_improve_count += 1
        restart_count += 1
        if no_improve_count > 80 and len(best) >= 80:
            break

    # Final polish: try to add any remaining elements (multiple passes)
    for _ in range(3):
        changed = False
        for x in range(1, N + 1):
            if x not in best:
                ok = True
                blist = list(best)
                for i in range(len(blist)):
                    a = blist[i]
                    row_a = bad_triple[a]
                    for j in range(i + 1, len(blist)):
                        b = blist[j]
                        if x in row_a[b]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best.add(x)
                    changed = True
        if not changed:
            break

    return list(best) if best else [1]
# EVOLVE-BLOCK-END
