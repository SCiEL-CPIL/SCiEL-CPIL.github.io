import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---- Row regeneration (core structural move) ----
    def gen_row(used, k_val, randomize, p_smallest=0.8, window=4, rng=None, cap=400):
        """Generate one row of k_val+1 entries starting at 0, avoiding used diffs."""
        if rng is None:
            rng = random
        row = [0]
        cur = set()  # current row's internal diffs
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            forbidden = used | cur
            while x < cap:
                new_d = {x - a for a in row}
                if not (new_d & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            if not candidates:
                return None
            if randomize:
                if rng.random() < p_smallest:
                    nxt = candidates[0]
                else:
                    nxt = rng.choice(candidates)
            else:
                nxt = candidates[0]
            row.append(nxt)
            cur |= {nxt - a for a in row[:-1]}
        return row

    # ---- Deterministic tightening: pull each element down ----
    def tighten(rows, max_iter=200):
        """Pull each element (right to left, row by row) down to its smallest valid value."""
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    # Try to reduce row[pos] to row[pos-1]+1 upward
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    # Check if target works (internal + cross-row)
                    # Build internal diffs with candidate
                    cand = row[:pos] + [target] + row[pos + 1:]
                    # monotonic check
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # internal distinctness
                    idiffs = set()
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in idiffs:
                                ok = False
                                break
                            idiffs.add(d)
                        if not ok:
                            break
                    if not ok:
                        continue
                    # cross-row check
                    for idx in range(n_rows):
                        if idx == r:
                            continue
                        od = row_diffs(rows[idx])
                        if idiffs & od:
                            ok = False
                            break
                    if ok:
                        row[pos] = target
                        improved = True
            if not improved:
                break
        return rows

    # ---- Greedy initial solution with symmetry breaking ----
    def greedy_initial(rng):
        used = set()
        rows = []
        prev_first = 0
        for r in range(n_rows):
            row = None
            attempts = 0
            while row is None and attempts < 200:
                attempts += 1
                row = gen_row(used, k, randomize=True, p_smallest=0.85,
                              window=3, rng=rng, cap=400)
                if row is not None:
                    # symmetry breaking: row[1] > prev_first
                    if r > 0 and row[1] <= prev_first:
                        row = None
                        continue
            if row is None:
                # deterministic fallback
                row = gen_row(used, k, randomize=False, rng=rng, cap=500)
            rows.append(row)
            prev_first = row[1]
            used |= row_diffs(row)
        return rows

    # ---- Backtracking search for a specific target scope ----
    def backtracking_search(target_scope, rng, max_nodes=50000):
        """
        Attempts to construct n_rows of k+1 integers starting at 0,
        strictly increasing, max element <= target_scope,
        with all pairwise differences distinct across all rows.
        Returns list of rows or None.
        """
        used_diffs = set()
        rows = []
        nodes_visited = 0
        
        def build_row(row_idx, pos, current_row, current_row_diffs):
            nonlocal nodes_visited
            if time.time() > deadline - 5:
                return False
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False

            if pos == k + 1:
                rows.append(list(current_row))
                used_diffs.update(current_row_diffs)
                return True

            # Pruning: ensure we can fit remaining elements
            remaining = k + 1 - pos
            max_possible_x = target_scope - (remaining - 1)
            
            start_x = current_row[-1] + 1 if current_row else 1
            if start_x > max_possible_x:
                return False

            # Determine candidates
            # We prioritize smaller x to keep scope low, but allow some randomness
            # to escape local minima.
            candidates = []
            for x in range(start_x, max_possible_x + 1):
                new_d = {x - a for a in current_row}
                if not (new_d & used_diffs) and not (new_d & current_row_diffs):
                    candidates.append(x)
                # Limit candidates to avoid huge branching factor
                if len(candidates) > 20:
                    break

            if not candidates:
                return False

            # Randomize order: mix of smallest and random
            # To improve search, we try a mix: 50% smallest, 50% random from top 10
            if rng.random() < 0.5:
                # Try smallest first
                order = candidates
            else:
                # Try a random subset of small candidates
                top_candidates = candidates[:min(10, len(candidates))]
                rng.shuffle(top_candidates)
                order = top_candidates

            for x in order:
                new_d = {x - a for a in current_row}
                new_row = current_row + [x]
                new_diffs = current_row_diffs | new_d
                
                # Recursive step: try to build this row further
                if build_row(row_idx, pos + 1, new_row, new_diffs):
                    return True
                
                # Backtrack: remove x and its diffs
                # Note: current_row and current_row_diffs are immutable in this call context effectively
                # We pass new_row and new_diffs to the next call, so we just return to the loop
                
            return False

        # Symmetry breaking: enforce row[i][1] > row[i-1][1]
        # We will handle this by iterating rows and passing constraints if needed, 
        # but for simplicity in the first pass, we just build rows sequentially.
        # A more advanced version would pass 'prev_first' to restrict candidates.
        
        for r in range(n_rows):
            # Reset nodes for each row to allow more time for later rows? 
            # No, global node count is better for fairness.
            if not build_row(r, 0, [0], set()):
                return None
        
        if len(rows) != n_rows:
            return None
            
        return rows

    # ---- Simulated annealing with row regeneration ----
    def anneal(rows, rng, max_iter=20000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 8.0
        cooling = 0.998
        min_T = 0.3

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = row_diffs(old_row)
            used -= old_diffs

            # randomize parameters
            p_smallest = 0.6 + 0.3 * rng.random()
            window = 2 + rng.randint(0, 3)
            new_row = gen_row(used, k, randomize=True, p_smallest=p_smallest,
                              window=window, rng=rng, cap=300)

            if new_row is None:
                used |= old_diffs
                continue

            new_diffs = row_diffs(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max

            if delta < 0:
                rows[r_idx] = new_row
                used |= new_diffs
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used |= new_diffs
                    current_max = new_global
                else:
                    used |= old_diffs

            T = max(min_T, T * cooling)

        return best_rows

    # ---- Local element perturbation (fine-tuning) ----
    def local_refine(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        other_diffs = [set() for _ in range(n_rows)]
        for idx in range(n_rows):
            other_diffs[idx] = row_diffs(rows[idx])
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            # try to move element down or up slightly
            deltas = [-1, -2, 1, 2, -3, 3, -4, 4]
            rng.shuffle(deltas)
            for delta in deltas:
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                test_row = row[:]
                test_row[pos] = new_val
                idiffs = set()
                ok = True
                for i in range(len(test_row)):
                    ri = test_row[i]
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - ri
                        if d in idiffs:
                            ok = False
                            break
                        idiffs.add(d)
                    if not ok:
                        break
                if not ok:
                    continue
                # cross-row
                conflict = False
                for idx in range(n_rows):
                    if idx == r_idx:
                        continue
                    if idiffs & other_diffs[idx]:
                        conflict = True
                        break
                if conflict:
                    continue
                # accept if it improves or keeps scope
                other_max = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row[-1])
                if new_scope <= current_max:
                    rows[r_idx] = test_row
                    other_diffs[r_idx] = idiffs
                    current_max = new_scope
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break  # one change per iteration
        return best_rows

    # ---- Gap-exchange local search: swap gaps between rows ----
    def gap_exchange_search(rows, rng, max_iter=3000):
        """
        Represent each row by its k consecutive gaps (differences between
        consecutive elements). A move swaps one gap from row A with one gap
        from row B, then re-validates. This explores structural reconfigurations
        that single-element shifts cannot reach.
        """
        rows = [list(r) for r in rows]
        other_diffs = [set() for _ in range(n_rows)]
        for idx in range(n_rows):
            other_diffs[idx] = row_diffs(rows[idx])
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        def get_gaps(row):
            return [row[i + 1] - row[i] for i in range(len(row) - 1)]

        def set_gaps(row, gaps):
            new_row = [0]
            for g in gaps:
                new_row.append(new_row[-1] + g)
            return new_row

        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            if n_rows < 2:
                break

            # Pick two distinct rows
            r1, r2 = rng.sample(range(n_rows), 2)
            gaps1 = get_gaps(rows[r1])
            gaps2 = get_gaps(rows[r2])

            if len(gaps1) != k or len(gaps2) != k:
                continue

            # Pick a gap position from each row
            p1 = rng.randint(0, k - 1)
            p2 = rng.randint(0, k - 1)

            # Swap the gaps
            g1 = gaps1[p1]
            g2 = gaps2[p2]
            if g1 == g2:
                continue

            new_gaps1 = list(gaps1)
            new_gaps2 = list(gaps2)
            new_gaps1[p1] = g2
            new_gaps2[p2] = g1

            # Build new rows
            new_row1 = set_gaps(rows[r1], new_gaps1)
            new_row2 = set_gaps(rows[r2], new_gaps2)

            # Validate new_row1: strictly increasing (gaps are positive so OK)
            # Check internal distinctness of diffs
            idiffs1 = set()
            ok1 = True
            for i in range(len(new_row1)):
                ri = new_row1[i]
                for j in range(i + 1, len(new_row1)):
                    d = new_row1[j] - ri
                    if d in idiffs1:
                        ok1 = False
                        break
                    idiffs1.add(d)
                if not ok1:
                    break
            if not ok1:
                continue

            idiffs2 = set()
            ok2 = True
            for i in range(len(new_row2)):
                ri = new_row2[i]
                for j in range(i + 1, len(new_row2)):
                    d = new_row2[j] - ri
                    if d in idiffs2:
                        ok2 = False
                        break
                    idiffs2.add(d)
                if not ok2:
                    break
            if not ok2:
                continue

            # Cross-row conflict check: new_row1 vs all other rows (except r2), new_row2 vs all (except r1)
            conflict = False
            for idx in range(n_rows):
                if idx == r1 or idx == r2:
                    continue
                if idiffs1 & other_diffs[idx] or idiffs2 & other_diffs[idx]:
                    conflict = True
                    break
            if conflict:
                continue
            # Cross conflict between the two new rows
            if idiffs1 & idiffs2:
                continue

            # Compute new scope
            other_max = 0
            for i in range(n_rows):
                if i != r1 and i != r2:
                    other_max = max(other_max, rows[i][-1])
            new_scope = max(other_max, new_row1[-1], new_row2[-1])

            if new_scope < current_max:
                rows[r1] = new_row1
                rows[r2] = new_row2
                other_diffs[r1] = idiffs1
                other_diffs[r2] = idiffs2
                current_max = new_scope
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            elif new_scope == current_max:
                # Accept with some probability to escape local minima
                if rng.random() < 0.3:
                    rows[r1] = new_row1
                    rows[r2] = new_row2
                    other_diffs[r1] = idiffs1
                    other_diffs[r2] = idiffs2

        return best_rows

    # ---- 2-element swap perturbation ----
    def two_elem_swap(rows, rng, max_iter=3000):
        """Swap two elements within a row (preserving order by re-sorting)
        or swap an element from one row with one from another."""
        rows = [list(r) for r in rows]
        other_diffs = [set() for _ in range(n_rows)]
        for idx in range(n_rows):
            other_diffs[idx] = row_diffs(rows[idx])
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            if len(row) < 3:
                continue
            # Pick two positions to swap
            p1, p2 = rng.sample(range(1, len(row)), 2)
            # Ensure p1 < p2 for clarity
            if p1 > p2:
                p1, p2 = p2, p1

            # Swap values
            test_row = row[:]
            test_row[p1], test_row[p2] = test_row[p2], test_row[p1]

            # Must remain strictly increasing: after swap, check
            ok = True
            for i in range(1, len(test_row)):
                if test_row[i] <= test_row[i - 1]:
                    ok = False
                    break
            if not ok:
                continue

            idiffs = set()
            for i in range(len(test_row)):
                ri = test_row[i]
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - ri
                    if d in idiffs:
                        ok = False
                        break
                    idiffs.add(d)
                if not ok:
                    break
            if not ok:
                continue

            conflict = False
            for idx in range(n_rows):
                if idx == r_idx:
                    continue
                if idiffs & other_diffs[idx]:
                    conflict = True
                    break
            if conflict:
                continue

            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_scope = max(other_max, test_row[-1])

            if new_scope < current_max:
                rows[r_idx] = test_row
                other_diffs[r_idx] = idiffs
                current_max = new_scope
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            elif new_scope == current_max:
                if rng.random() < 0.2:
                    rows[r_idx] = test_row
                    other_diffs[r_idx] = idiffs

        return best_rows

    # ---- Main search ----
    rng = random.Random(42)

    # 1. Initial solution
    best_rows = greedy_initial(rng)
    if best_rows is None or not is_valid(best_rows):
        # absolute fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = gen_row(used, k, randomize=False, rng=rng, cap=600)
            best_rows.append(row)
            used |= row_diffs(row)

    # 2. Tighten
    best_rows = tighten(best_rows)
    if not is_valid(best_rows):
        best_rows = greedy_initial(rng)
    best_scope = scope_of(best_rows)

    # 3. Anneal from initial
    rows = anneal(best_rows, rng, max_iter=15000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 4. Local refine
    rows = local_refine(best_rows, rng, max_iter=4000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 4b. Gap-exchange search
    rows = gap_exchange_search(best_rows, rng, max_iter=2000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 4c. Two-element swap perturbation
    rows = two_elem_swap(best_rows, rng, max_iter=2000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 5. Tighten again after local refine
    best_rows = tighten(best_rows)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)

    # 6. Targeted Backtracking Search
    # Try to find a solution with scope < best_scope
    # We decrement the target scope and try to find a valid configuration
    target = best_scope - 1
    # Limit the number of backtracking attempts to ensure we finish in time
    max_bt_attempts = 50
    bt_attempts = 0
    
    while target > 100 and bt_attempts < max_bt_attempts and time.time() < deadline - 30:
        bt_attempts += 1
        rng_bt = random.Random(10000 + bt_attempts * 7919)
        
        # Allocate node budget based on remaining time
        time_left = deadline - 30 - time.time()
        # Estimate nodes per second roughly (conservative)
        est_nodes = int(time_left * 1000) 
        if est_nodes < 10000:
            est_nodes = 10000
        if est_nodes > 100000:
            est_nodes = 100000
            
        sol = backtracking_search(target, rng_bt, max_nodes=est_nodes)
        
        if sol and is_valid(sol):
            new_scope = scope_of(sol)
            if new_scope < best_scope:
                best_rows = sol
                best_scope = new_scope
                # If we found a better solution, tighten it
                best_rows = tighten(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
                # Try to improve further with local refine
                rows_ref = local_refine(best_rows, rng_bt, max_iter=2000)
                if is_valid(rows_ref):
                    s_ref = scope_of(rows_ref)
                    if s_ref < best_scope:
                        best_rows = rows_ref
                        best_scope = s_ref
                # Continue trying lower targets
                target = best_scope - 1
            else:
                # If valid but not better, still update target? 
                # No, we want strictly better.
                target -= 1
        else:
            # Failed to find solution for this target, try next lower target
            target -= 1

    # 7. Diversified restarts: fresh greedy + anneal
    for trial in range(8):
        if time.time() > deadline - 25:
            break
        rng2 = random.Random(1000 + trial * 137)
        rows = greedy_initial(rng2)
        if rows is None or not is_valid(rows):
            continue
        rows = tighten(rows)
        if not is_valid(rows):
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        # anneal this diverse start
        rows2 = anneal(rows, rng2, max_iter=8000)
        if is_valid(rows2):
            s2 = scope_of(rows2)
            if s2 < best_scope:
                best_rows = rows2
                best_scope = s2
        # local refine
        rows3 = local_refine(rows2 if is_valid(rows2) else rows, rng2, max_iter=3000)
        if is_valid(rows3):
            s3 = scope_of(rows3)
            if s3 < best_scope:
                best_rows = rows3
                best_scope = s3
        # gap-exchange on this diverse start
        rows4 = gap_exchange_search(rows3 if is_valid(rows3) else rows, rng2, max_iter=1500)
        if is_valid(rows4):
            s4 = scope_of(rows4)
            if s4 < best_scope:
                best_rows = rows4
                best_scope = s4
        # two-element swap
        rows5 = two_elem_swap(rows4 if is_valid(rows4) else rows3, rng2, max_iter=1500)
        if is_valid(rows5):
            s5 = scope_of(rows5)
            if s5 < best_scope:
                best_rows = rows5
                best_scope = s5

    # 8. Final high-precision polish: exhaustive small-perturbation search
    if time.time() < deadline - 5:
        polish_rows = [list(r) for r in best_rows]
        polish_max = scope_of(polish_rows)
        # Multiple passes of tightening + local refine + gap exchange
        for polish_pass in range(4):
            if time.time() > deadline - 5:
                break
            polish_rows = tighten(polish_rows, max_iter=100)
            if is_valid(polish_rows):
                polish_max = scope_of(polish_rows)
            rng_polish = random.Random(99999 + polish_pass * 31)
            polish_rows = local_refine(polish_rows, rng_polish, max_iter=1500)
            if is_valid(polish_rows):
                polish_max = scope_of(polish_rows)
            # Gap exchange in polish
            polish_rows = gap_exchange_search(polish_rows, rng_polish, max_iter=1000)
            if is_valid(polish_rows):
                polish_max = scope_of(polish_rows)
            # Two-element swap in polish
            polish_rows = two_elem_swap(polish_rows, rng_polish, max_iter=1000)
            if is_valid(polish_rows):
                polish_max = scope_of(polish_rows)
            polish_rows = tighten(polish_rows, max_iter=100)
            if is_valid(polish_rows):
                polish_max = scope_of(polish_rows)
            if polish_max < best_scope:
                best_rows = polish_rows
                best_scope = polish_max

    # Final tightening
    best_rows = tighten(best_rows, max_iter=300)
    if not is_valid(best_rows):
        # revert to last known valid
        best_rows = greedy_initial(random.Random(7))
        best_rows = tighten(best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
