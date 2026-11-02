import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    used.add(row[j] - ri)
        return used

    def diffs_for_pos(row, pos):
        """Differences involving element at pos (excluding its own pair with itself)."""
        v = row[pos]
        s = set()
        for i in range(L):
            if i < pos:
                s.add(v - row[i])
            elif i > pos:
                s.add(row[i] - v)
        return s

    def can_place(row, pos, new_val, used_set):
        """Check if we can set row[pos]=new_val, given used_set contains all current diffs."""
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        old = diffs_for_pos(row, pos)
        other = used_set - old
        v = new_val
        for i in range(L):
            if i < pos:
                d = v - row[i]
                if d in other:
                    return False
            elif i > pos:
                d = row[i] - v
                if d in other:
                    return False
        return True

    def apply_set(row, pos, new_val, used_set):
        old = diffs_for_pos(row, pos)
        for d in old:
            used_set.discard(d)
        v = new_val
        for i in range(L):
            if i < pos:
                used_set.add(v - row[i])
            elif i > pos:
                used_set.add(row[i] - v)
        row[pos] = new_val

    def local_compact(rows, time_limit):
        """Iteratively compact each element to its smallest feasible value."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = work[r_idx]
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old_val:
                        continue
                    upper = row[pos + 1] if pos < L - 1 else old_val
                    # try from lower up to old_val-1
                    for new_val in range(lower, old_val):
                        if new_val >= upper:
                            break
                        if can_place(row, pos, new_val, used_set):
                            apply_set(row, pos, new_val, used_set)
                            improved = True
                            break
        return work

    def greedy_construct(seed, strategy='basic'):
        rng = random.Random(seed)
        used = set()
        rows = []

        if strategy == 'random_order':
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows
            for row_idx in row_order:
                row = [0]
                while len(row) < L:
                    last = row[-1]
                    found = False
                    search_end = last + 35 + rng.randint(0, 25)
                    candidates = list(range(last + 1, min(search_end, lb + 250)))
                    if rng.random() < 0.5:
                        rng.shuffle(candidates)
                    for x in candidates:
                        nd = [x - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(x)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        for x in range(last + 1, lb + 400):
                            nd = [x - a for a in row]
                            if any(d in used for d in nd):
                                continue
                            row.append(x)
                            used.update(nd)
                            found = True
                            break
                        if not found:
                            return None
                row_slots[row_idx] = row
            if any(r is None for r in row_slots):
                return None
            return row_slots
        else:
            for _ in range(n_rows):
                row = [0]
                while len(row) < L:
                    last = row[-1]
                    found = False
                    if strategy == 'compact':
                        search_end = last + 22 + rng.randint(0, 12)
                    elif strategy == 'spread':
                        search_end = last + 45 + rng.randint(0, 35)
                    else:
                        search_end = last + 32 + rng.randint(0, 22)
                    candidates = list(range(last + 1, min(search_end, lb + 250)))
                    if rng.random() < 0.5:
                        rng.shuffle(candidates)
                    for x in candidates:
                        nd = [x - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(x)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        for x in range(last + 1, lb + 400):
                            nd = [x - a for a in row]
                            if any(d in used for d in nd):
                                continue
                            row.append(x)
                            used.update(nd)
                            found = True
                            break
                        if not found:
                            return None
                rows.append(row)
            return rows

    def random_restart(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                found = False
                jump = rng.randint(1, 55)
                x = last + jump
                for candidate in range(max(last + 1, x - 10), x + 12):
                    nd = [candidate - a for a in row]
                    if any(d in used for d in nd):
                        continue
                    row.append(candidate)
                    used.update(nd)
                    found = True
                    break
                if not found:
                    for candidate in range(last + 1, last + 250):
                        nd = [candidate - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(candidate)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    def backtrack_search(rows_init, max_scope, time_limit):
        """MRV backtracking to try to find a solution with scope <= max_scope."""
        rows = [list(r) for r in rows_init]
        used = build_used(rows)
        L_local = L
        found = [False]
        best = [None]

        def backtrack():
            if time.time() > time_limit:
                return False
            # MRV: pick unfilled position with fewest candidates
            # We treat "unfilled" as any position we still want to try to improve.
            # Simpler: try to reduce the max element via local moves.
            # Pick row with largest last element.
            target = max(range(n_rows), key=lambda i: rows[i][-1])
            row = rows[target]
            # Try to reduce the last element
            pos = L_local - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1
            for new_val in range(lower, old_val):
                if can_place(row, pos, new_val, used):
                    apply_set(row, pos, new_val, used)
                    # recurse to further improve
                    if backtrack():
                        return True
                    # undo
                    apply_set(row, pos, old_val, used)
                    # also try intermediate values? we already tried in order
                    break
            return False

        if backtrack():
            return [list(r) for r in rows]
        return None

    # Strategy 1: Many greedy attempts with different strategies
    strategies = ['basic', 'compact', 'spread', 'random_order']
    for i in range(3000):
        if time.time() > deadline - 8:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Strategy 2: Local compaction on best found
    if best_rows is not None:
        improved = local_compact(best_rows, deadline - 6)
        imp_scope = get_scope(improved)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Strategy 3: Random restarts + compact
    for i in range(800):
        if time.time() > deadline - 6:
            break
        rows = random_restart(100000 + i * 97)
        if rows is not None:
            rows = local_compact(rows, deadline - 5)
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Strategy 4: Final backtracking polish with tighter bound
    if best_rows is not None and time.time() < deadline - 3:
        # Try to find a solution with scope <= best_scope - 1
        polished = backtrack_search(best_rows, best_scope - 1, deadline - 2)
        if polished is not None:
            ps = get_scope(polished)
            if ps < best_scope:
                best_scope = ps
                best_rows = polished

    # Strategy 5: One more compaction pass
    if best_rows is not None and time.time() < deadline - 1:
        improved = local_compact(best_rows, deadline - 0.5)
        imp_scope = get_scope(improved)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
