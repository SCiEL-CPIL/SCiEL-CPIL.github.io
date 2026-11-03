import time
import random
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 147.0
    k1 = k + 1
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    if k1 <= 0:
        return []

    best = greedy_construct(n_rows, k)
    if not best:
        return []
    best_scope = max(max(r) for r in best)

    if best_scope <= lower_bound:
        return best

    best = _improve_scope(best, n_rows, k, deadline)
    best_scope = max(max(r) for r in best)
    if best_scope <= lower_bound:
        return best

    # Iterative deepening: try to find a valid solution with a smaller max entry.
    target = best_scope - 1
    while time.time() < deadline - 3.0:
        if target < lower_bound:
            break
        found = _backtracking_search(n_rows, k, target, deadline)
        if found:
            found_scope = max(max(r) for r in found)
            if found_scope < best_scope:
                best = found
                best_scope = found_scope
                best = _improve_scope(best, n_rows, k, deadline)
                best_scope = max(max(r) for r in best)
                if best_scope <= lower_bound:
                    return best
                target = best_scope - 1
        else:
            target -= 1
            if target < lower_bound:
                break

    # Final compaction
    best = _improve_scope(best, n_rows, k, deadline)
    return best


def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _backtracking_search(n_rows, k, max_scope, deadline):
    """
    Time-sliced backtracking with MRV + forward checking.
    Repeatedly restarts (different random orderings) within a wall-clock budget,
    returning the first complete valid solution found.
    """
    k1 = k + 1
    max_diff = max_scope

    # Precompute: for each candidate x, the bitmask of diffs {x - a : a in [0..k1-1]}
    # We need diffs with the row prefix, but for speed we precompute a full mask per x
    # and AND with a prefix mask.
    prefix_masks = [0] * (k1 + 1)
    for pos in range(1, k1 + 1):
        m = 0
        for i in range(pos):
            m |= (1 << (0 - i))  # placeholder, will be recomputed
        prefix_masks[pos] = m
    # Actually, diffs depend on the actual values, not just indices. So we compute on the fly.
    # For speed, we precompute nothing; instead we use a helper.

    def mask_for(x, row):
        m = 0
        for a in row:
            d = x - a
            m |= (1 << d)
        return m

    def valid_mask(x, row, used_bits):
        m = mask_for(x, row)
        return (m & used_bits) == 0

    # Forward-checking: after placing x at position pos, check that each remaining
    # position pos' in [pos+1, k1-1] has at least one valid candidate in
    # [x+1, max_scope - (k1-1-pos')].
    def forward_check(row, pos, x, used_bits):
        # used_bits already includes x's diffs
        for pos2 in range(pos + 1, k1):
            min_val = x + 1
            max_val = max_scope - (k1 - 1 - pos2)
            if max_val < min_val:
                return False
            found = False
            for y in range(min_val, max_val + 1):
                if valid_mask(y, row, used_bits):
                    found = True
                    break
            if not found:
                return False
        return True

    def build_row(pos, current_mark, row, used_bits, rng, deadline):
        if pos == k1:
            return list(row), used_bits

        remaining = k1 - pos
        max_x = max_scope - (remaining - 1)
        if max_x <= current_mark:
            return None, None

        # Collect valid candidates
        valid = []
        for x in range(current_mark + 1, max_x + 1):
            if valid_mask(x, row, used_bits):
                valid.append(x)

        if not valid:
            return None, None

        # MRV: score each candidate by the number of valid candidates it leaves
        # for the next position (fewest = most constrained = try first).
        # For speed, cap the number of candidates we score.
        n_score = min(len(valid), 60)
        scored = []
        for x in valid[:n_score]:
            # Count valid candidates for position pos+1 after placing x
            new_used = used_bits | mask_for(x, row)
            next_min = x + 1
            next_max = max_scope - (k1 - 1 - (pos + 1))
            if next_max < next_min:
                scored.append((0, x))  # infeasible
                continue
            cnt = 0
            for y in range(next_min, min(next_min + 200, next_max + 1)):
                if valid_mask(y, row + [x], new_used):
                    cnt += 1
                    if cnt > 50:
                        break
            scored.append((cnt, x))

        scored.sort(key=lambda t: (t[0], t[1]))
        top = [x for _, x in scored[:10]]

        # Mix in some small and random candidates for diversity
        small = [x for x in valid[:6] if x not in top]
        top.extend(small[:3])
        if len(valid) > 15 and rng.random() < 0.3:
            larger = valid[len(valid) // 2:]
            top.extend(rng.sample(larger, min(3, len(larger))))

        rng.shuffle(top)

        for x in top:
            new_used = used_bits | mask_for(x, row)
            new_row = row + [x]
            # Forward check
            if not forward_check(new_row, pos, x, new_used):
                continue
            res, res_used = build_row(pos + 1, x, new_row, new_used, rng, deadline)
            if res is not None:
                return res, res_used
        return None, None

    # Time-sliced restarts
    deadline_local = deadline - 2.0
    while time.time() < deadline_local:
        rng = random.Random(int(time.time() * 1000000))
        rows = []
        used_bits = 0
        ok = True
        for i in range(n_rows):
            row, used_bits = build_row(1, 0, [0], used_bits, rng, deadline_local)
            if row is None:
                ok = False
                break
            rows.append(row)
        if ok:
            return rows
        # If we're running out of time, stop
        if time.time() > deadline_local:
            break

    return None


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: repeatedly pull each element to its smallest
    feasible value, then try 2-element swaps to escape local minima.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k1):
            for j in range(i + 1, k1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k1):
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    for j in range(pos + 1, k1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                if ok:
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])
                    for j in range(pos + 1, k1):
                        old_diffs.add(row[j] - row[pos])
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_changed = True
                    break
        return any_changed

    # Phase 1: Repeatedly pull all rows to smallest feasible
    for _ in range(30):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Try 2-element swaps
    for _ in range(15):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos1 in range(1, k1):
                for pos2 in range(pos1 + 1, k1):
                    v1, v2 = row[pos1], row[pos2]
                    if v1 >= v2:
                        continue
                    if pos1 > 0 and v2 <= row[pos1 - 1]:
                        continue
                    if pos2 < k1 - 1 and v1 >= row[pos2 + 1]:
                        continue
                    new_diffs = set()
                    ok = True
                    for i in range(pos1):
                        d = v2 - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        for i in range(pos1 + 1, pos2):
                            d = v2 - row[i]
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        for i in range(pos2 + 1, k1):
                            d = row[i] - v2
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        for i in range(pos1 + 1, pos2):
                            d = v1 - row[i]
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        for i in range(pos2 + 1, k1):
                            d = row[i] - v1
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        old_diffs = set()
                        for i in range(pos1):
                            old_diffs.add(v1 - row[i])
                        for i in range(pos1 + 1, pos2):
                            old_diffs.add(v1 - row[i])
                        for i in range(pos2 + 1, k1):
                            old_diffs.add(row[i] - v1)
                        for i in range(pos1):
                            old_diffs.add(v2 - row[i])
                        for i in range(pos1 + 1, pos2):
                            old_diffs.add(v2 - row[i])
                        for i in range(pos2 + 1, k1):
                            old_diffs.add(row[i] - v2)
                        old_diffs.add(v2 - v1)
                        if not (new_diffs & (used_diffs - old_diffs)):
                            used_diffs.difference_update(old_diffs)
                            used_diffs.update(new_diffs)
                            row[pos1], row[pos2] = v2, v1
                            any_changed = True
                            break
                if any_changed:
                    break
            if any_changed:
                break
        if not any_changed:
            break

    # Phase 3: Final pull pass
    for _ in range(10):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
