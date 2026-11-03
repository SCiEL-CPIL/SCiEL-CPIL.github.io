import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    L = k + 1  # length of each row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != L:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # Greedy construction
    def greedy_construct(n_rows, k, rng=None):
        if rng is None:
            rng = random.Random(0)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                if rng.random() < 0.01:
                    x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Fast local compaction: pull each element to smallest feasible position
    def _get_diffs_for_pos(row, pos, L):
        diffs = set()
        for i in range(L):
            if i < pos:
                diffs.add(row[pos] - row[i])
            elif i > pos:
                diffs.add(row[i] - row[pos])
        return diffs

    def fast_compact_row(row, used_set, L):
        changed = False
        for pos in range(L - 1, 0, -1):
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            old_diffs = _get_diffs_for_pos(row, pos, L)
            for new_val in range(lower, old_val):
                if pos < L - 1 and new_val >= row[pos + 1]:
                    break
                new_diffs = set()
                for i in range(L):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                other = used_set - old_diffs
                if new_diffs & other:
                    continue
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                changed = True
                break
        return changed

    def local_compact(rows, n_rows, k, time_limit):
        work = [list(r) for r in rows]
        used_set = get_used_diffs(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                if time.time() > time_limit:
                    break
                if fast_compact_row(work[r_idx], used_set, L):
                    improved = True
        return work

    # Bitset-accelerated backtracking for scope reduction
    def backtracking_scope_search(n_rows, k, target_scope, deadline, seed_offset=0):
        rng = random.Random(seed_offset * 7919 + 31)
        L = k + 1
        rows = []
        
        used_diffs = [False] * (target_scope + 1)
        used_count = 0

        def build_row(pos, current_mark, row_so_far):
            nonlocal used_count
            if time.time() > deadline:
                return None

            if pos == L:
                return list(row_so_far)

            # Prune: need at least (L - pos - 1) more values, each at least 1 apart
            max_x = target_scope - (L - pos - 1)
            if max_x <= current_mark:
                return None

            # Find valid candidates
            valid_candidates = []
            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d <= target_scope and used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                if len(valid_candidates) >= 40:
                    break

            if not valid_candidates:
                return None

            # Try candidates, biased toward smaller
            n_to_try = min(15, len(valid_candidates))
            for ci in range(n_to_try):
                x = valid_candidates[ci]
                # Record new diffs
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    if d <= target_scope and used_diffs[d]:
                        break
                    new_diffs.append(d)
                else:
                    # All good
                    for d in new_diffs:
                        used_diffs[d] = True
                        used_count += 1
                    result = build_row(pos + 1, x, row_so_far + [x])
                    for d in new_diffs:
                        used_diffs[d] = False
                        used_count -= 1
                    if result is not None:
                        return result
            return None

        # Build rows one by one
        def build_all_rows(row_idx, current_rows):
            if time.time() > deadline:
                return None
            if row_idx == n_rows:
                return current_rows[:]
            
            row = build_row(1, 0, [0])
            if row is None:
                return None
            
            current_rows.append(row)
            # Mark diffs for this row
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= target_scope:
                        used_diffs[d] = True
                        used_count += 1
            
            result = build_all_rows(row_idx + 1, current_rows)
            
            # Unmark
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= target_scope:
                        used_diffs[d] = False
                        used_count -= 1
            current_rows.pop()
            
            return result

        result = build_all_rows(0, [])
        return result

    # Simulated Annealing with row regeneration
    def generate_row_bitset(used_diffs, k_val, max_scope, randomize=False, p_smallest=0.7, window=5):
        max_d = max_scope + 1
        forbidden_bits = 0
        for d in used_diffs:
            if d < max_d:
                forbidden_bits |= (1 << d)
        
        row = [0]
        row_diffs_bits = 0
        
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            
            while x < max_scope:
                conflict = False
                for a in row:
                    d = x - a
                    if d < max_d and (forbidden_bits >> d) & 1:
                        conflict = True
                        break
                    if d < max_d and (row_diffs_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
            
            for a in row:
                d = next_x - a
                if d < max_d:
                    row_diffs_bits |= (1 << d)
            
            row.append(next_x)
        
        return row

    # 1. Initial Solution: Deterministic Greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # 2. Aggressive local compaction first
    rows = local_compact(rows, n_rows, k, deadline - 100)
    new_max = max(max(r) for r in rows)
    if new_max < best_max:
        best_max = new_max
        best_rows = [r[:] for r in rows]
        current_max = best_max

    # 3. Backtracking scope reduction
    for reduction in range(1, 20):
        if time.time() > deadline - 40:
            break
        target = best_max - reduction
        if target < k * (k + 1) // 2:
            break
        result = backtracking_scope_search(n_rows, k, target, deadline - 35, seed_offset=reduction)
        if result is not None and is_valid(result):
            new_max = max(max(r) for r in result)
            if new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in result]
                rows = [r[:] for r in result]
                current_max = best_max
                # Compact further
                rows = local_compact(rows, n_rows, k, deadline - 35)
                new_max2 = max(max(r) for r in rows)
                if new_max2 < best_max:
                    best_max = new_max2
                    best_rows = [r[:] for r in rows]
                    current_max = best_max
                continue
        break

    # 4. Simulated Annealing with row regeneration
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    used = get_used_diffs(rows)
    
    sa_deadline = deadline - 20
    while time.time() < sa_deadline:
        if T < T_min:
            T = 5.0
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        new_row = generate_row_bitset(used, k, best_max + 20, randomize=True, p_smallest=0.8, window=3)
        
        if new_row is None:
            used |= old_diffs
            continue
        
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i + 1, len(new_row)))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            used |= old_diffs
            
        T *= cooling

    # 5. Final aggressive compaction
    final_rows = [list(r) for r in best_rows]
    final_rows = local_compact(final_rows, n_rows, k, deadline - 5)
    final_scope = max(max(r) for r in final_rows)
    if is_valid(final_rows) and final_scope < best_max:
        best_max = final_scope
        best_rows = [r[:] for r in final_rows]

    # 6. One more backtracking attempt with the improved baseline
    if time.time() < deadline - 8:
        for reduction in range(1, 8):
            if time.time() > deadline - 3:
                break
            target = best_max - reduction
            if target < k * (k + 1) // 2:
                break
            result = backtracking_scope_search(n_rows, k, target, deadline - 2, seed_offset=reduction + 100)
            if result is not None and is_valid(result):
                new_max = max(max(r) for r in result)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in result]
                    # Quick compact
                    best_rows = local_compact(best_rows, n_rows, k, deadline - 1)
                    new_max2 = max(max(r) for r in best_rows)
                    if new_max2 < best_max:
                        best_max = new_max2
                    break

    # Final validity check
    if not is_valid(best_rows):
        # Fallback: regenerate
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)

    return best_rows
