import time
import random
import math
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # counting lower bound

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

    def row_diffs(row):
        """Return set of all positive differences in a row."""
        s = set()
        for i in range(L):
            for j in range(i + 1, L):
                s.add(row[j] - row[i])
        return s

    def can_extend(row, new_val, used_set):
        """Check if new_val can be appended to row without conflicts."""
        for a in row:
            d = new_val - a
            if d in used_set:
                return False
        return True

    def greedy_build(seed, max_scope=None):
        """Build a complete valid assignment via randomized greedy."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                # Adaptive search window
                window = 25 + int(0.2 * len(used)) + rng.randint(0, 20)
                found = False
                start = last + 1
                end = min(start + window, (max_scope or lb + 400) + 1)
                candidates = list(range(start, end))
                rng.shuffle(candidates)
                for x in candidates:
                    if can_extend(row, x, used):
                        used.update([x - a for a in row])
                        row.append(x)
                        found = True
                        break
                if not found:
                    # Expand search
                    for x in range(start, (max_scope or lb + 600) + 1):
                        if can_extend(row, x, used):
                            used.update([x - a for a in row])
                            row.append(x)
                            found = True
                            break
                    if not found:
                        return None
            rows.append(row)
        return rows

    def backtrack_row(used_set, max_val, rng, time_limit):
        """
        Backtracking with MRV to build a single row within max_val.
        Returns row list or None.
        """
        row = [0]
        
        def backtrack(pos):
            if time.time() > time_limit:
                return False
            if pos == L:
                return True
            
            # MRV: try values that have fewest conflicts first
            last = row[-1]
            candidates = list(range(last + 1, max_val + 1))
            
            # Score candidates by number of new differences that are already used
            scored = []
            for x in candidates:
                conflicts = 0
                for a in row:
                    if (x - a) in used_set:
                        conflicts += 1
                scored.append((conflicts, x))
            
            scored.sort()
            
            for _, x in scored:
                new_diffs = [x - a for a in row]
                if any(d in used_set for d in new_diffs):
                    continue
                # Add diffs
                for d in new_diffs:
                    used_set.add(d)
                row.append(x)
                if backtrack(pos + 1):
                    return True
                # Undo
                row.pop()
                for d in new_diffs:
                    used_set.discard(d)
            return False
        
        if backtrack(1):
            return row
        return None

    def rebuild_row_backtrack(rows, r_idx, used_set, max_scope, rng, time_limit):
        """Rebuild row r_idx using backtracking within max_scope."""
        old_row = list(rows[r_idx])
        
        # Remove old diffs
        for d in row_diffs(old_row):
            used_set.discard(d)
        
        new_row = backtrack_row(used_set, max_scope, rng, time_limit)
        if new_row is None:
            # Restore old row
            for d in row_diffs(old_row):
                used_set.add(d)
            rows[r_idx] = old_row
            return False
        
        rows[r_idx] = new_row
        return True

    def can_place(row, pos, new_val, used_set):
        """Check if new_val can replace row[pos] without conflicts."""
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        # Compute old diffs involving pos
        old_diffs = set()
        for i in range(L):
            if i < pos:
                old_diffs.add(row[pos] - row[i])
            elif i > pos:
                old_diffs.add(row[i] - row[pos])
        other = used_set - old_diffs
        # Check new diffs
        for i in range(L):
            if i < pos:
                d = new_val - row[i]
                if d in other:
                    return False
            elif i > pos:
                d = row[i] - new_val
                if d in other:
                    return False
        return True

    def apply_set(row, pos, new_val, used_set):
        """Replace row[pos] with new_val, updating used_set."""
        old_val = row[pos]
        # Remove old diffs
        for i in range(L):
            if i < pos:
                used_set.discard(old_val - row[i])
            elif i > pos:
                used_set.discard(row[i] - old_val)
        # Add new diffs
        for i in range(L):
            if i < pos:
                used_set.add(new_val - row[i])
            elif i > pos:
                used_set.add(row[i] - new_val)
        row[pos] = new_val

    def simulated_annealing(rows_init, time_limit):
        """SA to minimize scope."""
        rows = [list(r) for r in rows_init]
        used = build_used(rows)
        current_scope = get_scope(rows)
        best_local = current_scope
        best_rows_local = [list(r) for r in rows]
        
        T = 50.0
        T_min = 0.1
        cooling = 0.997
        rng = random.Random(int(time.time() * 1000))
        
        while time.time() < time_limit and T > T_min:
            # With 15% probability, rebuild a random row using backtracking
            if rng.random() < 0.15:
                r_idx = rng.randint(0, n_rows - 1)
                allowed_scope = best_local + 3
                if rebuild_row_backtrack(rows, r_idx, used, allowed_scope, rng, time_limit):
                    new_scope = get_scope(rows)
                    delta = new_scope - current_scope
                    if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                        current_scope = new_scope
                        if current_scope < best_local:
                            best_local = current_scope
                            best_rows_local = [list(r) for r in rows]
                    else:
                        # Undo: rebuild old row
                        old_row = rows[r_idx]
                        for d in row_diffs(old_row):
                            used.discard(d)
                        # We can't easily restore, so just accept and continue
                        # Actually, let's just accept to keep it simple
                        current_scope = new_scope
                        if current_scope < best_local:
                            best_local = current_scope
                            best_rows_local = [list(r) for r in rows]
            else:
                # Single element move
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, L - 1)
                row = rows[r_idx]
                old_val = row[pos]
                
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < L - 1 else old_val + 30
                
                if lower >= upper:
                    T *= cooling
                    continue
                
                if pos == L - 1 and rng.random() < 0.6:
                    new_val = rng.randint(lower, min(old_val, upper))
                else:
                    new_val = rng.randint(lower, max(lower, upper))
                
                if new_val == old_val:
                    continue
                
                if not can_place(row, pos, new_val, used):
                    continue
                
                apply_set(row, pos, new_val, used)
                new_scope = get_scope(rows)
                
                delta = new_scope - current_scope
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                    current_scope = new_scope
                    if current_scope < best_local:
                        best_local = current_scope
                        best_rows_local = [list(r) for r in rows]
                else:
                    apply_set(row, pos, old_val, used)
            
            T *= cooling
        
        return best_rows_local, best_local

    def hill_climb_compact(rows, time_limit):
        """Aggressive local compaction."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = [(r, p) for r in range(n_rows) for p in range(L - 1, 0, -1)]
            random.shuffle(order)
            for r_idx, pos in order:
                if time.time() >= time_limit:
                    break
                row = work[r_idx]
                old_val = row[pos]
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < L - 1 else old_val
                if lower >= old_val:
                    continue
                for new_val in range(lower, min(old_val, upper + 1)):
                    if can_place(row, pos, new_val, used_set):
                        apply_set(row, pos, new_val, used_set)
                        improved = True
                        break
        return work

    def polish(rows, time_limit):
        """High-precision polish: iteratively try to reduce scope by 1."""
        work = [list(r) for r in rows]
        used_set = build_used(work)
        current_scope = get_scope(work)
        
        while time.time() < time_limit:
            # Try to reduce scope by 1
            target = current_scope - 1
            if target < lb:
                break
            
            improved = False
            # Try rebuilding each row with tighter bound
            for r_idx in range(n_rows):
                if time.time() > time_limit:
                    break
                if work[r_idx][-1] <= target:
                    continue
                # Try to rebuild this row within target
                old_row = list(work[r_idx])
                for d in row_diffs(old_row):
                    used_set.discard(d)
                
                new_row = backtrack_row(used_set, target, random.Random(r_idx * 1000 + int(time.time())), time_limit)
                if new_row is not None:
                    work[r_idx] = new_row
                    improved = True
                else:
                    for d in row_diffs(old_row):
                        used_set.add(d)
                    work[r_idx] = old_row
            
            if improved:
                current_scope = get_scope(work)
            else:
                break
        
        # Final compaction
        work = hill_climb_compact(work, time_limit)
        return work

    # Phase 1: Multiple greedy constructions with varied max_scope
    num_greedy = 3000
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        # Vary the max_scope to explore different regions
        ms = lb + 20 + (i % 50)
        rows = greedy_build(i * 131 + 17, ms)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lb + 3:
                    break

    if best_rows is None:
        # Fallback: try harder greedy
        for i in range(1000):
            if time.time() > deadline - 5:
                break
            rows = greedy_build(90000 + i * 73, lb + 100)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]

    if best_rows is None:
        return None

    # Phase 2: Simulated annealing
    if time.time() < deadline - 8:
        sa_rows, sa_scope = simulated_annealing(best_rows, deadline - 8)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Phase 3: Hill climb compaction
    if time.time() < deadline - 5:
        compacted = hill_climb_compact(best_rows, deadline - 5)
        c_scope = get_scope(compacted)
        if c_scope < best_scope:
            best_scope = c_scope
            best_rows = compacted

    # Phase 4: Polish pass
    if time.time() < deadline - 3:
        polished = polish(best_rows, deadline - 3)
        p_scope = get_scope(polished)
        if p_scope < best_scope:
            best_scope = p_scope
            best_rows = polished

    # Phase 5: Final SA round
    if time.time() < deadline - 2:
        sa2_rows, sa2_scope = simulated_annealing(best_rows, deadline - 2)
        if sa2_scope < best_scope:
            best_scope = sa2_scope
            best_rows = sa2_rows

    # Phase 6: Final compaction
    if time.time() < deadline - 1:
        final_compact = hill_climb_compact(best_rows, deadline - 1)
        f_scope = get_scope(final_compact)
        if f_scope < best_scope:
            best_scope = f_scope
            best_rows = final_compact

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
