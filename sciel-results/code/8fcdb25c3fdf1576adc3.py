"""Difference triangle set solver: ILS with targeted scope reduction and optimized conflict tracking."""
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Optimized Conflict Tracker using bytearray ----------
    class ConflictTracker:
        def __init__(self, rows):
            max_d = max(max(r) for r in rows) if rows else 0
            # Use bytearray for memory efficiency and speed
            # We only care if count is 0, 1, or >1 (capped at 2)
            self.conf = bytearray(max_d + 2)
            self.n_conf = 0
            for row in rows:
                for d in row_diffs_list(row):
                    self._add(d, 1)

        def _ensure(self, d):
            if d >= len(self.conf):
                # Extend by 16 bytes at a time to avoid frequent resizing
                self.conf.extend(b'\x00' * (d + 16 - len(self.conf)))

        def _add(self, d, delta):
            self._ensure(d)
            old = self.conf[d]
            new = old + delta
            if new < 0:
                new = 0 # Should not happen if balanced, but safe
            elif new > 2:
                new = 2 # Cap at 2 to fit in byte and simplify logic
            
            # Update n_conf: count of d where conf[d] >= 2
            if old < 2 and new >= 2:
                self.n_conf += 1
            elif old >= 2 and new < 2:
                self.n_conf -= 1
            
            self.conf[d] = new

        def remove_row(self, row):
            for d in row_diffs_list(row):
                self._add(d, -1)

        def add_row(self, row):
            for d in row_diffs_list(row):
                self._add(d, 1)

        def valid(self):
            return self.n_conf == 0

        def get_conflict_count(self):
            return self.n_conf

    # ---------- Row Builders ----------
    def greedy_row(used_set, jitter=0.0, cap=10**6, max_val=None):
        """
        Builds a row greedily.
        If max_val is provided, ensures all elements <= max_val.
        """
        row, x = [0], 0
        row_diffs = set()
        limit = cap if max_val is None else max_val
        while len(row) < k + 1:
            found = False
            # Look ahead a bit to handle jitter/skips
            for trial_x in range(x + 1, limit + 1):
                if trial_x > limit:
                    break
                new = {trial_x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter > 0 and random.random() < jitter and trial_x < 400:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    x = trial_x
                    found = True
                    break
            if not found:
                return None
        return row

    def used_set_except(rows, idx):
        used = set()
        for j in range(n_rows):
            if j != idx:
                used |= row_diffs_set(rows[j])
        return used

    def greedy_full(jitter=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, jitter=jitter)
            if row is None:
                return None
            rows.append(row)
            used |= row_diffs_set(row)
        return rows

    def interleaved_greedy(jitter=0.0):
        """Build rows in interleaved fashion: add one element to each row per round."""
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(k):
            # Order: prefer rows with smaller last element, with jitter
            order = sorted(range(n_rows), key=lambda i: rows[i][-1] + (random.random() * 0.5 if jitter > 0 else 0))
            for i in order:
                row = rows[i]
                x = row[-1] + 1
                found = False
                while x < 10**6:
                    new = {x - a for a in row}
                    # Check internal row diffs efficiently
                    internal_conflict = False
                    for d in new:
                        if d in used:
                            internal_conflict = True
                            break
                    if not internal_conflict:
                        # Check against existing row diffs
                        # We can maintain row_diffs set locally for speed
                        pass 
                    # For simplicity in interleaved, we recompute or use a local set
                    # Let's use a local set for row diffs to be safe and fast
                    row_diffs_local = set()
                    for i_r in range(len(row)):
                        for j_r in range(i_r + 1, len(row)):
                            row_diffs_local.add(row[j_r] - row[i_r])
                            
                    if not (new & used) and not (new & row_diffs_local):
                        if jitter > 0 and random.random() < jitter and x < 400:
                            x += 1
                            continue
                        row.append(x)
                        for d in new:
                            used.add(d)
                        found = True
                        break
                    x += 1
                if not found:
                    return None
        return rows

    # ---------- Targeted Polish: Try to reduce the scope of the bottleneck row ----------
    def targeted_polish(rows, max_passes=10):
        """
        Aggressively tries to reduce the global scope by targeting the row(s)
        with the maximum value.
        """
        rows = [list(r) for r in rows]
        improved = True
        passes = 0
        while improved and time_left() > 10 and passes < max_passes:
            improved = False
            passes += 1
            
            current_scope = scope(rows)
            
            # Find rows that define the current scope
            bottleneck_indices = [i for i in range(n_rows) if max(rows[i]) == current_scope]
            
            # Try to reduce each bottleneck row
            for idx in bottleneck_indices:
                if time_left() < 10:
                    break
                used_other = used_set_except(rows, idx)
                current_row_max = max(rows[idx])
                
                # Try to build a new row with max value < current_row_max
                # Start with current_row_max - 1 and go down if possible, or just try with jitter
                best_row = None
                best_max = current_row_max
                
                # Strategy 1: Try strict limit current_scope - 1
                # Strategy 2: Try with jitter to find a compact row
                for attempt in range(20):
                    if time_left() < 10:
                        break
                    # Decrease the limit slightly as we go, or use jitter
                    limit = current_scope - 1 - (attempt // 5)
                    if limit < current_row_max - 5: # Don't go too far down immediately
                        limit = current_scope - 1
                    
                    jitter = 0.0 if attempt < 5 else 0.1 + 0.05 * (attempt - 5)
                    
                    new_row = greedy_row(used_other, jitter=jitter, max_val=limit)
                    if new_row is not None:
                        new_max = max(new_row)
                        if new_max < best_max:
                            best_row = new_row
                            best_max = new_max
                
                if best_row is not None and best_max < current_row_max:
                    # Validate the whole set to be sure (should be valid by construction)
                    test_rows = [list(r) for r in rows]
                    test_rows[idx] = best_row
                    if is_valid(test_rows):
                        rows[idx] = best_row
                        improved = True
                        # Update current_scope for subsequent iterations in this pass?
                        # No, we break out to re-evaluate all bottlenecks
                        break 
                        
        return rows

    # ---------- General Local Polish (for non-bottleneck rows or general improvement) ----------
    def local_polish(rows, max_passes=5):
        rows = [list(r) for r in rows]
        improved = True
        passes = 0
        while improved and time_left() > 15 and passes < max_passes:
            improved = False
            passes += 1
            # Sort by max value descending to prioritize large rows
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 15:
                    break
                used_other = used_set_except(rows, i)
                current_max = max(rows[i])
                best_row = None
                best_max = current_max
                
                # Try to find a smaller row
                for attempt in range(15):
                    if time_left() < 15:
                        break
                    limit = current_max - 1
                    jitter = 0.1 * (attempt / 15.0)
                    new_row = greedy_row(used_other, jitter=jitter, max_val=limit)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                
                if best_row is not None:
                    rows[i] = best_row
                    improved = True
        return rows

    # ---------- Kick ----------
    def kick(rows):
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        for idx in indices:
            used_other = used_set_except(rows, idx)
            done = False
            # Try to rebuild with high jitter to escape local optima
            for attempt in range(20):
                jitter = 0.3 + 0.1 * attempt
                row = greedy_row(used_other, jitter=jitter, cap=600)
                if row is not None:
                    rows[idx] = row
                    done = True
                    break
            if not done:
                # Fallback: no jitter
                row = greedy_row(used_other, jitter=0.0, cap=10**6)
                if row is not None:
                    rows[idx] = row
        return rows

    # ---------- Initialization ----------
    best_rows = greedy_full(jitter=0.0)
    if best_rows is None:
        best_rows = interleaved_greedy(jitter=0.0)
    best_scope = scope(best_rows)

    # Try a few different initializations
    if time_left() > 30:
        for r in range(5):
            if time_left() < 30:
                break
            cand = interleaved_greedy(jitter=random.uniform(0.05, 0.2))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s

    # Initial targeted polish
    if time_left() > 20:
        polished = targeted_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # ---------- Main ILS Loop ----------
    cycles = 0
    while time_left() > 25:
        cycles += 1
        
        # 1. Targeted Polish (Priority)
        polished = targeted_polish(best_rows, max_passes=8)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        if time_left() < 20:
            break

        # 2. General Polish
        if time_left() > 30:
            polished2 = local_polish(best_rows, max_passes=3)
            if scope(polished2) < best_scope:
                best_rows = polished2
                best_scope = scope(polished2)

        # 3. Kick
        if time_left() < 15:
            break
        kicked = kick(best_rows)
        
        # 4. Polish the kicked solution
        if time_left() < 10:
            break
        polished3 = targeted_polish(kicked, max_passes=5)
        if scope(polished3) < best_scope:
            best_rows = polished3
            best_scope = scope(polished3)
        
        # Periodic restart
        if cycles % 10 == 0 and time_left() > 40:
            cand = interleaved_greedy(jitter=random.uniform(0.1, 0.3))
            if cand is not None:
                s = scope(cand)
                if s < best_scope:
                    best_rows = cand
                    best_scope = s
                    polished4 = targeted_polish(best_rows, max_passes=5)
                    if scope(polished4) < best_scope:
                        best_rows = polished4
                        best_scope = scope(polished4)

    # Final Polish
    if time_left() > 10:
        polished = targeted_polish(best_rows, max_passes=15)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final Validity Check
    if not is_valid(best_rows):
        # Fallback to a known valid greedy solution
        best_rows = greedy_full(jitter=0.0)

    return best_rows
