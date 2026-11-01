"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.

Combined approach:
1. Incremental difference-count tracking for fast conflict evaluation (inspiration 3).
2. Greedy initialization with multiple orderings (current program).
3. Backtracking with binary search on scope (inspiration 1).
4. High-frequency local search using incremental conflict counts (inspiration 3).
5. Simulated annealing with incremental tracking (current + inspiration 3).
6. Iterated local search with kicks (current program).
"""
import time
import random

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _diffs_of_row(row):
    """Return a set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _validate(rows):
    """Check all differences are distinct across all rows. Return True if valid."""
    seen = set()
    for row in rows:
        d = _diffs_of_row(row)
        if len(d) != len(row) * (len(row) - 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


def _greedy_build(n_rows, k, rng, order=None):
    """Build rows greedily in the given order."""
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows


class _ConflictTracker:
    """Incremental tracking of difference conflicts across all rows."""
    
    def __init__(self, rows, M_max):
        self.M_max = M_max
        self.diff_counts = [0] * (M_max + 1)
        self.num_dupes = 0
        self.rows = [list(r) for r in rows]
        
        # Initialize counts
        for row in self.rows:
            self._add_row_diffs(row)
        
        # Count duplicates
        self.num_dupes = sum(1 for c in self.diff_counts if c > 1)
    
    def _add_row_diffs(self, row):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= self.M_max:
                    old_c = self.diff_counts[d]
                    self.diff_counts[d] = old_c + 1
                    if old_c == 1 and self.diff_counts[d] == 2:
                        self.num_dupes += 1
    
    def _remove_row_diffs(self, row):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= self.M_max:
                    old_c = self.diff_counts[d]
                    self.diff_counts[d] = old_c - 1
                    if old_c == 2 and self.diff_counts[d] == 1:
                        self.num_dupes -= 1
    
    def _calc_delta_dupes(self, ri, ei, new_val):
        """Calculate change in num_dupes if we move element ei in row ri to new_val."""
        row = self.rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return 0
        
        delta = 0
        
        # Remove old diffs involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= self.M_max:
                c = self.diff_counts[d]
                if c == 2:
                    delta -= 1  # 2->1 removes a duplicate value
        
        # Add new diffs involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= self.M_max:
                c = self.diff_counts[d]
                if c == 1:
                    delta += 1  # 1->2 creates a duplicate value
        
        return delta
    
    def _apply_move(self, ri, ei, new_val):
        """Apply a move and update diff counts."""
        row = self.rows[ri]
        old_val = row[ei]
        
        # Remove old diffs involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= self.M_max:
                old_c = self.diff_counts[d]
                self.diff_counts[d] = old_c - 1
                if old_c == 2 and self.diff_counts[d] == 1:
                    self.num_dupes -= 1
        
        self.rows[ri][ei] = new_val
        
        # Add new diffs involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= self.M_max:
                old_c = self.diff_counts[d]
                self.diff_counts[d] = old_c + 1
                if old_c == 1 and self.diff_counts[d] == 2:
                    self.num_dupes += 1
    
    def _is_valid(self):
        return self.num_dupes == 0
    
    def _get_scope(self):
        return max(max(r) for r in self.rows)
    
    def _get_rows(self):
        return [list(r) for r in self.rows]


def _local_search_reduce_fast(tracker, n_rows, k, rng, time_budget):
    """
    Fast local search using incremental conflict tracking.
    Tries to reduce the maximum value by moving elements down.
    """
    best_rows = tracker._get_rows()
    best_max = tracker._get_scope()
    start = time.time()
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the maximum element
        max_row = -1
        max_val = -1
        for r in range(n_rows):
            if tracker.rows[r][-1] > max_val:
                max_val = tracker.rows[r][-1]
                max_row = r
        
        if max_row == -1:
            break
        
        row = tracker.rows[max_row]
        improved = False
        
        # Try to reduce the last element
        for new_val in range(row[-1] - 1, row[-2], -1):
            if new_val <= row[-2]:
                continue
            
            # Check if move is valid
            delta = tracker._calc_delta_dupes(max_row, k, new_val)
            if delta >= 0:  # No new conflicts
                tracker._apply_move(max_row, k, new_val)
                new_max = tracker._get_scope()
                if new_max < best_max:
                    best_max = new_max
                    best_rows = tracker._get_rows()
                improved = True
                break
        
        if not improved:
            # Try reducing a random non-last element in a random row
            r = rng.randrange(n_rows)
            if len(tracker.rows[r]) <= 2:
                break
            pos = rng.randrange(1, len(tracker.rows[r]) - 1)
            row = tracker.rows[r]
            old = row[pos]
            
            for new_val in range(old - 1, row[pos - 1], -1):
                if new_val <= row[pos - 1]:
                    continue
                if pos < k and new_val >= row[pos + 1]:
                    continue
                
                delta = tracker._calc_delta_dupes(r, pos, new_val)
                if delta >= 0:
                    tracker._apply_move(r, pos, new_val)
                    new_max = tracker._get_scope()
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = tracker._get_rows()
                    break
    
    return best_rows, best_max


def _sa_optimize_fast(tracker, n_rows, k, rng, time_budget):
    """
    Simulated annealing with incremental conflict tracking.
    """
    best_rows = tracker._get_rows()
    best_max = tracker._get_scope()
    start = time.time()
    T = max(1.0, best_max * 0.1)
    T_min = 0.5
    cooling = 0.997
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        row = tracker.rows[r]
        if len(row) <= 1:
            T *= cooling
            continue
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        span = max(1, min(15, int(T)))
        delta = rng.randint(-span, span)
        new_val = old_val + delta
        
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        delta_dupes = tracker._calc_delta_dupes(r, pos, new_val)
        
        if delta_dupes <= 0:
            # No new conflicts or fewer
            tracker._apply_move(r, pos, new_val)
            cur_max = tracker._get_scope()
            if cur_max < best_max:
                best_max = cur_max
                best_rows = tracker._get_rows()
        else:
            # Accept with some probability
            if delta_dupes <= 3 and rng.random() < 0.05 * (T / 100.0):
                tracker._apply_move(r, pos, new_val)
        
        T *= cooling
    
    return best_rows, best_max


def _iterated_local_search_fast(rows, n_rows, k, rng, time_budget):
    """
    Iterated Local Search with fast conflict tracking.
    """
    M_max = max(max(r) for r in rows) * 2 + 50
    tracker = _ConflictTracker(rows, M_max)
    
    best_rows = tracker._get_rows()
    best_max = tracker._get_scope()
    start = time.time()
    
    current_rows = [list(r) for r in rows]
    
    max_iter = 150
    per_iter_time = time_budget / max_iter if max_iter > 0 else 1.0
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Kick: perturb the current solution
        num_kicks = rng.randint(2, 5)
        kicked_rows = [list(r) for r in current_rows]
        
        for _ in range(num_kicks):
            r = rng.randrange(n_rows)
            row = kicked_rows[r]
            if len(row) <= 1:
                continue
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            span = rng.randint(3, 10)
            new_val = old_val + rng.randint(-span, span)
            
            if new_val <= row[pos - 1] or new_val <= 0:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
            
            row[pos] = new_val
        
        # Check validity after kick
        if not _validate(kicked_rows):
            kicked_rows = [list(r) for r in best_rows]
        
        # Create tracker for kicked solution
        M_max_k = max(max(r) for r in kicked_rows) * 2 + 50
        tracker_k = _ConflictTracker(kicked_rows, M_max_k)
        
        # Run local search on the kicked solution
        ls_rows, ls_max = _local_search_reduce_fast(tracker_k, n_rows, k, rng, per_iter_time)
        
        if ls_max < best_max:
            best_max = ls_max
            best_rows = [list(r) for r in ls_rows]
            current_rows = [list(r) for r in ls_rows]
        else:
            current_rows = [list(r) for r in ls_rows]
        
        # Occasionally restart from best
        if it % 10 == 0 and it > 0:
            current_rows = [list(r) for r in best_rows]
    
    return best_rows, best_max


def _backtrack_with_order(n_rows, k, max_val, rng, time_budget=15.0):
    """
    Backtracking that tries different row orderings.
    """
    start = time.time()
    used = set()
    rows = [None] * n_rows
    
    order = list(range(n_rows))
    rng.shuffle(order)
    
    def build_single_row(used_so_far, time_limit):
        """Build one row given the current used set."""
        if time.time() - start > time_limit:
            return None
        
        row = [0]
        used_row = set()
        
        def extend(current_row, current_used):
            if time.time() - start > time_limit:
                return None
            
            if len(current_row) == k + 1:
                return list(current_row)
            
            last = current_row[-1]
            for val in range(last + 1, max_val + 1):
                new_diffs = set()
                ok = True
                for a in current_row:
                    d = val - a
                    if d in current_used or d in used_so_far:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                
                result = extend(current_row + [val], current_used | new_diffs)
                if result is not None:
                    return result
            
            return None
        
        return extend(row, used_row)
    
    for r in order:
        if time.time() - start > time_budget:
            return None
        
        row = build_single_row(used, time_budget)
        if row is None:
            return None
        rows[r] = row
        used |= _diffs_of_row(row)
    
    return rows


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Greedy baseline with several orderings
    num_greedy = 20
    for t in range(num_greedy):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, rng, order)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)

    # Phase 2: Backtracking with binary search on max_val
    if _time_left() > 30.0:
        total_diffs = n_rows * k * (k + 1) // 2
        lo = max(total_diffs, k)
        hi = best_max
        
        bt_start = time.time()
        while lo <= hi and _time_left() > 10.0 and time.time() - bt_start < 60.0:
            mid = (lo + hi) // 2
            if _time_left() < 5.0:
                break
            
            bt_rows = _backtrack_with_order(n_rows, k, mid, rng, time_budget=8.0)
            if bt_rows is not None:
                if _validate(bt_rows):
                    cur_max = max(max(r) for r in bt_rows)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in bt_rows]
                    hi = mid - 1
                else:
                    hi = mid - 1
            else:
                lo = mid + 1
    
    # Phase 3: Fast Iterated Local Search
    if _time_left() > 20.0:
        budget = min(60.0, _time_left() - 10.0)
        ils_rows, ils_max = _iterated_local_search_fast(best_rows, n_rows, k, rng, budget)
        if ils_max < best_max:
            best_max = ils_max
            best_rows = ils_rows

    # Phase 4: Fast Simulated Annealing polish
    if _time_left() > 15.0:
        budget = min(30.0, _time_left() - 5.0)
        M_max = best_max * 2 + 50
        tracker = _ConflictTracker(best_rows, M_max)
        sa_rows, sa_max = _sa_optimize_fast(tracker, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Phase 5: Final local search polish
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 2.0)
        M_max = best_max * 2 + 50
        tracker = _ConflictTracker(best_rows, M_max)
        ls_rows, ls_max = _local_search_reduce_fast(tracker, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Verify
    if not _validate(best_rows):
        best_rows = _greedy_build(n_rows, k, rng)

    return best_rows
