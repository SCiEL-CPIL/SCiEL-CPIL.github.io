"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Counting lower bound on the largest entry (informational)
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    best_rows = None
    best_max = float('inf')

    # ---- Constructor: pure greedy (baseline) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # ---- Validation ----
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # ---- Simulated Annealing Components ----
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def calculate_energy(rows):
        # Minimize sum of max entries to encourage compactness
        return sum(r[-1] for r in rows)

    def rebuild_row(used_diffs, k, max_val_limit):
        """
        Greedily construct a row of length k+1 starting at 0,
        avoiding conflicts with used_diffs.
        Returns the row or None if it cannot be completed within max_val_limit.
        """
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val_limit:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
        return row

    def simulated_annealing(initial_rows, n_rows, k, rng):
        rows = [list(r) for r in initial_rows]
        used_diffs = set()
        for row in rows:
            used_diffs |= get_diffs(row)
        
        current_energy = calculate_energy(rows)
        best_rows = [list(r) for r in rows]
        best_energy = current_energy
        best_max = max(r[-1] for r in rows)

        # SA Parameters
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        max_iters = 50000

        for i in range(max_iters):
            if _time_left() < 5.0:
                break
            if T < T_min:
                break

            # Pick a random row to regenerate
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = get_diffs(old_row)
            
            # Remove old diffs from global set
            used_diffs -= old_diffs

            # Try to rebuild the row
            # Limit the max value to something reasonable (current max + 20)
            current_max_global = max(r[-1] for r in rows)
            max_val_limit = current_max_global + 20
            
            new_row = rebuild_row(used_diffs, k, max_val_limit)
            
            if new_row is None:
                # Revert
                used_diffs |= old_diffs
                continue

            # Calculate new energy
            new_energy = current_energy - old_row[-1] + new_row[-1]
            
            # Check validity (rebuild_row ensures no conflict with used_diffs, 
            # and row is increasing by construction)
            
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept move
                rows[r_idx] = new_row
                used_diffs |= get_diffs(new_row)
                current_energy = new_energy
                
                cur_max = max(r[-1] for r in rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in rows]
            else:
                # Revert
                used_diffs |= old_diffs

            T *= cooling_rate

        return best_rows, best_max

    rng = random.Random(42)

    # Baseline: deterministic greedy
    baseline = greedy_construct(n_rows, k)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)

    # Run Simulated Annealing
    if _time_left() > 10.0:
        sa_rows, sa_max = simulated_annealing(baseline, n_rows, k, rng)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)

    return best_rows
