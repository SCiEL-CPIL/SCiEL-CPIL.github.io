import time
import random
import math
import sys
import numpy as np

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def greedy_construct():
        """Fast greedy construction to get a baseline."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    def csp_build_row(used, row_len):
        """Build a single row using backtracking with constraint propagation."""
        row = [0]
        
        def backtrack(pos, current_row):
            if pos == row_len:
                return True
            # Try candidate values starting from current_row[-1] + 1
            start_val = current_row[-1] + 1
            for val in range(start_val, start_val + 200):
                valid = True
                for a in current_row:
                    d = val - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    # Check forward: can we still complete?
                    # Simple heuristic: check if enough space remains
                    current_row.append(val)
                    if backtrack(pos + 1, current_row):
                        return True
                    current_row.pop()
            return False
        
        if backtrack(1, row):
            return row
        return None

    def build_initial_rows():
        """Build initial rows using greedy + CSP hybrid."""
        used = set()
        rows = []
        for _ in range(n_rows):
            # Try CSP first
            row = csp_build_row(used, row_len)
            if row is None:
                # Fall back to greedy
                row = [0]
                x = 0
                while len(row) < row_len:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
            else:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used.add(row[j] - row[i])
            rows.append(row)
        return rows

    # Get baseline
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)
    best_sum = sum(sum(r) for r in best_rows)

    # Incremental state for SA
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.sum_entries = 0
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.sum_entries = 0
            for r in self.rows:
                self.sum_entries += sum(r)
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in self.diff_count:
                            self.diff_count[d] += 1
                            if self.diff_count[d] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d] = 1
        
        def get_cost(self):
            return self.collisions, self.max_val, self.sum_entries
        
        def is_valid(self):
            return self.collisions == 0
        
        def try_move(self, r_idx, i_idx, new_val):
            """Try to change rows[r_idx][i_idx] to new_val. Return True if bounds OK."""
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False
            
            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return False
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return False
            
            # Remove old differences
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            # Apply change
            old_max = self.max_val
            old_sum = self.sum_entries
            self.rows[r_idx][i_idx] = new_val
            self.sum_entries += new_val - old_val
            
            # Add new differences
            new_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
            
            # Update max_val
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == old_max and new_val < old_max:
                self.max_val = max(max(r) for r in self.rows)
            
            return True
        
        def undo_move(self, r_idx, i_idx, old_val):
            """Undo a move by restoring old_val."""
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return
            
            # Remove new differences
            new_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            # Restore old value
            self.rows[r_idx][i_idx] = old_val
            self.sum_entries += old_val - new_val
            
            # Add old differences
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
            
            # Update max_val
            if old_val > self.max_val:
                self.max_val = old_val
            elif new_val == self.max_val and old_val < new_val:
                self.max_val = max(max(r) for r in self.rows)

    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_collisions, current_max, current_sum = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_sum = current_sum
        best_rows = [r[:] for r in state.rows]
        
        # Adaptive temperature
        T = 30.0
        cooling = 0.9998
        min_T = 0.3
        start = time.time()
        steps = 0
        
        while time.time() - start < time_limit:
            # Choose a random row and position
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            # Determine range for new value
            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = best_max + 20
            
            if lower > upper:
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
            
            # Try a random value
            new_val = random.randint(lower, upper)
            old_val = state.rows[r_idx][i_idx]
            if new_val == old_val:
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
            
            if not state.try_move(r_idx, i_idx, new_val):
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
            
            new_collisions, new_max, new_sum = state.get_cost()
            
            # Cost: (collisions, max_val, sum_entries)
            # Acceptance: prefer fewer collisions, then smaller max, then smaller sum
            if new_collisions < current_collisions:
                accept = True
            elif new_collisions == current_collisions:
                if new_max < current_max:
                    accept = True
                elif new_max == current_max:
                    if new_sum < current_sum:
                        accept = True
                    else:
                        delta = (new_max - current_max) * 10 + (new_sum - current_sum) / 10.0
                        p = math.exp(-delta / max(T, 0.1))
                        accept = random.random() < p
                else:
                    delta = (new_max - current_max) * 10 + (new_sum - current_sum) / 10.0
                    p = math.exp(-delta / max(T, 0.1))
                    accept = random.random() < p
            else:
                # More collisions - rarely accept
                delta = (new_collisions - current_collisions) * 100 + (new_max - current_max) * 10 + (new_sum - current_sum) / 10.0
                p = math.exp(-delta / max(T, 0.1))
                accept = random.random() < p
            
            if accept:
                current_collisions = new_collisions
                current_max = new_max
                current_sum = new_sum
                if (current_collisions < best_collisions or 
                    (current_collisions == best_collisions and current_max < best_max) or
                    (current_collisions == best_collisions and current_max == best_max and current_sum < best_sum)):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_sum = current_sum
                    best_rows = [r[:] for r in state.rows]
            else:
                state.undo_move(r_idx, i_idx, old_val)
            
            T *= cooling
            if T < min_T:
                T = min_T
            steps += 1
            
            # Periodic restart of temperature if stuck
            if steps % 5000 == 0 and current_collisions == best_collisions:
                if random.random() < 0.1:
                    T = max(T, 5.0)
        
        return best_rows, best_max, best_sum, best_collisions

    def local_polish(rows, time_limit):
        """Final high-precision polish: try to reduce max by targeted moves."""
        state = SAState(rows)
        current_collisions, current_max, current_sum = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_sum = current_sum
        best_rows = [r[:] for r in state.rows]
        
        start = time.time()
        T = 2.0
        cooling = 0.9995
        
        while time.time() - start < time_limit:
            # Target the largest entry
            r_idx = random.randint(0, n_rows - 1)
            # Focus on the last few positions
            i_idx = random.randint(max(1, row_len - 3), row_len - 1)
            
            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = state.rows[r_idx][i_idx] - 1  # Try to decrease
            
            if lower > upper:
                # Try to decrease
                upper = state.rows[r_idx][i_idx] - 1
                if lower > upper:
                    T *= cooling
                    continue
            
            new_val = random.randint(lower, upper)
            old_val = state.rows[r_idx][i_idx]
            if new_val == old_val:
                T *= cooling
                continue
            
            if not state.try_move(r_idx, i_idx, new_val):
                T *= cooling
                continue
            
            new_collisions, new_max, new_sum = state.get_cost()
            
            if new_collisions < current_collisions:
                accept = True
            elif new_collisions == current_collisions:
                if new_max < current_max:
                    accept = True
                elif new_max == current_max and new_sum < current_sum:
                    accept = True
                else:
                    delta = (new_max - current_max) * 10 + (new_sum - current_sum) / 10.0
                    p = math.exp(-delta / max(T, 0.1))
                    accept = random.random() < p
            else:
                delta = (new_collisions - current_collisions) * 100 + (new_max - current_max) * 10
                p = math.exp(-delta / max(T, 0.1))
                accept = random.random() < p
            
            if accept:
                current_collisions = new_collisions
                current_max = new_max
                current_sum = new_sum
                if (current_collisions < best_collisions or
                    (current_collisions == best_collisions and current_max < best_max) or
                    (current_collisions == best_collisions and current_max == best_max and current_sum < best_sum)):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_sum = current_sum
                    best_rows = [r[:] for r in state.rows]
            else:
                state.undo_move(r_idx, i_idx, old_val)
            
            T *= cooling
        
        return best_rows, best_max, best_sum, best_collisions

    # Main loop
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Create initial state
        if random.random() < 0.4:
            # Use best rows as starting point
            init_rows = [r[:] for r in best_rows]
        else:
            # Random initialization within a bound
            bound = best_max + 10
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        
        # Run SA for a portion of remaining time
        sa_time = min(remaining * 0.6, 10.0)
        if sa_time < 1.0:
            break
        
        local_rows, local_max, local_sum, local_collisions = simulate_annealing(init_rows, sa_time)
        
        # Check if better
        if (local_collisions < 0) or \
           (local_collisions == 0 and local_max < best_max) or \
           (local_collisions == 0 and local_max == best_max and local_sum < best_sum):
            best_rows = local_rows
            best_max = local_max
            best_sum = local_sum
        
        # If we have time, do a polish pass
        remaining = deadline - time.time()
        if remaining > 5.0 and best_collisions == 0:
            polish_time = min(remaining * 0.3, 5.0)
            polished_rows, polished_max, polished_sum, polished_coll = local_polish(best_rows, polish_time)
            if (polished_coll < 0) or \
               (polished_coll == 0 and polished_max < best_max) or \
               (polished_coll == 0 and polished_max == best_max and polished_sum < best_sum):
                best_rows = polished_rows
                best_max = polished_max
                best_sum = polished_sum

    # Final validation and return
    # Ensure validity
    used = set()
    valid = True
    for r in best_rows:
        if len(r) != row_len:
            valid = False
            break
        if r[0] != 0:
            valid = False
            break
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    valid = False
                    break
                used.add(d)
        if not valid:
            break
        for i in range(len(r)-1):
            if r[i] >= r[i+1]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Return greedy baseline as fallback
        return baseline_rows
    
    return best_rows
