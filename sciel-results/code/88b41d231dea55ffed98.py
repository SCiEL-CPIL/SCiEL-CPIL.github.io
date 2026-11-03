import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # --- State Management for Efficient Cost Updates ---
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            for ri, r in enumerate(self.rows):
                self.row_max[ri] = r[-1]
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
            return self.collisions, self.max_val
        
        def is_valid(self):
            return self.collisions == 0
            
        def _update_max_after_change(self, r_idx, old_val, new_val):
            """Update max_val and row_max after changing rows[r_idx]."""
            old_row_max = self.row_max[r_idx]
            # New max of this row
            if new_val > old_row_max:
                new_row_max = new_val
            else:
                # Need to find max of remaining elements
                # Since row is strictly increasing, if old_val was the max (i.e., last element or was max),
                # we need to check. Actually, old_val might not be the row max.
                # The row max is the last element. If we changed an element that isn't the last,
                # and new_val < old_row_max, row_max stays the same.
                # If we changed the last element, new_val is the new row max.
                # If we changed a non-last element and new_val > old_row_max, that can't happen
                # because row is strictly increasing and last element is the max.
                # So: if i_idx == row_len-1, new_row_max = new_val
                # else if new_val > old_row_max: impossible (strictly increasing)
                # else: new_row_max = old_row_max
                new_row_max = old_row_max
            self.row_max[r_idx] = new_row_max
            
            # Update global max
            if new_row_max > self.max_val:
                self.max_val = new_row_max
            elif old_row_max == self.max_val:
                # Need to recompute global max
                self.max_val = max(self.row_max)
        
        def try_move(self, r_idx, i_idx, new_val):
            """
            Attempt to change rows[r_idx][i_idx] to new_val.
            Returns (old_val, delta_collisions, delta_max) if successful, else None.
            """
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return None
            
            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return None
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return None
            
            # Calculate delta in collisions and max
            old_row = self.rows[r_idx]
            delta_collisions = 0
            old_max = self.max_val
            
            # Identify affected differences
            affected_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d_old = abs(old_val - old_row[j])
                    d_new = abs(new_val - old_row[j])
                    affected_diffs.append((d_old, d_new))
            
            # Calculate collision delta
            changes = {}
            for d_old, d_new in affected_diffs:
                if d_old == d_new:
                    continue
                if d_old not in changes:
                    changes[d_old] = 0
                changes[d_old] -= 1
                if d_new not in changes:
                    changes[d_new] = 0
                changes[d_new] += 1
            
            for d, delta in changes.items():
                current_count = self.diff_count.get(d, 0)
                new_count = current_count + delta
                if new_count < 0:
                    return None
                old_contrib = max(0, current_count - 1)
                new_contrib = max(0, new_count - 1)
                delta_collisions += (new_contrib - old_contrib)
            
            # Apply the move
            self.rows[r_idx][i_idx] = new_val
            
            # Update diff_count
            for d_old, d_new in affected_diffs:
                if d_old == d_new:
                    continue
                if d_old in self.diff_count:
                    self.diff_count[d_old] -= 1
                    if self.diff_count[d_old] == 0:
                        del self.diff_count[d_old]
                if d_new in self.diff_count:
                    self.diff_count[d_new] += 1
                else:
                    self.diff_count[d_new] = 1
            
            self.collisions += delta_collisions
            
            # Update max_val
            self._update_max_after_change(r_idx, old_val, new_val)
            delta_max = self.max_val - old_max
            
            return old_val, delta_collisions, delta_max

        def undo_move(self, r_idx, i_idx, old_val, delta_collisions, delta_max):
            """Undo a move."""
            new_val = self.rows[r_idx][i_idx]
            
            # Revert diff_count
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d_new = abs(new_val - old_row[j])
                    d_old = abs(old_val - old_row[j])
                    if d_new == d_old:
                        continue
                    if d_new in self.diff_count:
                        self.diff_count[d_new] -= 1
                        if self.diff_count[d_new] == 0:
                            del self.diff_count[d_new]
                    if d_old in self.diff_count:
                        self.diff_count[d_old] += 1
                    else:
                        self.diff_count[d_old] = 1
            
            self.rows[r_idx][i_idx] = old_val
            self.collisions -= delta_collisions
            
            # Update max_val
            self._update_max_after_change(r_idx, new_val, old_val)

    # --- Initialization ---
    
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
    
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_state = SAState(baseline_rows)
    best_collisions, best_max = best_state.get_cost()
    
    current_state = SAState(baseline_rows)
    current_collisions, current_max = current_state.get_cost()
    
    T = 100.0
    cooling = 0.999
    last_improvement_time = start_time
    
    while time.time() < deadline:
        if time.time() - last_improvement_time > 10.0:
            # Stagnation: Perturb current state
            for _ in range(n_rows * row_len):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                lower = current_state.rows[r_idx][i_idx - 1] + 1
                upper = current_state.rows[r_idx][i_idx + 1] - 1 if i_idx < row_len - 1 else current_max + 10
                if lower < upper:
                    new_val = random.randint(lower, upper)
                    res = current_state.try_move(r_idx, i_idx, new_val)
                    if res:
                        _, d_col, _ = res
                        if d_col <= 0 or random.random() < math.exp(-d_col * 10 / T):
                            pass
                        else:
                            current_state.undo_move(r_idx, i_idx, res[0], res[1], res[2])
            last_improvement_time = time.time()
            T = max(T, 50.0)
        
        # Choose move
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, row_len - 1)
        
        lower = current_state.rows[r_idx][i_idx - 1] + 1
        if i_idx < row_len - 1:
            upper = current_state.rows[r_idx][i_idx + 1] - 1
        else:
            upper = current_max + 5
        
        if lower > upper:
            continue
            
        move_made = False
        for _ in range(5):
            new_val = random.randint(lower, upper)
            if new_val == current_state.rows[r_idx][i_idx]:
                continue
                
            res = current_state.try_move(r_idx, i_idx, new_val)
            if res is None:
                continue
                
            old_val, d_col, d_max = res
            new_collisions = current_collisions + d_col
            new_max = current_max + d_max
            
            accepted = False
            if new_collisions < current_collisions:
                accepted = True
            elif new_collisions == current_collisions:
                if new_max < current_max:
                    accepted = True
                elif new_max == current_max:
                    accepted = random.random() < 0.5
                else:
                    if current_collisions == 0:
                        delta = new_max - current_max
                        if delta < 10:
                            accepted = random.random() < math.exp(-delta / T)
                    else:
                        delta = new_max - current_max
                        if delta < 50:
                            accepted = random.random() < math.exp(-delta / (T * 2))
            else:
                delta = new_collisions - current_collisions
                penalty = delta * 100
                if penalty < 1000:
                    accepted = random.random() < math.exp(-penalty / T)
            
            if accepted:
                current_collisions = new_collisions
                current_max = new_max
                move_made = True
                if (current_collisions, current_max) < (best_collisions, best_max):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_rows = [r[:] for r in current_state.rows]
                    last_improvement_time = time.time()
                break
            else:
                current_state.undo_move(r_idx, i_idx, old_val, d_col, d_max)
        
        T *= cooling
        if T < 1.0:
            T = 1.0

    # --- Final Polish: Try to reduce the scope further ---
    # Use a dedicated local search that focuses on reducing max_val
    polish_deadline = start_time + 118.0
    polish_state = SAState(best_rows)
    polish_collisions, polish_max = polish_state.get_cost()
    
    if polish_collisions == 0:
        # Try to reduce the largest entry
        # Find the row with the maximum value
        for _ in range(200000):
            if time.time() > polish_deadline:
                break
            
            # Find the row and position with the largest value
            max_r = 0
            max_val = 0
            for ri in range(n_rows):
                if polish_state.rows[ri][-1] > max_val:
                    max_val = polish_state.rows[ri][-1]
                    max_r = ri
            
            # Try to reduce the last element of the max row
            # Or try to reduce any element that contributes to the max
            r_idx = max_r
            # Try different positions
            for i_idx in range(row_len - 1, 0, -1):
                if time.time() > polish_deadline:
                    break
                
                current_val = polish_state.rows[r_idx][i_idx]
                lower = polish_state.rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = polish_state.rows[r_idx][i_idx + 1] - 1
                else:
                    upper = current_val - 1  # Only try to reduce
                
                if lower > upper:
                    continue
                
                # Try values from current_val-1 down to lower
                for new_val in range(current_val - 1, lower - 1, -1):
                    if time.time() > polish_deadline:
                        break
                    
                    res = polish_state.try_move(r_idx, i_idx, new_val)
                    if res is None:
                        continue
                    
                    old_v, d_col, d_max = res
                    if d_col == 0 and d_max < 0:
                        # Accept: reduces max without creating collisions
                        polish_max += d_max
                        if polish_max < best_max:
                            best_max = polish_max
                            best_rows = [r[:] for r in polish_state.rows]
                        break
                    else:
                        polish_state.undo_move(r_idx, i_idx, old_v, d_col, d_max)
                        break  # If it doesn't work, try next position
        
        # Also try random moves to escape local optima in the valid region
        T_polish = 10.0
        for _ in range(100000):
            if time.time() > polish_deadline:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            current_val = polish_state.rows[r_idx][i_idx]
            lower = polish_state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = polish_state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = current_val  # Don't increase the last element
            
            if lower > upper:
                continue
            
            new_val = random.randint(lower, upper)
            if new_val == current_val:
                continue
            
            res = polish_state.try_move(r_idx, i_idx, new_val)
            if res is None:
                continue
            
            old_v, d_col, d_max = res
            if d_col == 0 and d_max <= 0:
                polish_max += d_max
                if polish_max < best_max:
                    best_max = polish_max
                    best_rows = [r[:] for r in polish_state.rows]
            else:
                polish_state.undo_move(r_idx, i_idx, old_v, d_col, d_max)

    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(7, 5)
    diffs = set()
    valid = True
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    max_val = max(max(r) for r in rows)
    print(f"Valid: {valid}, Max: {max_val}")
