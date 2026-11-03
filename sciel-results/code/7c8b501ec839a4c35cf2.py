"""Difference triangle set (7,5): minimize the largest mark (scope).

Strategy: Simulated Annealing with global diff-counting for O(1) conflict checks,
enabling global scope compression moves (row rebuilds) that escape local optima.
Combines incremental state tracking with multi-phase SA (coarse then fine) and
row-rebuild moves for better exploration.
"""
import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def greedy_construct():
        """Fast greedy construction to get a valid baseline."""
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
    
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            for r in self.rows:
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(self.row_len):
                    for j in range(i + 1, self.row_len):
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
        
        def try_move(self, r_idx, i_idx, new_val):
            """Try to change rows[r_idx][i_idx] to new_val. Return True if applied."""
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False
            
            # Check bounds: strictly increasing
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx - 1]:
                return False
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx + 1]:
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
            self.rows[r_idx][i_idx] = new_val
            
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
        
        def try_row_rebuild(self, r_idx, new_row):
            """
            Try to replace entire row r_idx with new_row.
            new_row must be strictly increasing, start with 0, length row_len.
            Returns True if applied.
            """
            # Validate new_row
            if len(new_row) != self.row_len or new_row[0] != 0:
                return False
            for i in range(self.row_len - 1):
                if new_row[i] >= new_row[i + 1]:
                    return False
            
            old_row = self.rows[r_idx][:]
            
            # Check if new_row has internal distinct differences
            new_diffs = set()
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = new_row[j] - new_row[i]
                    if d in new_diffs:
                        return False
                    new_diffs.add(d)
            
            # Remove old differences
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = old_row[j] - old_row[i]
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            # Apply new row
            self.rows[r_idx] = new_row[:]
            
            # Add new differences
            for d in new_diffs:
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
            
            # Update max_val
            old_max = max(old_row)
            new_max = max(new_row)
            if new_max > self.max_val:
                self.max_val = new_max
            elif old_max == self.max_val and new_max < old_max:
                self.max_val = max(max(r) for r in self.rows)
            
            return True
        
        def undo_row_rebuild(self, r_idx, old_row):
            """Undo a row rebuild by restoring old_row."""
            new_row = self.rows[r_idx][:]
            
            # Remove new differences
            new_diffs = set()
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = new_row[j] - new_row[i]
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            # Restore old row
            self.rows[r_idx] = old_row[:]
            
            # Add old differences
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = old_row[j] - old_row[i]
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
            
            # Update max_val
            old_max = max(old_row)
            new_max = max(new_row)
            if old_max > self.max_val:
                self.max_val = old_max
            elif new_max == self.max_val and old_max < new_max:
                self.max_val = max(max(r) for r in self.rows)
    
    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_collisions, current_max = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_rows = [r[:] for r in state.rows]
        
        T = 50.0
        cooling = 0.9995
        start = time.time()
        
        while time.time() - start < time_limit:
            # Choose a random row and position
            r_idx = random.randint(0, n_rows - 1)
            
            # 20% chance of row rebuild, 80% chance of single element move
            if random.random() < 0.2:
                # Row rebuild: try to construct a better row
                old_row = state.rows[r_idx][:]
                # Generate a candidate row with smaller max
                target_max = best_max - random.randint(1, 5)
                if target_max < 5:
                    target_max = 5
                
                new_row = None
                # Try a few random constructions
                for _ in range(10):
                    candidate = [0]
                    remaining = row_len - 1
                    # Greedily try to fill with small values
                    for _ in range(remaining):
                        found = False
                        for x in range(1, target_max + 1):
                            if x <= candidate[-1]:
                                continue
                            # Check internal distinctness
                            valid = True
                            for a in candidate:
                                d = x - a
                                # Check if this diff is already in candidate
                                for b in candidate:
                                    if b != a and d == abs(x - b):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                candidate.append(x)
                                found = True
                                break
                        if not found:
                            candidate = None
                            break
                    if candidate is not None and len(candidate) == row_len:
                        new_row = candidate
                        break
                
                if new_row is None:
                    continue
                
                if state.try_row_rebuild(r_idx, new_row):
                    new_collisions, new_max = state.get_cost()
                    # Accept if valid or with probability based on temperature
                    if new_collisions == 0:
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [r[:] for r in state.rows]
                    else:
                        # Accept with probability if collisions are low
                        delta_cost = (new_collisions - current_collisions) + (new_max - current_max)
                        if random.random() < math.exp(-delta_cost / T):
                            pass  # Keep the move
                        else:
                            state.undo_row_rebuild(r_idx, old_row)
                current_collisions, current_max = state.get_cost()
            
            else:
                # Single element move
                i_idx = random.randint(1, row_len - 1)
                
                # Determine range for new value
                lower = state.rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = state.rows[r_idx][i_idx + 1] - 1
                else:
                    upper = best_max + 10
                
                if lower > upper:
                    continue
                
                # Try a few random values
                for _ in range(3):
                    new_val = random.randint(lower, upper)
                    old_val = state.rows[r_idx][i_idx]
                    if new_val == old_val:
                        continue
                    
                    if state.try_move(r_idx, i_idx, new_val):
                        new_collisions, new_max = state.get_cost()
                        
                        # Acceptance criteria
                        if new_collisions < current_collisions:
                            # Always accept if fewer collisions
                            pass
                        elif new_collisions == current_collisions:
                            if new_max < current_max:
                                pass  # Accept improvement
                            elif new_max == current_max:
                                # Accept with some probability
                                if random.random() < 0.1:
                                    pass
                                else:
                                    state.undo_move(r_idx, i_idx, old_val)
                            else:
                                # Worse max, accept with probability
                                delta = new_max - current_max
                                if random.random() < math.exp(-delta / T):
                                    pass
                                else:
                                    state.undo_move(r_idx, i_idx, old_val)
                        else:
                            # More collisions, accept with small probability
                            delta = (new_collisions - current_collisions) * 10 + (new_max - current_max)
                            if random.random() < math.exp(-delta / T):
                                pass
                            else:
                                state.undo_move(r_idx, i_idx, old_val)
                        
                        # Update best if valid and improved
                        current_collisions, current_max = state.get_cost()
                        if current_collisions == 0 and current_max < best_max:
                            best_max = current_max
                            best_rows = [r[:] for r in state.rows]
                        break  # Only try one value per iteration
            
            T *= cooling
        
        return best_rows, best_max
    
    # Multi-restart approach
    best_rows_overall = None
    best_max_overall = float('inf')
    
    # First, get a greedy baseline
    baseline_rows = greedy_construct()
    baseline_max = max(max(r) for r in baseline_rows)
    best_rows_overall = [r[:] for r in baseline_rows]
    best_max_overall = baseline_max
    
    # Run SA with multiple restarts
    num_restarts = 5
    time_per_restart = 115.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # For restarts after the first, use a perturbed version of the best solution
        if restart == 0:
            initial_rows = baseline_rows
        else:
            # Perturb the best solution
            initial_rows = [r[:] for r in best_rows_overall]
            # Randomly swap some elements
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 2)
                # Swap with neighbor
                j_idx = i_idx + 1
                # Only swap if it maintains strict increase
                if initial_rows[r_idx][i_idx] < initial_rows[r_idx][j_idx]:
                    # Try a small perturbation instead
                    delta = random.randint(-5, 5)
                    new_val = initial_rows[r_idx][i_idx] + delta
                    if new_val > initial_rows[r_idx][i_idx - 1] and new_val < initial_rows[r_idx][i_idx + 1]:
                        initial_rows[r_idx][i_idx] = new_val
        
        rows, max_val = simulate_annealing(initial_rows, time_per_restart)
        
        if max_val < best_max_overall:
            best_max_overall = max_val
            best_rows_overall = rows
    
    return best_rows_overall
