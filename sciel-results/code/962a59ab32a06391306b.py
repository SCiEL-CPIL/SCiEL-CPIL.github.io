"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
import math
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing with local search for robust exploration.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    # Greedy baseline
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def get_all_diffs(rows):
        """Get all differences across all rows."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_diffs(row))
        return all_diffs
    
    def count_conflicts(rows):
        """Count number of duplicate differences."""
        diffs = get_all_diffs(rows)
        freq = {}
        for d in diffs:
            freq[d] = freq.get(d, 0) + 1
        conflicts = 0
        for d, c in freq.items():
            if c > 1:
                conflicts += c - 1
        return conflicts
    
    def is_valid(rows, n_rows, k):
        """Check if rows are valid."""
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_max_entry(rows):
        """Get the largest entry across all rows."""
        return max(max(r) for r in rows)
    
    def random_rows(n_rows, k, max_val):
        """Generate random rows with entries in [1, max_val]."""
        rows = []
        for _ in range(n_rows):
            # Generate k random increasing values from 1..max_val
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            row = [0] + candidates[:k]
            row.sort()
            rows.append(row)
        return rows
    
    def local_search(rows, n_rows, k, max_val, deadline, max_iter=5000):
        """
        Local search: try to reduce conflicts by moving individual elements.
        Returns improved rows or None.
        """
        current_rows = [list(r) for r in rows]
        current_conflicts = count_conflicts(current_rows)
        
        if current_conflicts == 0:
            return current_rows
        
        best_rows = [list(r) for r in current_rows]
        best_conflicts = current_conflicts
        best_max = get_max_entry(current_rows)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random row and position to modify
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # positions 1..k (0 is fixed at 0)
            
            old_val = current_rows[row_idx][pos]
            
            # Try a new value
            new_val = random.randint(1, max_val)
            
            # Ensure increasing order
            if pos > 1 and new_val <= current_rows[row_idx][pos - 1]:
                continue
            if pos < k and new_val >= current_rows[row_idx][pos + 1]:
                continue
            
            current_rows[row_idx][pos] = new_val
            
            # Check if still increasing
            if not all(current_rows[row_idx][i] < current_rows[row_idx][i+1] for i in range(k)):
                current_rows[row_idx][pos] = old_val
                continue
            
            new_conflicts = count_conflicts(current_rows)
            new_max = get_max_entry(current_rows)
            
            # Accept if better or equal
            if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_max < best_max):
                best_conflicts = new_conflicts
                best_max = new_max
                best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                current_rows[row_idx][pos] = old_val
        
        if best_conflicts == 0:
            return best_rows
        return None
    
    def simulated_annealing(n_rows, k, max_val, deadline, initial_temp=100.0, cooling=0.995):
        """
        Simulated annealing to find valid rows with minimal max entry.
        """
        # Start from a random feasible-ish configuration
        current_rows = random_rows(n_rows, k, max_val)
        current_conflicts = count_conflicts(current_rows)
        
        best_rows = [list(r) for r in current_rows]
        best_conflicts = current_conflicts
        best_max = get_max_entry(current_rows)
        
        temp = initial_temp
        
        while time.time() < deadline and temp > 0.1:
            # Pick a random row and position to modify
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = current_rows[row_idx][pos]
            new_val = random.randint(1, max_val)
            
            # Ensure increasing order
            valid_move = True
            if pos > 1 and new_val <= current_rows[row_idx][pos - 1]:
                valid_move = False
            if pos < k and new_val >= current_rows[row_idx][pos + 1]:
                valid_move = False
            
            if not valid_move:
                temp *= cooling
                continue
            
            current_rows[row_idx][pos] = new_val
            
            new_conflicts = count_conflicts(current_rows)
            new_max = get_max_entry(current_rows)
            
            delta_conflicts = new_conflicts - current_conflicts
            delta_max = new_max - best_max
            
            # Acceptance probability
            if delta_conflicts < 0 or (delta_conflicts == 0 and delta_max < 0):
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_max < best_max):
                    best_conflicts = new_conflicts
                    best_max = new_max
                    best_rows = [list(r) for r in current_rows]
            else:
                # Compute acceptance probability based on total "cost"
                # Cost = conflicts * 1000 + max_entry
                old_cost = current_conflicts * 1000 + get_max_entry(current_rows)
                new_cost = new_conflicts * 1000 + new_max
                delta_cost = new_cost - old_cost
                
                if temp > 0 and delta_cost < 0 or random.random() < math.exp(-delta_cost / temp):
                    current_conflicts = new_conflicts
                    if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_max < best_max):
                        best_conflicts = new_conflicts
                        best_max = new_max
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[row_idx][pos] = old_val
            
            temp *= cooling
        
        # Run local search from best solution
        if best_conflicts > 0:
            result = local_search(best_rows, n_rows, k, max_val, deadline, max_iter=2000)
            if result is not None:
                best_rows = result
                best_conflicts = 0
        
        return best_rows, best_conflicts
    
    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    best_conflicts = 0  # Greedy is valid
    
    # Strategy: try max_entry values from lb upward with time budgets
    max_search_range = min(greedy_max, lb + 30)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        
        if remaining_time < 2.0:
            break
        
        # Allocate time based on distance from lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(25.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(8.0, remaining_time)
        else:
            budget = min(4.0, remaining_time)
        
        if budget < 1.0:
            break
        
        deadline = time.time() + budget
        
        # Run simulated annealing
        sa_rows, sa_conflicts = simulated_annealing(n_rows, k, max_entry, deadline)
        
        if sa_conflicts == 0:
            sa_max = get_max_entry(sa_rows)
            if sa_max <= max_entry:
                best_rows = sa_rows
                best_max = sa_max
                break  # Found valid solution for this max_entry
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                deadline = time.time() + min(remaining_time, 20.0)
                sa_rows, sa_conflicts = simulated_annealing(n_rows, k, mid, deadline)
                if sa_conflicts == 0:
                    sa_max = get_max_entry(sa_rows)
                    if sa_max <= mid:
                        best_rows = sa_rows
                        best_max = sa_max
    
    # Final validation and cleanup
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
