"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing with bitset-based conflict detection to find
    tight packings. The search space is the assignment of row elements, and
    the cost function penalizes both the maximum entry and the number of
    duplicate differences.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 118.0  # safety margin before 120s kill
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    def compute_row_diffs(row):
        """Compute all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def count_conflicts(rows):
        """Count the number of duplicate differences across all rows."""
        from collections import Counter
        all_diffs = []
        for row in rows:
            all_diffs.extend(compute_row_diffs(row))
        counter = Counter(all_diffs)
        # Number of excess duplicates
        return sum(c - 1 for c in counter.values() if c > 1)
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
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
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    # Greedy baseline
    def greedy_construct():
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
    
    best_rows = greedy_construct()
    best_max = get_max_entry(best_rows)
    best_conflicts = count_conflicts(best_rows)
    
    # If greedy already has no conflicts, that's a valid solution
    if best_conflicts == 0:
        best_valid = True
    else:
        best_valid = False
    
    # Simulated annealing
    def simulated_annealing(max_iters, initial_temp, cooling_rate):
        nonlocal best_rows, best_max, best_conflicts, best_valid
        
        # Start from greedy solution
        current_rows = [list(r) for r in best_rows]
        current_max = best_max
        current_conflicts = best_conflicts
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Random move: pick a random row and a random position, perturb it
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # positions 1..k (position 0 is always 0)
            
            # Generate a new value for this position
            # Try to keep the row increasing
            old_val = current_rows[row_idx][pos]
            
            # Sample new value: biased toward nearby values or random in range
            if random.random() < 0.7:
                # Local perturbation
                delta = random.randint(-10, 10)
                new_val = old_val + delta
            else:
                # Larger jump
                new_val = random.randint(1, current_max + 5)
            
            # Ensure new_val > previous element
            if pos > 0 and new_val <= current_rows[row_idx][pos - 1]:
                new_val = current_rows[row_idx][pos - 1] + 1
            
            # Ensure new_val < next element (or adjust)
            if pos < k and new_val >= current_rows[row_idx][pos + 1]:
                # Shift subsequent elements if needed
                for p in range(pos + 1, k + 1):
                    if current_rows[row_idx][p] <= new_val:
                        current_rows[row_idx][p] = new_val + (p - pos)
                new_val = current_rows[row_idx][pos]
            
            # Apply the change
            old_row = list(current_rows[row_idx])
            current_rows[row_idx][pos] = new_val
            # Re-sort if necessary (shouldn't be needed if we maintained order)
            # But let's be safe
            current_rows[row_idx] = sorted(set(current_rows[row_idx]))
            # Ensure length is k+1 and starts with 0
            if len(current_rows[row_idx]) != k + 1 or current_rows[row_idx][0] != 0:
                # Revert
                current_rows[row_idx] = old_row
                continue
            
            # Compute new cost
            new_max = get_max_entry(current_rows)
            new_conflicts = count_conflicts(current_rows)
            
            # Cost: primarily conflicts, secondarily max entry
            old_cost = current_conflicts * 1000 + current_max
            new_cost = new_conflicts * 1000 + new_max
            
            # Acceptance probability
            if new_cost < old_cost:
                current_conflicts = new_conflicts
                current_max = new_max
            else:
                # Accept with probability exp(-delta/T)
                delta = new_cost - old_cost
                T = initial_temp * (cooling_rate ** it)
                if T > 0 and random.random() < pow(2.718281828, -delta / T):
                    current_conflicts = new_conflicts
                    current_max = new_max
                else:
                    # Revert
                    current_rows[row_idx] = old_row
                    continue
            
            # Update best
            if new_conflicts == 0 and new_max < best_max:
                best_rows = [list(r) for r in current_rows]
                best_max = new_max
                best_conflicts = 0
                best_valid = True
            elif new_conflicts < best_conflicts:
                best_rows = [list(r) for r in current_rows]
                best_conflicts = new_conflicts
                best_max = new_max
                best_valid = (new_conflicts == 0)
        
        return best_rows, best_max, best_conflicts, best_valid
    
    # Run simulated annealing
    # Phase 1: Long run with high temperature
    max_iters = 50000
    initial_temp = 50.0
    cooling_rate = 0.999
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        best_rows, best_max, best_conflicts, best_valid = simulated_annealing(
            max_iters, initial_temp, cooling_rate
        )
    
    # Phase 2: If we have time, try more annealing with lower temperature
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5 and best_conflicts > 0:
        # Try to reduce conflicts further
        max_iters2 = 30000
        initial_temp2 = 20.0
        cooling_rate2 = 0.9995
        best_rows, best_max, best_conflicts, best_valid = simulated_annealing(
            max_iters2, initial_temp2, cooling_rate2
        )
    
    # Validate final solution
    if not validate(best_rows, n_rows, k):
        # Fall back to greedy
        best_rows = greedy_construct()
    
    return best_rows

# EVOLVE-BLOCK-END
