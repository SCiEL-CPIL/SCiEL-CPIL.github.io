"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs

    def get_row_diffs(row):
        """Return list of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_cost(rows):
        """
        Compute a cost function: 
        - Large penalty for duplicate differences
        - Penalty for large scope
        Returns (total_cost, num_conflicts, max_entry)
        """
        all_diffs = []
        max_entry = 0
        for row in rows:
            for val in row:
                if val > max_entry:
                    max_entry = val
            all_diffs.extend(get_row_diffs(row))
        
        # Count conflicts: number of diffs that appear more than once
        from collections import Counter
        diff_counts = Counter(all_diffs)
        num_conflicts = sum(c - 1 for c in diff_counts.values() if c > 1)
        
        # Cost: heavy penalty for conflicts, plus scope
        conflict_penalty = 10000 * num_conflicts
        scope_penalty = max_entry
        total_cost = conflict_penalty + scope_penalty
        
        return total_cost, num_conflicts, max_entry

    def is_valid(rows):
        """Check if all differences are distinct."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    def scope_of(rows):
        return max(max(row) for row in rows)

    # Initialize with a random valid-ish configuration
    rng = random.Random(12345)
    
    def random_initial_rows():
        """Generate n_rows random rows that are increasing and start at 0."""
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                # Try random step sizes
                for _ in range(1000):
                    step = rng.randint(1, 15)
                    x += step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                else:
                    # Fallback: just increment
                    x += 1
                    row.append(x)
                    used_diffs |= {x - a for a in row}
            rows.append(row)
        return rows

    # Get initial solution
    current_rows = random_initial_rows()
    current_cost, current_conflicts, current_max = compute_cost(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    best_scope = current_max

    # Simulated annealing
    temperature = 50.0
    cooling_rate = 0.9995
    min_temperature = 0.1
    
    while temperature > min_temperature and time.time() < DEADLINE:
        # Generate a neighbor by modifying one element in one row
        row_idx = rng.randint(0, n_rows - 1)
        if num_elements <= 1:
            break
        # Don't move the first element (must be 0)
        elem_idx = rng.randint(1, num_elements - 1)
        
        old_val = current_rows[row_idx][elem_idx]
        
        # Choose new value: random nearby or random in range
        if rng.random() < 0.3:
            # Small perturbation
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(1, current_max + 10)
        
        # Ensure the row remains strictly increasing
        if elem_idx > 0 and new_val <= current_rows[row_idx][elem_idx - 1]:
            continue
        if elem_idx < num_elements - 1 and new_val >= current_rows[row_idx][elem_idx + 1]:
            continue
        if new_val <= 0:
            continue
        
        # Apply move
        current_rows[row_idx][elem_idx] = new_val
        new_cost, new_conflicts, new_max = compute_cost(current_rows)
        
        # Accept or reject based on simulated annealing
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < pow(2.718281828, -delta_cost / temperature):
            current_cost = new_cost
            current_conflicts = new_conflicts
            current_max = new_max
            
            # Update best if valid
            if current_conflicts == 0:
                if current_max < best_scope:
                    best_scope = current_max
                    best_cost = current_cost
                    best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            current_rows[row_idx][elem_idx] = old_val
        
        # Cool down
        temperature *= cooling_rate
        
        # Periodically check time and try to restart if stuck
        if time.time() > DEADLINE - 5:
            break
    
    # Final validation
    if is_valid(best_rows):
        return best_rows
    else:
        # Fallback: generate a valid solution using greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
