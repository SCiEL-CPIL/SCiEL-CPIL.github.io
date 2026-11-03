"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import sys
import time
import copy

def construct(n_rows, k):
    """
    Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences a_j - a_i (i < j) are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing with constraint repair to find compact solutions.
    """
    start_time = time.time()
    max_time = 110.0  # Leave some time for overhead
    
    # Counting bound: minimum possible max value
    counting_bound = n_rows * (k + 1) * k // 2
    
    # Start with a greedy solution
    best_rows = _greedy_solution(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    if best_max <= counting_bound:
        return best_rows
    
    # Simulated annealing to improve
    current_rows = _greedy_solution(n_rows, k)
    current_max = max(max(row) for row in current_rows)
    
    best_rows = copy.deepcopy(current_rows)
    best_max = current_max
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    max_iterations = 50000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time:
            break
        
        # Generate a neighbor solution
        neighbor_rows = _generate_neighbor(current_rows, n_rows, k)
        neighbor_max = max(max(row) for row in neighbor_rows)
        
        # Check if neighbor is valid
        if not _is_valid(neighbor_rows, n_rows, k):
            # Try to repair
            neighbor_rows = _repair_solution(neighbor_rows, n_rows, k)
            if neighbor_rows is None:
                continue
            neighbor_max = max(max(row) for row in neighbor_rows)
        
        # Accept if better or with probability based on temperature
        delta = neighbor_max - current_max
        if delta < 0:
            # Better solution
            current_rows = neighbor_rows
            current_max = neighbor_max
            if current_max < best_max:
                best_max = current_max
                best_rows = copy.deepcopy(current_rows)
                if best_max <= counting_bound:
                    break
        else:
            # Worse or equal - accept with probability exp(-delta/T)
            if T > T_min and random.random() < math.exp(-delta / T):
                current_rows = neighbor_rows
                current_max = neighbor_max
        
        # Cool down
        T *= cooling_rate
    
    return best_rows

def _greedy_solution(n_rows, k):
    """
    Generate a valid solution using greedy approach.
    Returns a list of n_rows rows, each with k+1 increasing integers starting at 0.
    """
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
                x += 1
            else:
                x += 1
        rows.append(row)
    
    return rows

def _is_valid(rows, n_rows, k):
    """
    Check if the solution is valid:
    1. Each row has k+1 elements starting at 0
    2. Each row is strictly increasing
    3. All positive differences across all rows are distinct
    """
    if len(rows) != n_rows:
        return False
    
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        # Check all differences are unique
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
                used_diffs.add(d)
    
    return True

def _generate_neighbor(rows, n_rows, k):
    """
    Generate a neighbor solution by making a small random change.
    Strategies:
    1. Shift an element in a row
    2. Swap two rows
    3. Perturb a row by adding random offsets
    """
    neighbor = copy.deepcopy(rows)
    strategy = random.randint(0, 2)
    
    if strategy == 0:
        # Shift an element in a random row
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # Not the first element (must stay 0)
        shift = random.randint(-3, 3)
        if shift != 0:
            neighbor[row_idx][pos] += shift
            # Ensure the row remains strictly increasing
            if pos > 0 and neighbor[row_idx][pos] <= neighbor[row_idx][pos-1]:
                neighbor[row_idx][pos] = neighbor[row_idx][pos-1] + 1
            if pos < k and neighbor[row_idx][pos] >= neighbor[row_idx][pos+1]:
                # Need to adjust subsequent elements
                for i in range(pos+1, k+1):
                    if neighbor[row_idx][i] <= neighbor[row_idx][i-1]:
                        neighbor[row_idx][i] = neighbor[row_idx][i-1] + 1
    
    elif strategy == 1:
        # Swap two rows
        if n_rows >= 2:
            i, j = random.sample(range(n_rows), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    
    else:
        # Perturb a row by adding random small offsets to non-first elements
        row_idx = random.randint(0, n_rows - 1)
        for pos in range(1, k+1):
            if random.random() < 0.3:
                neighbor[row_idx][pos] += random.randint(-2, 2)
        # Fix to ensure strictly increasing
        for pos in range(1, k+1):
            if neighbor[row_idx][pos] <= neighbor[row_idx][pos-1]:
                neighbor[row_idx][pos] = neighbor[row_idx][pos-1] + 1
    
    return neighbor

def _repair_solution(rows, n_rows, k):
    """
    Try to repair an invalid solution by adjusting elements to resolve conflicts.
    Returns a valid solution or None if repair fails.
    """
    # This is a simple repair: try to fix one conflict at a time
    # For now, just return None if invalid (the main loop will try other neighbors)
    if _is_valid(rows, n_rows, k):
        return rows
    return None

def _compute_objective(rows):
    """
    Compute the objective function: maximize the number of valid constraints satisfied,
    and minimize the maximum entry.
    """
    used_diffs = set()
    conflicts = 0
    max_val = 0
    
    for row in rows:
        max_val = max(max_val, max(row))
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs:
                    conflicts += 1
                else:
                    used_diffs.add(d)
    
    return max_val, conflicts

# EVOLVE-BLOCK-END
