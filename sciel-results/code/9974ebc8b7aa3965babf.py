import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 145.0  # 5s safety margin for 150s limit
    
    # The number of differences in one row is k * (k+1) / 2
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    best_rows = None
    best_M = float('inf')
    
    # Greedy construction for initial upper bound
    initial_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(initial_rows)
    best_rows = initial_rows
    
    # Try to improve by searching for solutions with smaller M
    M_lower = max(total_diffs_needed, k * (k + 1) // 2 + k, 20)
    
    # For (7,5), M_lower = 105, best_M might be around 112-120
    # Try M values from M_lower up to best_M - 1
    for M in range(M_lower, best_M):
        if time.time() > deadline - 10:
            break
        
        # Try to find a packing with max entry <= M using simulated annealing
        rows = _solve_with_sa(n_rows, k, M, deadline - 5)
        
        if rows is not None and _is_valid(rows, k):
            best_rows = rows
            best_M = M
            break
    
    return best_rows

def _solve_with_sa(n_rows, k, M, time_limit):
    """
    Use Simulated Annealing to find a valid packing with max entry <= M.
    """
    # Represent each row as a list of k+1 integers starting at 0
    # Objective: minimize conflicts (duplicates in differences) and maximize validity
    
    # Initialize with greedy rows capped at M
    current_rows = _greedy_construct_capped(n_rows, k, M)
    
    # If already valid, return
    if _is_valid(current_rows, k):
        return current_rows
    
    # Simulated Annealing
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.001
    max_iterations = 100000
    
    best_rows = None
    best_score = float('inf')
    
    current_score = _score_rows(current_rows, k)
    
    iterations = 0
    start_time = time.time()
    
    while T > min_T and iterations < max_iterations:
        if time.time() > start_time + time_limit:
            break
        
        # Generate neighbor
        neighbor_rows = _generate_neighbor(current_rows, k, M)
        neighbor_score = _score_rows(neighbor_rows, k)
        
        # Accept if better or with probability exp(-delta/T)
        delta = neighbor_score - current_score
        if delta < 0 or random.random() < _exp_neg(delta / T):
            current_rows = neighbor_rows
            current_score = neighbor_score
            
            if current_score < best_score:
                best_score = current_score
                best_rows = [row[:] for row in current_rows]
                
                # If valid, we can stop early
                if best_score == 0:
                    return best_rows
        
        # Cool down
        T *= cooling_rate
        iterations += 1
    
    if best_score == 0:
        return best_rows
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > start_time + time_limit:
            break
        
        current_rows = _greedy_construct_capped(n_rows, k, M)
        random.shuffle(current_rows)
        # Slightly perturb
        for i in range(n_rows):
            if len(current_rows[i]) > 2:
                # Swap two random elements (keeping 0 and sorted)
                idx1 = random.randint(1, len(current_rows[i]) - 2)
                idx2 = random.randint(idx1 + 1, len(current_rows[i]) - 1)
                current_rows[i][idx1], current_rows[i][idx2] = current_rows[i][idx2], current_rows[i][idx1]
                current_rows[i].sort()
        
        current_score = _score_rows(current_rows, k)
        best_score_local = current_score
        best_rows_local = [row[:] for row in current_rows]
        
        if current_score == 0:
            return best_rows_local
        
        T = 5.0
        for _ in range(20000):
            if time.time() > start_time + time_limit:
                break
            
            neighbor_rows = _generate_neighbor(current_rows, k, M)
            neighbor_score = _score_rows(neighbor_rows, k)
            
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < _exp_neg(delta / T):
                current_rows = neighbor_rows
                current_score = neighbor_score
                
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_rows_local = [row[:] for row in current_rows]
                    if best_score_local == 0:
                        return best_rows_local
            
            T *= 0.998
        
        if best_score_local < best_score:
            best_score = best_score_local
            best_rows = best_rows_local
    
    if best_score == 0:
        return best_rows
    
    return None

def _score_rows(rows, k):
    """
    Score the rows: number of conflicting differences (duplicates across all rows).
    0 means valid.
    """
    used = {}
    conflicts = 0
    for row in rows:
        if len(row) != k + 1:
            return float('inf')
        if row[0] != 0:
            return float('inf')
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    conflicts += 1
                else:
                    used[d] = True
    return conflicts

def _generate_neighbor(rows, k, M):
    """
    Generate a neighboring solution by modifying one row slightly.
    """
    new_rows = [row[:] for row in rows]
    idx = random.randint(0, len(new_rows) - 1)
    row = new_rows[idx]
    
    # Choose a strategy
    strategy = random.random()
    
    if strategy < 0.5:
        # Adjust one element
        if len(row) > 2:
            pos = random.randint(1, len(row) - 1)
            delta = random.choice([-2, -1, 1, 2])
            new_val = row[pos] + delta
            if 0 < new_val <= M and new_val not in row:
                row[pos] = new_val
                row.sort()
    elif strategy < 0.8:
        # Replace one element
        if len(row) > 2:
            pos = random.randint(1, len(row) - 1)
            new_val = random.randint(1, M)
            if new_val not in row:
                row[pos] = new_val
                row.sort()
    else:
        # Regenerate a random part of the row
        if len(row) > 2:
            # Keep first and last, regenerate middle
            if len(row) >= 4:
                mid_start = random.randint(1, len(row) - 3)
                mid_end = random.randint(mid_start + 1, len(row) - 1)
                remaining = set(range(1, M + 1)) - set(row)
                needed = mid_end - mid_start
                if len(remaining) >= needed:
                    new_mids = random.sample(sorted(remaining), needed)
                    new_row = row[:mid_start] + new_mids + row[mid_end:]
                    new_row.sort()
                    # Ensure strictly increasing and within bounds
                    if all(0 < new_row[i] < new_row[i+1] for i in range(len(new_row)-1)) and new_row[-1] <= M:
                        new_rows[idx] = new_row
    
    return new_rows

def _exp_neg(x):
    """Safely compute exp(-x) for x >= 0."""
    if x > 50:
        return 0.0
    return __import__('math').exp(-x)

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _greedy_construct_capped(n_rows, k, M):
    """
    Greedy construction with max entry capped at M.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1 and x < M:
            x += 1
            if x > M:
                break
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        if len(row) < k + 1:
            # Fill with random values
            while len(row) < k + 1:
                candidate = random.randint(1, M)
                if candidate not in row:
                    new = {candidate - a for a in row}
                    if not new & used and len(new) == len(row):
                        row.append(candidate)
                        used |= new
                        row.sort()
            if len(row) < k + 1:
                # Last resort: fill with largest possible
                row = [0] + list(range(1, k + 1))
                row.sort()
                # Remove conflicts
                for d in {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}:
                    used.discard(d)
                used.update({row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))})
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
