import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_rows = None
    best_max_val = float('inf')
    
    # First, run a deterministic greedy to get a baseline
    rows = greedy_construct(n_rows, k, used_diffs=None)
    if rows:
        max_val = max(max(row) for row in rows)
        best_rows = rows
        best_max_val = max_val
        
    # Now, run randomized restarts
    # We need to be careful with the time limit.
    # Each run is O(n * k * search_space).
    
    while time.time() - start_time < time_limit:
        rows = randomized_greedy_construct(n_rows, k, randomness=0.1)
        if rows:
            max_val = max(max(row) for row in rows)
            if max_val < best_max_val:
                best_max_val = max_val
                best_rows = rows
                # If we found a very good one, we might tighten the search?
                # Or just keep going.
                
    return best_rows

def greedy_construct(n_rows, k, used_diffs=None):
    if used_diffs is None:
        used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_diffs:
                row.append(x)
                used_diffs |= new
        rows.append(row)
    return rows

def randomized_greedy_construct(n_rows, k, randomness=0.1):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Find a few candidates
            candidates = []
            limit = 10 # Look ahead 10 steps
            found = 0
            while found < limit:
                x += 1
                new = {x - a for a in row}
                if not new & used_diffs:
                    candidates.append(x)
                    found += 1
            
            if not candidates:
                return None # Should not happen if problem is solvable, but safety
            
            # Pick one
            if random.random() < randomness and len(candidates) > 1:
                # Pick randomly from candidates
                x = random.choice(candidates)
            else:
                # Pick the smallest
                x = candidates[0]
                
            new = {x - a for a in row}
            row.append(x)
            used_diffs |= new
        rows.append(row)
    return rows
