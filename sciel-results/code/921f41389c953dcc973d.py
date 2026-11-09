"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Greedy construction as baseline
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # Try randomized greedy with different orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(rows)
        used2 = set()
        new_rows = []
        valid = True
        
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if new & used2:
                    continue
                # Check if this fits within current best scope
                if x > best_scope:
                    valid = False
                    break
                row.append(x)
                used2 |= new
            if not valid:
                break
            new_rows.append(row)
        
        if valid and len(new_rows) == n_rows:
            scope = max(max(r) for r in new_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in new_rows]
    
    return best_rows
# EVOLVE-BLOCK-END
