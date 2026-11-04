"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Greedy base solution
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
    best_rows = rows
    
    # Try to improve with randomized restarts
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 200:
        attempts += 1
        
        # Randomized greedy with different strategies
        used, rows = set(), []
        success = True
        for row_idx in range(n_rows):
            row, x = [0], 0
            row_success = True
            while len(row) < k + 1:
                # Try candidates in a shuffled order near the greedy choice
                base_x = x + 1
                candidates = list(range(base_x, base_x + 50))
                random.shuffle(candidates)
                
                found = False
                for cand in candidates:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                
                if not found:
                    # Try larger values
                    for cand in range(x + 50, x + 200):
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            found = True
                            break
                
                if not found:
                    row_success = False
                    break
            
            if not row_success:
                success = False
                break
            
            rows.append(row)
        
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope <= 112:  # counting bound
                    break
    
    return best_rows
# EVOLVE-BLOCK-END
