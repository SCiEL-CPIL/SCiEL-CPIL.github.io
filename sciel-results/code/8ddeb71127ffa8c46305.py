"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0
    
    num_entries = k + 1
    diffs_per_row = num_entries * (num_entries - 1) // 2
    
    def get_diffs(row):
        return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
    
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_entries:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_sol = greedy()
    best_max = max(max(r) for r in best_sol)
    
    def local_search(initial_rows, max_time=10.0):
        nonlocal best_sol, best_max
        current = [list(r) for r in initial_rows]
        end_time = time.time() + max_time
        
        while time.time() < end_time:
            # Find row with max entry
            max_idx = max(range(n_rows), key=lambda i: max(current[i]))
            row = current[max_idx]
            improved = False
            
            # Try to reduce entries from largest to smallest
            for pos in range(num_entries - 1, 0, -1):
                old = row[pos]
                # Try smaller values
                for new_val in range(row[pos-1] + 1, old):
                    test_row = list(row)
                    test_row[pos] = new_val
                    if any(test_row[i] >= test_row[i+1] for i in range(num_entries-1)):
                        continue
                    td = get_diffs(test_row)
                    if len(td) != diffs_per_row:
                        continue
                    # Check against others
                    ok = True
                    for i, r in enumerate(current):
                        if i == max_idx:
                            continue
                        if td & get_diffs(r):
                            ok = False
                            break
                    if not ok:
                        continue
                    current[max_idx] = test_row
                    nm = max(max(r) for r in current)
                    if nm < best_max:
                        best_max = nm
                        best_sol = [list(r) for r in current]
                    improved = True
                    break
                if improved:
                    break
            
            if not improved:
                # Random perturbation
                ridx = random.randint(0, n_rows - 1)
                pidx = random.randint(1, num_entries - 1)
                row = current[ridx]
                for _ in range(20):
                    lo = row[pidx-1] + 1
                    hi = row[pidx] + 10
                    if hi <= lo:
                        continue
                    nv = random.randint(lo, hi)
                    test_row = list(row)
                    test_row[pidx] = nv
                    if any(test_row[i] >= test_row[i+1] for i in range(num_entries-1)):
                        continue
                    td = get_diffs(test_row)
                    if len(td) != diffs_per_row:
                        continue
                    ok = True
                    for i, r in enumerate(current):
                        if i == ridx:
                            continue
                        if td & get_diffs(r):
                            ok = False
                            break
                    if not ok:
                        continue
                    current[ridx] = test_row
                    nm = max(max(r) for r in current)
                    if nm < best_max:
                        best_max = nm
                        best_sol = [list(r) for r in current]
                    break
        
        return current
    
    # Run local search with multiple restarts
    while time.time() < time_limit - 5:
        local_search(best_sol, max_time=5.0)
    
    return best_sol
# EVOLVE-BLOCK-END
