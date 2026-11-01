"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0
    
    # Start with greedy construction as baseline
    best_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # Try randomized greedy with local search improvements
    for attempt in range(50):
        if time.time() > deadline:
            break
        rows = _randomized_greedy(n_rows, k)
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        
        # Try local search to reduce scope
        improved = _local_search(rows, n_rows, k, time.time() + 2.0)
        if improved is not None:
            imp_scope = max(max(r) for r in improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = [list(r) for r in improved]
    
    return best_rows

def _greedy_construct(n_rows, k):
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
    return rows

def _randomized_greedy(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Try candidates in random order near the minimum
            candidates = []
            for x_try in range(x + 1, x + 20):
                new = {x_try - a for a in row}
                if not new & used:
                    candidates.append(x_try)
            if not candidates:
                # Fallback to linear search
                x_try = x + 1
                while True:
                    new = {x_try - a for a in row}
                    if not new & used:
                        candidates.append(x_try)
                        break
                    x_try += 1
            
            # Pick randomly from early candidates for diversity
            if len(candidates) > 1:
                x = candidates[min(1, len(candidates) - 1)]
            else:
                x = candidates[0]
            
            new = {x - a for a in row}
            row.append(x)
            used |= new
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline):
    """Try to reduce the maximum entry by swapping elements."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_rows):
            for j in range(1, len(best[i])):
                # Try to reduce element j
                for new_val in range(best[i][j] - 1, 0, -1):
                    if new_val <= best[i][j-1]:
                        break
                    trial = list(best[i])
                    trial[j] = new_val
                    if _is_valid_single_row(trial, best, i):
                        best[i] = trial
                        new_scope = max(max(r) for r in best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
        if not improved:
            break
    
    return best

def _is_valid_single_row(row, all_rows, row_idx):
    """Check if a single row modification keeps all differences unique."""
    # Collect all differences except from this row
    used = set()
    for i, r in enumerate(all_rows):
        if i == row_idx:
            continue
        for a in range(len(r)):
            for b in range(a + 1, len(r)):
                used.add(r[b] - r[a])
    
    # Check new row's differences
    for a in range(len(row)):
        for b in range(a + 1, len(row)):
            diff = row[b] - row[a]
            if diff in used:
                return False
            used.add(diff)
    return True
# EVOLVE-BLOCK-END
