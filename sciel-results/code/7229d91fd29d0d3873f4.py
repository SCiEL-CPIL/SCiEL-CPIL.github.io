"""
Hybrid approach:
1. Fast greedy column-wise construction to get an initial feasible set.
2. Local search / simulated annealing to refine and minimize the scope (largest entry).
"""
import time
import random
import math

def _deadline_guard(start, margin=5.0, limit=145.0):
    """Return True if we should stop (elapsed + margin >= limit)."""
    return (time.time() - start) + margin >= limit

def _score_of(rows):
    """Largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _is_valid(rows, k):
    """Check that all positive differences within rows are distinct across rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _build_greedy(n_rows, k, strategy="col_asc", rng=None):
    """
    Build a feasible set using greedy column-wise assignment.
    """
    if rng is None:
        rng = random.Random(0)
    
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()

    for col in range(k):
        if strategy == "col_asc":
            row_order = sorted(range(n_rows), key=lambda r: rows[r][-1])
        elif strategy == "col_desc":
            row_order = sorted(range(n_rows), key=lambda r: -rows[r][-1])
        else:
            row_order = list(range(n_rows))
            rng.shuffle(row_order)

        for r in row_order:
            x = rows[r][-1] + 1
            while True:
                conflict = False
                for i in range(len(rows[r])):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    break
                x += 1
            
            for i in range(len(rows[r])):
                used_diffs.add(x - rows[r][i])
            rows[r].append(x)
    
    return rows

def _local_search(rows, k, n_rows, start_time, limit=145.0, max_iters=10000):
    """
    Local search to reduce the maximum entry.
    We try to replace an element in a row with a smaller feasible value.
    """
    best_rows = [list(r) for r in rows]
    best_score = _score_of(best_rows)
    
    # Precompute used diffs for quick update
    used_diffs = set()
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    rng = random.Random(42)
    current_rows = [list(r) for r in best_rows]
    current_score = best_score
    
    for _ in range(max_iters):
        if _deadline_guard(start_time, 5.0, limit):
            break
        
        # Pick a random row and a random position (from 1 to k)
        r = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        
        old_val = current_rows[r][pos]
        
        # Remove this row's diffs from used_diffs temporarily
        row_diffs = set()
        for i in range(len(current_rows[r])):
            for j in range(i + 1, len(current_rows[r])):
                row_diffs.add(current_rows[r][j] - current_rows[r][i])
        for d in row_diffs:
            used_diffs.discard(d)
        
        # Find a better value
        found_improvement = False
        # Try to find a smaller value if pos is the last element (to reduce max)
        # Or try random valid values to escape local minima
        if pos == k and old_val > current_rows[r][pos-1] + 1:
            # Try to reduce the last element
            for new_val in range(current_rows[r][k-1] + 1, old_val):
                conflict = False
                for i in range(k): 
                    d = new_val - current_rows[r][i]
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    # Accept if it reduces the max
                    if new_val < current_rows[r][k]:
                        for i in range(k):
                            used_diffs.add(new_val - current_rows[r][i])
                        current_rows[r][k] = new_val
                        new_score = _score_of(current_rows)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_rows = [list(row) for row in current_rows]
                        found_improvement = True
                        break
        else:
            # Try a random valid value near the current one to explore
            # This helps escape local minima
            search_range = 10
            lower = current_rows[r][pos-1] + 1
            upper = min(old_val + search_range, best_score + 10)
            if upper > lower:
                # Try a few random candidates
                for _attempt in range(5):
                    new_val = rng.randint(lower, upper)
                    if new_val == old_val:
                        continue
                    conflict = False
                    for i in range(len(current_rows[r])):
                        if i == pos:
                            continue
                        d = abs(new_val - current_rows[r][i])
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        # Check if new_val maintains increasing order
                        if pos > 0 and new_val <= current_rows[r][pos-1]:
                            continue
                        if pos < k and new_val >= current_rows[r][pos+1]:
                            continue
                        
                        # Accept with some probability (simulated annealing style)
                        # Here we just accept if it doesn't increase the score too much or if it's better
                        for i in range(len(current_rows[r])):
                            if i != pos:
                                used_diffs.add(abs(new_val - current_rows[r][i]))
                        
                        current_rows[r][pos] = new_val
                        new_score = _score_of(current_rows)
                        
                        # Simple acceptance: accept if better or equal, or with small prob if worse
                        if new_score <= current_score or rng.random() < 0.1:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_rows = [list(row) for row in current_rows]
                            found_improvement = True
                            break
                        else:
                            # Reject, restore diffs
                            for i in range(len(current_rows[r])):
                                if i != pos:
                                    used_diffs.discard(abs(new_val - current_rows[r][i]))
                            current_rows[r][pos] = old_val

        # Re-add the row's diffs (which may have changed)
        # We need to be careful: if we accepted a new value, the diffs are already updated in used_diffs
        # If we rejected, we restored old_val, but the diffs from old_val are NOT in used_diffs (we removed them at start)
        # So we must add the current row's diffs back to used_diffs.
        
        # Calculate current diffs of the row
        current_row_diffs = set()
        for i in range(len(current_rows[r])):
            for j in range(i + 1, len(current_rows[r])):
                current_row_diffs.add(current_rows[r][j] - current_rows[r][i])
        
        # Add them to used_diffs
        for d in current_row_diffs:
            used_diffs.add(d)

    return best_rows, best_score

def construct(n_rows, k):
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple initial solutions using different greedy strategies
    initial_solutions = []
    
    # Strategy 1: Column-wise ascending
    rows1 = _build_greedy(n_rows, k, "col_asc")
    initial_solutions.append(rows1)
    
    # Strategy 2: Column-wise descending
    rows2 = _build_greedy(n_rows, k, "col_desc")
    initial_solutions.append(rows2)
    
    # Strategy 3: Random
    rng = random.Random(42)
    rows3 = _build_greedy(n_rows, k, "random", rng)
    initial_solutions.append(rows3)
    
    # Phase 2: Local search on each initial solution
    best_rows = None
    best_score = float('inf')
    
    for sol in initial_solutions:
        if _deadline_guard(start_time, 10.0, limit):
            break
        
        refined_rows, score = _local_search(sol, k, n_rows, start_time, limit)
        if score < best_score:
            best_score = score
            best_rows = refined_rows
    
    if best_rows is None:
        # Fallback to the first solution if local search didn't run
        best_rows = initial_solutions[0]
        best_score = _score_of(best_rows)
    
    # Final validation
    if not _is_valid(best_rows, k):
        # If the result is invalid, fall back to the first greedy solution
        # which is guaranteed valid by construction
        best_rows = initial_solutions[0]
    
    return best_rows
