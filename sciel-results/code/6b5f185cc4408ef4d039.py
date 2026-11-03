import time
import random

def construct(n_rows, k):
    start_time = time.time()
    # Leave 5 seconds for safety and cleanup
    deadline = start_time + 115.0
    
    # 1. Baseline: Deterministic Greedy (Seed behavior)
    best_rows = _greedy_construct(n_rows, k, randomize=False)
    if best_rows is None:
        return [] # Should not happen for valid inputs
    
    best_scope = max(max(row) for row in best_rows)
    
    # If k is small, we might find optimal quickly.
    # If n_rows is small, we can do more restarts.
    
    # 2. Randomized Search
    # We try to find a solution with a smaller scope.
    # We limit the number of restarts based on time.
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Randomize the search
        rows = _greedy_construct(n_rows, k, randomize=True)
        if rows is not None:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # If we find a very good solution, we might want to continue to see if we can do better
                # But if we hit the theoretical lower bound, we can stop?
                # Lower bound is roughly k^2 / 2 ? No, counting bound is 112 for (4,7).
                # We don't know the exact lower bound easily.
                
    return best_rows

def _greedy_construct(n_rows, k, randomize=False):
    used = set()
    rows = []
    
    # Number of candidates to consider
    # If randomize, we look at a window of valid candidates
    N_CANDIDATES = 30 if randomize else 1
    
    for _ in range(n_rows):
        row = [0]
        success = True
        for _ in range(k):
            last = row[-1]
            candidates = []
            x = last + 1
            
            # Find N_CANDIDATES valid x
            while len(candidates) < N_CANDIDATES:
                x += 1
                # Check if x creates unique differences with previous elements
                # Diff: x - a for a in row
                # Must not be in used
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            
            if not candidates:
                # This shouldn't happen if we don't bound x, 
                # but if we did, we would fail.
                # Since we don't bound x, we always find candidates.
                # But if N_CANDIDATES is 0? No, it's 1 or 30.
                # So this block is effectively dead code unless logic changes.
                success = False
                break
            
            if randomize:
                # Weighted choice: prefer smaller x, but allow larger
                # Simple uniform choice from the window
                x = random.choice(candidates)
            else:
                x = candidates[0]
            
            row.append(x)
            # Update used set
            for a in row[:-1]:
                used.add(x - a)
        
        if not success:
            return None
            
        rows.append(row)
        
    return rows
