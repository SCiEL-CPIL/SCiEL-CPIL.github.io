import time
import random
import bisect
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    seed = 0

    # Phase 1: Backtracking with degree heuristic for the first few rows
    # Phase 2: Greedy + local search for the rest
    while time.time() < deadline - 30:
        rows = _backtracking_construct(n_rows, k, seed)
        
        if rows is None:
            seed += 1
            continue
        
        # Try to polish with simulated annealing
        if time.time() < deadline - 15:
            rows = _simulated_annealing(rows, n_rows, k, time.time(), deadline - 5)
        
        # Calculate scope
        scope = max(max(r) for r in rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
            # If we hit the lower bound, we can't do better
            if best_scope == lower_bound:
                break
        
        seed += 1

    # Final polish pass if time permits
    if best_rows is not None and time.time() < deadline - 5:
        best_rows = _polish_max(best_rows, n_rows, k, time.time(), deadline - 1)
        best_scope = max(max(r) for r in best_rows)

    # If no solution found (should not happen), return simple greedy
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
        
    return best_rows

def _scope(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _simple_greedy(n_rows, k):
    """Simple greedy fallback."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)
    
    return rows

def _backtracking_construct(n_rows, k, seed):
    """
    Backtracking search that builds rows one element at a time.
    Uses a degree heuristic: at each step, prefer candidates that leave
    the most remaining valid difference slots for subsequent rows.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    def backtrack(row_idx, pos_idx, current_row, depth=0):
        if depth > 100000:  # Safety limit
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos_idx == 0:
            current_row = [0]
            return backtrack(row_idx, 1, current_row, depth + 1)
        
        if pos_idx == k + 1:
            # Row complete
            rows.append(current_row)
            return backtrack(row_idx + 1, 0, [], depth + 1)
        
        last_x = current_row[-1]
        # Try candidates in random order with bias toward smaller values
        # but weighted by degree heuristic
        
        candidates = []
        x = last_x + 1
        max_search = last_x + 50  # Limit search space for backtracking
        
        while x <= max_search and len(candidates) < 20:
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & used):
                # Degree heuristic: count how many future diffs are still available
                # Approximate by checking how many small diffs are still free
                score = _degree_score(new_diffs, used, x)
                candidates.append((x, score))
            x += 1
        
        if not candidates:
            return False
        
        # Sort by score (higher is better) but add randomness
        rng.shuffle(candidates)
        candidates.sort(key=lambda t: -t[1])
        
        # Try top candidates with some randomness
        for x, score in candidates[:5]:
            new_diffs = {x - a for a in current_row}
            used.update(new_diffs)
            current_row.append(x)
            
            if backtrack(row_idx, pos_idx + 1, current_row, depth + 1):
                return True
            
            # Undo
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    if backtrack(0, 0, []):
        return rows
    return None

def _degree_score(new_diffs, used, x):
    """
    Heuristic score: prefer candidates that use smaller differences
    (leaves more room for future rows) and don't fragment the diff space too much.
    """
    # Prefer smaller x (tighter packing)
    base_score = -x * 0.1
    
    # Penalize using large gaps in the diff space
    gap_penalty = 0
    for d in new_diffs:
        # Check if this diff creates a large gap
        # Approximate: penalize if d is much larger than typical
        gap_penalty += max(0, d - 20) * 0.05
    
    return base_score - gap_penalty

def _simulated_annealing(rows, n_rows, k, start_time, deadline):
    """Simulated annealing to reduce scope."""
    rng = random.Random(int(start_time * 10000) % 1000000)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    # Precompute used diffs for fast validity check
    used = set()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    t0 = 5.0
    t_end = 0.1
    cooling = 0.97
    
    t = t0
    no_improve = 0
    
    while time.time() < deadline and t > t_end:
        # Pick a random element to modify
        row_idx = rng.randint(0, n_rows - 1)
        pos_idx = rng.randint(1, k)
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Determine valid range for new value
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val - 1
        
        if upper <= lower:
            t *= cooling
            continue
        
        # Try to move to a smaller value (bias toward reduction)
        if rng.random() < 0.7:
            search_range = min(20, old_val - lower)
            if search_range > 0:
                new_val = rng.randint(lower, min(upper, old_val - 1))
            else:
                t *= cooling
                continue
        else:
            new_val = rng.randint(lower, upper)
        
        if new_val == old_val:
            t *= cooling
            continue
        
        # Compute new diffs for this row
        old_row = current_rows[row_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        new_row = list(old_row)
        new_row[pos_idx] = new_val
        
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Check validity: new_diffs minus old_diffs should not intersect with (used minus old_diffs)
        added = new_diffs - old_diffs
        removed = old_diffs - new_diffs
        
        if added & (used - old_diffs):
            t *= cooling
            continue
        
        # Apply move
        current_rows[row_idx] = new_row
        used.difference_update(removed)
        used.update(added)
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta < 0 or rng.random() < math.exp(-delta / t):
            current_scope = new_scope
            if delta < 0:
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Undo
            current_rows[row_idx] = old_row
            used.difference_update(added)
            used.update(removed)
            no_improve += 1
        
        t *= cooling
        
        if no_improve > 100:
            break
    
    return current_rows

def _polish_max(rows, n_rows, k, start_time, deadline):
    """Targeted polish: try to reduce the maximum element."""
    rng = random.Random(int(start_time * 10000) % 1000000)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    # Precompute used diffs
    used = set()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    # Find the row with the maximum element
    max_row_idx = 0
    for i, row in enumerate(current_rows):
        if row[-1] > current_rows[max_row_idx][-1]:
            max_row_idx = i
    
    # Try to reduce the last element of the max row
    max_pos = k
    old_val = current_rows[max_row_idx][max_pos]
    
    lower = current_rows[max_row_idx][max_pos - 1] + 1
    upper = old_val - 1
    
    if upper >= lower:
        for new_val in range(upper, lower - 1, -1):
            if time.time() > deadline:
                break
            
            old_row = current_rows[max_row_idx]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            new_row = list(old_row)
            new_row[max_pos] = new_val
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            added = new_diffs - old_diffs
            removed = old_diffs - new_diffs
            
            if not (added & (used - old_diffs)):
                # Valid move
                current_rows[max_row_idx] = new_row
                used.difference_update(removed)
                used.update(added)
                current_scope = _scope(current_rows)
                break
    
    # Also try reducing other elements in the max row
    for pos_idx in range(k - 1, 0, -1):
        if time.time() > deadline:
            break
        
        old_v = current_rows[max_row_idx][pos_idx]
        lower_v = current_rows[max_row_idx][pos_idx - 1] + 1
        upper_v = current_rows[max_row_idx][pos_idx + 1] - 1 if pos_idx < k else old_v - 1
        
        if upper_v < lower_v:
            continue
        
        for new_v in range(upper_v, lower_v - 1, -1):
            if new_v >= old_v:
                continue
            
            old_row = current_rows[max_row_idx]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            new_row = list(old_row)
            new_row[pos_idx] = new_v
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            added = new_diffs - old_diffs
            removed = old_diffs - new_diffs
            
            if not (added & (used - old_diffs)):
                current_rows[max_row_idx] = new_row
                used.difference_update(removed)
                used.update(added)
                current_scope = _scope(current_rows)
                break
    
    return current_rows
