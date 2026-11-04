import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Candidate pool size for the greedy selection. Larger = more lookahead,
    # but each candidate costs O(k) diff checks.
    c = 16

    seed = 0

    while time.time() < deadline:
        rows = _greedy_degree(n_rows, k, seed, c)
        
        # Try to polish
        if time.time() < deadline - 20:
            rows = _polish(rows, n_rows, k, time.time(), deadline - 5)
        
        # Calculate scope
        scope = 0
        for row in rows:
            if row:
                scope = max(scope, row[-1])
        
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
            # If we hit the lower bound, we can't do better
            if best_scope == lower_bound:
                break
        
        seed += 1

    # If no solution found (should not happen), return simple greedy
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
        
    return best_rows

def _greedy_degree(n_rows, k, seed, c):
    """Greedy with degree-based heuristic: prefer candidates that create fewer
    'conflicts' with already-used differences in the sense of minimizing future
    constraints. We score each candidate by how many used differences it is
    'close to' (i.e., how many used diffs are within a small range of the new
    diffs), preferring candidates that keep the difference set sparse."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            # Find valid candidates
            valid_xs = []
            scores = []
            last_x = row[-1]
            limit = 500 
            
            x = last_x + 1
            while len(valid_xs) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    # Score: count how many used diffs are "near" the new diffs
                    # A simple proxy: sum of min distances from each new diff to nearest used diff
                    score = 0
                    for d in new_diffs:
                        min_dist = min(abs(d - u) for u in used) if used else 0
                        score += min_dist
                    scores.append(score)
                x += 1
            
            if not valid_xs:
                # Expand limit if needed
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = 0
                        for d in new_diffs:
                            min_dist = min(abs(d - u) for u in used) if used else 0
                            score += min_dist
                        scores.append(score)
                    x += 1
            
            # Pick the candidate with the highest score (most "spread out")
            # with some randomness to explore
            if len(valid_xs) > 1 and rng.random() < 0.7:
                # Weighted selection favoring higher scores
                max_score = max(scores)
                min_score = min(scores)
                if max_score > min_score:
                    weights = [s - min_score + 1 for s in scores]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cumulative = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = i
                            break
                else:
                    chosen_idx = rng.randint(0, len(valid_xs) - 1)
            else:
                chosen_idx = rng.randint(0, len(valid_xs) - 1)
            
            chosen_x = valid_xs[chosen_idx]
            
            # Update row and used
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            
        rows.append(row)
        
    return rows

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

def _polish(rows, n_rows, k, start_time, deadline):
    """Local search: try to reduce the scope by moving elements to smaller positions."""
    rng = random.Random(int(start_time * 1000) % 100000)
    
    def _get_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def _valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    current_rows = [list(r) for r in rows]
    current_scope = max(max(r) for r in current_rows)
    
    for _ in range(50):  # max polish iterations
        if time.time() > deadline:
            break
        
        # Try to move a random element to a smaller position
        row_idx = rng.randint(0, n_rows - 1)
        pos_idx = rng.randint(1, k)  # positions 1..k (position 0 is always 0)
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Try moving it to a smaller value
        # The new value must be > current_rows[row_idx][pos_idx-1] and < current_rows[row_idx][pos_idx+1] if pos_idx < k
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val - 1
        
        if upper < lower:
            continue
        
        # Try a few candidate values
        candidates = list(range(upper, lower - 1, -1))  # try smaller values first
        rng.shuffle(candidates)
        
        for new_val in candidates[:10]:
            if new_val >= old_val:
                continue
            
            # Temporarily update
            old = current_rows[row_idx][pos_idx]
            current_rows[row_idx][pos_idx] = new_val
            
            if _valid(current_rows):
                new_scope = max(max(r) for r in current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    break
                else:
                    # Revert if no improvement
                    current_rows[row_idx][pos_idx] = old
            else:
                current_rows[row_idx][pos_idx] = old
    
    return current_rows
