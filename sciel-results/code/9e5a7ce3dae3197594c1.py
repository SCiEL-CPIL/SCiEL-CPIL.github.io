import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin against 120s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
    
    def greedy_solution(seed=0):
        """Generate a solution using randomized greedy."""
        random.seed(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                limit = x + 500
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 20:
                            break
                    x += 1
                
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Pick randomly from first few candidates
                pick_from = min(len(candidates), 5)
                chosen = random.choice(candidates[:pick_from])
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows
    
    def local_search(rows, max_time=5.0):
        """Try to improve the solution by reducing the maximum entry."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        while improved and time.time() - start < max_time:
            improved = False
            
            # Find the element with the largest value
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_i == 0:
                break
            
            row = rows[max_r]
            current_val = row[max_i]
            
            # Determine valid range
            min_val = row[max_i - 1] + 1
            max_allowed = row[max_i + 1] - 1 if max_i < len(row) - 1 else current_val - 1
            
            if max_allowed < min_val:
                break
            
            # Remove current differences from used
            old_diffs = set()
            for a in row:
                if a != current_val:
                    d = abs(a - current_val)
                    old_diffs.add(d)
            used -= old_diffs
            
            new_val_found = None
            # Try smallest valid replacement first
            for v in range(min_val, current_val):
                valid = True
                new_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - v)
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    new_val_found = v
                    break
            
            if new_val_found is not None:
                row[max_i] = new_val_found
                used |= new_diffs
                improved = True
                score = get_score(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
            else:
                used |= old_diffs
                break
        
        return best_rows
    
    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds
    for seed in range(200):
        if time.time() - start_time > time_limit - 10:
            break
        
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        # Apply local search
        rows = local_search(rows, max_time=2.0)
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            # Early exit if we hit the theoretical bound
            if score <= 112:
                break
    
    if best_rows is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_rows
