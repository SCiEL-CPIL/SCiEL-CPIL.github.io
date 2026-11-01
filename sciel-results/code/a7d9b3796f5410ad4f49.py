import time
import random
import math

def _greedy_init(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _calc_diffs(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used.add(row[j] - row[i])
    return used

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initialize
    try:
        rows = _greedy_init(n_rows, k)
    except Exception:
        # Fallback to simple sequential if greedy fails (unlikely)
        rows = [[i * (n_rows + 1) for i in range(k + 1)] for _ in range(n_rows)]
        
    used_diffs = _calc_diffs(rows)
    
    current_score = max(max(row) for row in rows)
    best_rows = [r[:] for r in rows]
    best_score = current_score
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    alpha = 0.9995
    
    # Helper to check validity and return diffs
    def get_diffs(row, idx, val):
        return {abs(row[j] - val) for j in range(len(row)) if j != idx}

    while time.time() < deadline:
        # SA Phase
        # Perform a batch of moves
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            r = random.randint(0, n_rows - 1)
            i = random.randint(1, k) # Index 0 is fixed at 0
            
            low = rows[r][i-1] + 1
            # Allow expansion slightly to escape local minima
            high = rows[r][i+1] - 1 if i < k else current_score + 10
            
            if high < low:
                continue
                
            found_valid = False
            new_v = -1
            
            # Try a few random candidates
            # Bias towards smaller values?
            # We can sample from [low, min(high, current_score)] more often
            # and [current_score, high] less often
            max_val = min(high, current_score)
            
            for _ in range(5):
                if random.random() < 0.9 and max_val >= low:
                    v = random.randint(low, max_val)
                else:
                    v = random.randint(low, high)
                    
                if v == rows[r][i]:
                    continue
                
                old_val = rows[r][i]
                old_diffs = get_diffs(rows[r], i, old_val)
                new_diffs = get_diffs(rows[r], i, v)
                
                # Check conflicts
                conflict = False
                for d in new_diffs:
                    if d in used_diffs:
                        if d not in old_diffs:
                            conflict = True
                            break
                            
                if not conflict:
                    found_valid = True
                    new_v = v
                    break
            
            if found_valid:
                # Apply move
                old_val = rows[r][i]
                
                # Update used_diffs
                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                
                rows[r][i] = new_v
                
                # Update score
                new_score = max(max(row) for row in rows)
                
                # Acceptance
                delta = new_score - current_score
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [row[:] for row in rows]
                else:
                    # Revert
                    rows[r][i] = old_val
                    used_diffs.difference_update(new_diffs)
                    used_diffs.update(old_diffs)
            
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart / Perturb
            # Make a few random valid moves
            for _ in range(10):
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k)
                low = rows[r][i-1] + 1
                high = rows[r][i+1] - 1 if i < k else current_score + 20
                if high >= low:
                    for _ in range(10):
                        v = random.randint(low, high)
                        if v == rows[r][i]: continue
                        old_val = rows[r][i]
                        old_diffs = get_diffs(rows[r], i, old_val)
                        new_diffs = get_diffs(rows[r], i, v)
                        conflict = False
                        for d in new_diffs:
                            if d in used_diffs and d not in old_diffs:
                                conflict = True
                                break
                        if not conflict:
                            used_diffs.difference_update(old_diffs)
                            used_diffs.update(new_diffs)
                            rows[r][i] = v
                            current_score = max(max(row) for row in rows)
                            break
            T = 5.0 # Reset temperature

    # Final Polish: Greedy improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(n_rows):
            for i in range(1, k+1):
                low = rows[r][i-1] + 1
                high = rows[r][i+1] - 1 if i < k else current_score
                # Try to reduce the value
                for v in range(rows[r][i] - 1, low - 1, -1):
                    if v < low: break
                    old_val = rows[r][i]
                    old_diffs = get_diffs(rows[r], i, old_val)
                    new_diffs = get_diffs(rows[r], i, v)
                    conflict = False
                    for d in new_diffs:
                        if d in used_diffs:
                            if d not in old_diffs:
                                conflict = True
                                break
                    if not conflict:
                        # Valid improvement
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        rows[r][i] = v
                        current_score = max(max(row) for row in rows)
                        if current_score < best_score:
                            best_score = current_score
                            best_rows = [row[:] for row in rows]
                        improved = True
                        break

    return best_rows
