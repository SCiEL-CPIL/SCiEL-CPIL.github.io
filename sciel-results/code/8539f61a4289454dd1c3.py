import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = k + 1
    diffs_per_row = n * (n - 1) // 2
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def evaluate_rows(rows):
        used = set()
        max_entry = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
                    if row[j] > max_entry:
                        max_entry = row[j]
        return max_entry, conflicts
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Initial solution
    best_rows = greedy_construct()
    best_score, _ = evaluate_rows(best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # Simulated Annealing
    T = 20.0
    cooling_rate = 0.9998
    min_T = 0.05
    
    current_rows = [row[:] for row in best_rows]
    current_score, current_conflicts = evaluate_rows(current_rows)
    
    while time.time() < deadline and T > min_T:
        # Choose a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Choose a split point (shift suffix)
        s_idx = random.randint(1, n - 1)
        
        # Choose delta
        # Bias towards negative to reduce max
        if random.random() < 0.6:
            delta = random.randint(-15, 5)
        else:
            delta = random.randint(-10, 10)
        
        if delta == 0:
            T *= cooling_rate
            continue
            
        # Check validity of shift for ordering
        if delta < 0 and row[s_idx] + delta <= row[s_idx - 1]:
            # Cannot shift left enough
            # Adjust delta to be the minimum valid one
            min_delta = row[s_idx - 1] - row[s_idx] + 1
            if min_delta > 0:
                delta = min_delta
            else:
                T *= cooling_rate
                continue
        
        # If delta > 0, it might increase max, which is bad, but SA might accept it to escape local minima
        # We'll just let the scoring handle it.
        
        # Apply move
        for j in range(s_idx, n):
            row[j] += delta
        
        new_score, new_conflicts = evaluate_rows(current_rows)
        
        # Score function: We want to minimize score, but conflicts are fatal.
        # If there are conflicts, the score is effectively infinite or very high.
        # Let's define effective_score = score + conflicts * 10000
        old_eff = current_score + current_conflicts * 10000
        new_eff = new_score + new_conflicts * 10000
        
        if new_eff < old_eff:
            current_score = new_score
            current_conflicts = new_conflicts
            if new_conflicts == 0 and new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in current_rows]
        else:
            delta_eff = new_eff - old_eff
            # Acceptance probability
            # Use exp(-delta/T)
            if random.random() < pow(2.718281828, -delta_eff / T):
                current_score = new_score
                current_conflicts = new_conflicts
            else:
                # Revert
                for j in range(s_idx, n):
                    row[j] -= delta
        
        T *= cooling_rate
    
    return best_rows
