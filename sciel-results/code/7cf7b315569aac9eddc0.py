import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Helper to generate a row given a set of used differences
    def generate_row(used_diffs, k, randomize=False, p_smallest=0.7, window=5):
        row = [0]
        current_row_diffs = set()
        
        for _ in range(k):
            candidates = []
            x = row[-1] + 1
            # We need to find x such that {x - a for a in row} are disjoint from (used_diffs U current_row_diffs)
            forbidden = used_diffs | current_row_diffs
            
            while x < 200:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
                
            row.append(next_x)
            # Update current_row_diffs
            current_row_diffs |= {next_x - a for a in row[:-1]}
            
        return row

    # Initial deterministic greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_row_diffs = set()
        for _ in range(k):
            x = row[-1] + 1
            forbidden = used | current_row_diffs
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    row.append(x)
                    current_row_diffs |= new_diffs
                    break
                x += 1
        rows.append(row)
        used |= current_row_diffs
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        # Pick a random row to regenerate
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx from used
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        # Regenerate
        new_row = generate_row(used, k, randomize=True, p_smallest=0.8, window=3)
        
        if new_row is None:
            # Rollback
            used |= old_diffs
            continue
        
        # Check if new_row is valid against used (it should be by construction)
        # Calculate new max
        new_max_val = max(new_row)
        # The global max is max(max of other rows, new_max_val)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0:
            # Accept
            rows[r_idx] = new_row
            used |= set(d for i in range(len(new_row)) for j in range(i+1, len(new_row)) for d in [new_row[j]-new_row[i]])
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Accept with probability
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            
            if random.random() < prob:
                rows[r_idx] = new_row
                used |= set(d for i in range(len(new_row)) for j in range(i+1, len(new_row)) for d in [new_row[j]-new_row[i]])
                current_max = new_global_max
            else:
                # Rollback
                used |= old_diffs
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
