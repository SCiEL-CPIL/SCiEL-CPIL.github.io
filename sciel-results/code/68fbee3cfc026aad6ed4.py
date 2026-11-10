import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def compute_diff_set(rows):
        """Return set of all positive differences in rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def randomized_greedy():
        """Randomized greedy with weighted candidate selection."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                candidates = []
                x = current_max + 1
                max_x = current_max + max(30, k * 3)
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    return None
                
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    weights = [1.0 / (c - current_max) for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break
                
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            rows.append(row)
        
        return rows
    
    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Randomized greedy restarts
    num_restarts = 0
    while time.time() < deadline - 15:  # Leave 15s for local search
        rows = randomized_greedy()
        if rows is not None and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        num_restarts += 1
    
    # Phase 3: Local search with simulated annealing
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.997
        
        def try_modify(rows):
            """Try a random modification and return modified rows if valid, else None."""
            new_rows = [r[:] for r in rows]
            mod_type = random.choice(['shift', 'replace', 'swap', 'insert_remove', 'row_swap'])
            
            row_idx = random.randint(0, len(new_rows) - 1)
            row = new_rows[row_idx]
            
            if mod_type == 'shift':
                if len(row) > 1:
                    idx = random.randint(1, len(row) - 1)
                    delta = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                    new_val = row[idx] + delta
                    if new_val > row[idx - 1] and (idx == len(row) - 1 or new_val < row[idx + 1]):
                        row[idx] = new_val
                        if is_valid(new_rows):
                            return new_rows
                        row[idx] = new_val - delta  # revert
                    return None
            
            elif mod_type == 'replace':
                if len(row) > 1:
                    idx = random.randint(1, len(row) - 1)
                    for new_val in range(max(row[idx - 1] + 1, row[idx] - 15), min(row[idx] + 15, (row[idx + 1] - 1 if idx < len(row) - 1 else row[idx] + 15))):
                        row[idx] = new_val
                        if is_valid(new_rows):
                            return new_rows
                    row[idx] = row[idx]  # no change
                    return None
            
            elif mod_type == 'swap':
                if len(row) > 2:
                    i, j = sorted(random.sample(range(1, len(row)), 2))
                    row[i], row[j] = row[j], row[i]
                    if all(row[t] < row[t + 1] for t in range(len(row) - 1)):
                        if is_valid(new_rows):
                            return new_rows
                    row[i], row[j] = row[j], row[i]
                    return None
            
            elif mod_type == 'insert_remove':
                if len(row) > 2:
                    remove_idx = random.randint(1, len(row) - 2)
                    saved = row.pop(remove_idx)
                    for add_val in range(row[-1] + 1, row[-1] + 20):
                        row.append(add_val)
                        if is_valid(new_rows):
                            return new_rows
                        row.pop()
                    row.insert(remove_idx, saved)
                    return None
            
            elif mod_type == 'row_swap':
                if len(new_rows) > 1:
                    other_idx = random.choice([i for i in range(len(new_rows)) if i != row_idx])
                    # Swap the non-zero parts of two rows
                    new_rows[row_idx], new_rows[other_idx] = new_rows[other_idx][:], new_rows[row_idx][:]
                    if is_valid(new_rows):
                        return new_rows
                    new_rows[row_idx], new_rows[other_idx] = new_rows[other_idx][:], new_rows[row_idx][:]
                    return None
            
            return None
        
        steps = 0
        while time.time() < deadline - 2:
            modified = try_modify(current_rows)
            if modified is not None:
                new_scope = get_scope(modified)
                if new_scope < current_scope:
                    current_rows = modified
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                elif random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_rows = modified
                    current_scope = new_scope
            
            temperature *= cooling_rate
            steps += 1
            
            # Occasionally restart from a fresh greedy solution
            if steps % 100 == 0 and random.random() < 0.15:
                fresh = greedy_construct()
                if is_valid(fresh):
                    fresh_scope = get_scope(fresh)
                    if fresh_scope < current_scope:
                        current_rows = fresh
                        current_scope = fresh_scope
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]
    
    return best_rows
