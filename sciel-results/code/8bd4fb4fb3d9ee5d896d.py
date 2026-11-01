import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial greedy construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used |= {x - a for a in row}
            rows.append(row)
        return rows, used

    best_rows, best_used = greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    current_rows = [row[:] for row in best_rows]
    current_used = set(best_used)
    current_max = best_max
    
    def get_max(rows):
        return max(max(row) for row in rows)
    
    def try_move(row_idx, elem_idx, new_val):
        row = current_rows[row_idx]
        if elem_idx > 0 and new_val <= row[elem_idx-1]:
            return False
        if elem_idx < len(row)-1 and new_val >= row[elem_idx+1]:
            return False
        
        old_val = row[elem_idx]
        if old_val == new_val:
            return False
            
        old_diffs = set()
        for j in range(len(row)):
            if j != elem_idx:
                d = abs(old_val - row[j])
                old_diffs.add(d)
                
        new_diffs = set()
        for j in range(len(row)):
            if j != elem_idx:
                d = abs(new_val - row[j])
                new_diffs.add(d)
                
        for d in new_diffs:
            if d in current_used and d not in old_diffs:
                return False
                
        current_used -= old_diffs
        current_used |= new_diffs
        row[elem_idx] = new_val
        return True

    steps = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_steps > max_no_improve:
            # Restart from best
            current_rows = [row[:] for row in best_rows]
            current_used = set(best_used)
            current_max = best_max
            no_improve_steps = 0
            # Perturb best solution
            for _ in range(5):
                r_idx = random.randint(0, n_rows - 1)
                e_idx = random.randint(1, k)
                old_val = current_rows[r_idx][e_idx]
                new_val = old_val + random.randint(-10, 10)
                if try_move(r_idx, e_idx, new_val):
                    pass # Keep perturbation
            
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, k)
        
        row = current_rows[r_idx]
        old_val = row[e_idx]
        
        candidates = set()
        
        if e_idx == k:
            # Try to reduce
            for delta in range(1, 20):
                candidates.add(old_val - delta)
            # Some random nearby
            for _ in range(10):
                candidates.add(old_val + random.randint(-5, 5))
        else:
            # Try nearby
            for delta in range(-10, 11):
                candidates.add(old_val + delta)
            # Try to move to a "gap" if possible?
            # Just random jump
            if random.random() < 0.1:
                candidates.add(old_val + random.randint(-50, 50))
        
        moved = False
        # Sort candidates by "desirability"
        # Prefer moves that reduce max or keep it same
        # For last element, smaller is better.
        # For others, maybe closer to current is better?
        
        # Just try them in random order to avoid bias, or sorted by distance
        for new_val in sorted(candidates, key=lambda x: random.random()):
            if new_val <= 0: continue
            if try_move(r_idx, e_idx, new_val):
                moved = True
                new_max = get_max(current_rows)
                if new_max < current_max:
                    current_max = new_max
                    best_rows = [row[:] for row in current_rows]
                    best_used = set(current_used)
                    no_improve_steps = 0
                elif new_max == current_max:
                    no_improve_steps += 1
                else:
                    # Should not happen as we only accept if valid, but validity doesn't guarantee max <= current_max
                    # Wait, try_move doesn't check max. It only checks diff collisions.
                    # A valid move can increase the max.
                    # We should reject moves that increase max.
                    # Revert
                    # We need to revert the move
                    # This is messy. Better to check max before committing?
                    # Or just revert.
                    
                    # Let's revert
                    current_used -= {abs(new_val - row[j]) for j in range(len(row)) if j != e_idx}
                    current_used |= {abs(old_val - row[j]) for j in range(len(row)) if j != e_idx}
                    row[e_idx] = old_val
                    moved = True # We tried a move, it was valid structurally but we rejected it
                    break # Try another candidate? Or break?
                    # Let's break and try another random element next step.
                
                break
        
        steps += 1

    return best_rows
