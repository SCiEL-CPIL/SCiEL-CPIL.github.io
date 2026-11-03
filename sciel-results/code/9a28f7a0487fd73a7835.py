import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0
    
    def compute_max_entry(rows):
        return max(max(row) for row in rows)
    
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def is_valid(rows):
        used = set()
        for row in rows:
            # Check strictly increasing and starts at 0
            if not row or row[0] != 0:
                return False
            for i in range(len(row)-1):
                if row[i+1] <= row[i]:
                    return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    def local_search(rows, max_iters=10000):
        current_rows = [row[:] for row in rows]
        current_max = compute_max_entry(current_rows)
        best_rows = [row[:] for row in current_rows]
        best_max = current_max
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random row and a random position
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # positions 1..k (0 is fixed)
            
            old_val = current_rows[r][pos]
            
            # Try to replace with a smaller value
            # Candidate range: [current_rows[r][pos-1] + 1, old_val - 1]
            low = current_rows[r][pos-1] + 1
            high = old_val - 1
            
            if low > high:
                continue
                
            # Try a few random candidates in this range
            tried = set()
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                new_val = random.randint(low, high)
                if new_val in tried:
                    continue
                tried.add(new_val)
                
                # Check if new_val is valid in this row
                valid = True
                # Check increasing
                if new_val <= current_rows[r][pos-1] or (pos < k and new_val >= current_rows[r][pos+1]):
                    valid = False
                
                if not valid:
                    continue
                
                # Check differences with other elements in the same row
                row_diffs_new = set()
                for i in range(len(current_rows[r])):
                    if i != pos:
                        d = abs(current_rows[r][i] - new_val)
                        row_diffs_new.add(d)
                
                # Check if any of these new diffs conflict with existing used diffs
                # We need to recompute used diffs excluding the old_val's contributions
                # This is expensive. Instead, let's just do a full validation on a copy.
                # For speed, we can do a quick check first.
                
                # Quick check: remove old diffs, add new diffs, see if any duplicate
                # This is still O(k^2) per move.
                
                # Let's just try the move and validate the whole board
                # But that's O(n_rows * k^2).
                # Let's optimize:
                # 1. Compute diffs for row r with old_val
                old_diffs = set()
                for i in range(len(current_rows[r])):
                    if i != pos:
                        old_diffs.add(abs(current_rows[r][i] - old_val))
                
                # 2. Compute diffs for row r with new_val
                new_diffs = set()
                for i in range(len(current_rows[r])):
                    if i != pos:
                        new_diffs.add(abs(current_rows[r][i] - new_val))
                
                # 3. Check if new_diffs intersect with (all_used - old_diffs)
                # all_used is expensive to maintain. Let's just do a full validation if quick check passes.
                
                # Quick check: if new_diffs has any duplicates within itself -> invalid
                if len(new_diffs) != len([d for d in new_diffs]):
                    continue
                
                # If new_diffs intersects with old_diffs, that's fine (they replace each other)
                # But if new_diffs intersects with diffs from OTHER rows, that's bad.
                
                # We don't have a global used set readily available without recomputing.
                # Let's just do a full validation. It's simpler and likely fast enough for small n_rows, k.
                
                temp_rows = [row[:] for row in current_rows]
                temp_rows[r][pos] = new_val
                if is_valid(temp_rows):
                    # Accept if it reduces max or is equal (for diversity)
                    new_max = compute_max_entry(temp_rows)
                    if new_max <= best_max:
                        current_rows = temp_rows
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [row[:] for row in current_rows]
                    break
        
        return best_rows, best_max
    
    # Initial greedy solution
    rows = greedy_construct()
    best_rows = [row[:] for row in rows]
    best_max = compute_max_entry(best_rows)
    
    # Run local search multiple times with different random seeds
    random.seed(42)
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb the current best solution to escape local minima
        perturbed = [row[:] for row in best_rows]
        # Randomly swap some values
        for _ in range(10):
            r1 = random.randint(0, n_rows - 1)
            r2 = random.randint(0, n_rows - 1)
            if r1 == r2:
                continue
            p1 = random.randint(1, k)
            p2 = random.randint(1, k)
            
            # Check if swap is valid
            val1 = perturbed[r1][p1]
            val2 = perturbed[r2][p2]
            
            # Simple check: after swap, rows must be increasing
            # This is complex, so let's just skip complex perturbations
            # and rely on the local search itself to explore.
            pass
        
        # Run local search
        improved_rows, improved_max = local_search(perturbed, max_iters=20000)
        
        if improved_max < best_max:
            best_max = improved_max
            best_rows = [row[:] for row in improved_rows]
    
    return best_rows
