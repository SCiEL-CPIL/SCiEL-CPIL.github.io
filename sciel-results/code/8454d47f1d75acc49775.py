import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # Leave some margin for overhead
    
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    def compute_scope(rows):
        return max(max(r) for r in rows)
    
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows, used
    
    def local_search(rows, used_diffs, max_iterations=5000):
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        no_improve_count = 0
        
        for it in range(max_iterations):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick a row with the maximum value
            max_val = max(max(r) for r in rows)
            candidate_rows = [i for i, r in enumerate(rows) if max(r) == max_val]
            r_idx = random.choice(candidate_rows)
            row = rows[r_idx]
            
            # Try to reduce the last element first, then others
            # Focus on positions that can reduce scope
            target_positions = []
            if row[-1] == max_val:
                target_positions.append(k)  # Last position
            for j in range(1, k):
                if row[j] > row[j-1] + 1:
                    target_positions.append(j)
            
            if not target_positions:
                target_positions = list(range(1, k+1))
            
            j = random.choice(target_positions)
            old_val = row[j]
            
            lower = row[j-1] + 1
            upper = row[j+1] - 1 if j < k else current_scope
            
            if lower > upper:
                continue
            
            # Try to find a smaller valid value
            found = False
            for x in range(lower, old_val):
                # Check if x is valid
                old_diffs = set()
                for i in range(j):
                    old_diffs.add(old_val - row[i])
                for i in range(j+1, k+1):
                    old_diffs.add(row[i] - old_val)
                
                new_diffs = set()
                for i in range(j):
                    new_diffs.add(x - row[i])
                for i in range(j+1, k+1):
                    new_diffs.add(row[i] - x)
                
                # Check conflicts
                conflict = False
                for d in new_diffs:
                    if d in used_diffs and d not in old_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Valid move - update
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    row[j] = x
                    found = True
                    
                    new_scope = compute_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    break
            
            if not found:
                no_improve_count += 1
            
            if no_improve_count > 500:
                break
        
        return best_rows, best_scope
    
    # Phase 1: Get a good initial solution
    rows, used_diffs = greedy_init()
    best_rows, best_scope = local_search(rows, used_diffs, max_iterations=2000)
    
    # Phase 2: Multiple restarts with different seeds
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try different greedy orders
        random.seed(restart * 12345 + 67890)
        rows, used_diffs = greedy_init()
        
        # Apply local search
        candidate_rows, candidate_scope = local_search(rows, used_diffs, max_iterations=1500)
        
        if candidate_scope < best_scope:
            best_scope = candidate_scope
            best_rows = [r[:] for r in candidate_rows]
    
    # Phase 3: Final intensive local search on the best solution
    if time.time() - start_time < time_limit - 5:
        # Reconstruct used_diffs from best_rows
        used_diffs = set()
        for r in best_rows:
            used_diffs |= get_row_diffs(r)
        
        best_rows, best_scope = local_search(best_rows, used_diffs, max_iterations=10000)
    
    return best_rows
