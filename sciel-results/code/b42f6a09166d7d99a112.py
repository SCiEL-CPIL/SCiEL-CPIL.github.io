import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s limit
    start_time = time.time()
    
    # Total differences needed: n_rows * C(k+1, 2)
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on max entry
    # We need total_diffs distinct positive integers, each <= max_entry
    # So max_entry >= total_diffs
    lower_bound = total_diffs
    
    best_result = None
    best_max = float('inf')
    
    def generate_rows_up_to(max_val, n_rows, k):
        """Generate all valid rows (lists of k+1 increasing ints starting at 0) with max <= max_val."""
        # A row is determined by choosing k values from 1..max_val
        # row = [0] + sorted(chosen)
        # Number of such rows: C(max_val, k)
        rows = []
        for combo in combinations(range(1, max_val + 1), k):
            row = [0] + list(combo)
            rows.append(row)
        return rows
    
    def row_diffs(row):
        """Get all pairwise differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def backtrack_search(max_val, n_rows, k):
        """Backtracking search for n_rows rows with max entry <= max_val."""
        # Pre-generate all valid rows up to max_val
        all_rows = generate_rows_up_to(max_val, n_rows, k)
        
        # Pre-compute differences for each row
        row_diffs_list = [row_diffs(r) for r in all_rows]
        
        # Sort rows by their maximum value (prefer smaller rows first)
        order = sorted(range(len(all_rows)), key=lambda i: all_rows[i][-1])
        
        used_diffs = set()
        chosen = []
        
        def bt(idx, chosen_count):
            if time.time() > deadline:
                return None
            if chosen_count == n_rows:
                return chosen[:]
            
            for i in order:
                if i in used_indices:
                    continue
                # Check if this row's differences are all unused
                if row_diffs_list[i] & used_diffs:
                    continue
                
                # Try this row
                used_indices.add(i)
                new_used = used_diffs | row_diffs_list[i]
                chosen.append(all_rows[i])
                
                result = bt(idx + 1, chosen_count + 1)
                if result is not None:
                    return result
                
                # Backtrack
                chosen.pop()
                used_indices.discard(i)
            
            return None
        
        used_indices = set()
        return bt(0, 0)
    
    # Try increasing max_val starting from lower bound
    # But C(max_val, k) can be huge, so we need to be smart
    
    # For k=5, n_rows=7: total_diffs = 7*15 = 105
    # Lower bound = 105, best known = 112
    
    # Instead of brute force backtracking over all C(max_val, k) rows,
    # use a smarter approach: random sampling + local search
    
    def random_valid_row(max_val, k):
        """Generate a random valid row with max <= max_val."""
        if max_val < k:
            return None
        chosen = sorted(random.sample(range(1, max_val + 1), k))
        return [0] + chosen
    
    def is_valid_configuration(rows, n_rows, k):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True
    
    def local_search(rows, max_val, n_rows, k, iterations=5000):
        """Local search to minimize max entry."""
        current_rows = [row[:] for row in rows]
        current_max = max(max(row) for row in current_rows)
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Try to reduce the max by modifying a random row
            row_idx = random.randint(0, n_rows - 1)
            old_row = current_rows[row_idx]
            old_diffs = row_diffs(old_row)
            
            # Generate a new random row with smaller max
            target_max = old_row[-1] - 1
            if target_max < k:
                target_max = k
            
            new_row = None
            for _ in range(100):
                candidate = random_valid_row(target_max, k)
                if candidate is None:
                    break
                cand_diffs = row_diffs(candidate)
                # Check against all other rows' diffs
                conflict = False
                for j, other_row in enumerate(current_rows):
                    if j == row_idx:
                        continue
                    other_diffs = row_diffs(other_row)
                    if cand_diffs & other_diffs:
                        conflict = True
                        break
                if not conflict:
                    new_row = candidate
                    break
            
            if new_row is not None:
                current_rows[row_idx] = new_row
                new_max = max(max(row) for row in current_rows)
                if new_max < current_max:
                    current_max = new_max
        
        return current_rows, current_max
    
    # Strategy: try to find a solution with max_val from lower_bound upward
    # Use random initialization + local search
    
    best_rows = None
    best_max_val = float('inf')
    
    # First, try a greedy approach to get an initial solution
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_diffs:
                row.append(x)
                used_diffs |= new
        greedy_rows.append(row)
    
    greedy_max = max(max(row) for row in greedy_rows)
    best_rows = greedy_rows
    best_max_val = greedy_max
    
    # Now try to improve with local search
    # Try multiple random restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        random_rows = []
        valid = True
        for _ in range(n_rows):
            # Try to generate a random row that doesn't conflict
            row = None
            for attempt in range(200):
                # Use a range around the current best
                max_try = best_max_val
                candidate = random_valid_row(max_try, k)
                if candidate is None:
                    continue
                cand_diffs = row_diffs(candidate)
                conflict = False
                for existing in random_rows:
                    if cand_diffs & row_diffs(existing):
                        conflict = True
                        break
                if not conflict:
                    row = candidate
                    break
            
            if row is None:
                valid = False
                break
            random_rows.append(row)
        
        if not valid:
            continue
        
        random_max = max(max(row) for row in random_rows)
        
        # Local search to improve
        improved_rows, improved_max = local_search(random_rows, random_max, n_rows, k, iterations=3000)
        
        if improved_max < best_max_val:
            best_max_val = improved_max
            best_rows = improved_rows
    
    # Final polish: try to reduce further
    if best_rows is not None and time.time() < deadline - 5:
        polished_rows, polished_max = local_search(best_rows, best_max_val, n_rows, k, iterations=5000)
        if polished_max < best_max_val:
            best_max_val = polished_max
            best_rows = polished_rows
    
    return best_rows

