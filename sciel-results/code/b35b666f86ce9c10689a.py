import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    
    Uses randomized local search (simulated annealing / hill climbing)
    to find a compact configuration.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_scope(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    def get_conflicts(rows):
        """Return count of duplicate differences."""
        from collections import Counter
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        counter = Counter(diffs)
        return sum(c - 1 for c in counter.values() if c > 1)
    
    def generate_greedy():
        """Generate an initial solution using greedy algorithm."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Generate initial solution
    best_rows = generate_greedy()
    best_scope = get_scope(best_rows)
    best_conflicts = get_conflicts(best_rows)
    
    if best_conflicts == 0:
        # Already valid, try to improve
        pass
    
    # Local search with random restarts
    def local_search(rows, time_limit=30.0):
        """Run local search on the given rows."""
        rows = [row[:] for row in rows]
        current_scope = get_scope(rows)
        current_conflicts = get_conflicts(rows)
        best_local = rows[:]
        best_local_scope = current_scope
        best_local_conflicts = current_conflicts
        
        t = time.time()
        max_iterations = 100000
        iterations = 0
        no_improve = 0
        
        while time.time() - t < time_limit and iterations < max_iterations:
            iterations += 1
            
            # Pick a random row and position to modify
            r = random.randint(0, n_rows - 1)
            if len(rows[r]) <= 1:
                continue
            # Only modify non-zero entries (index >= 1)
            if len(rows[r]) < 2:
                continue
            pos = random.randint(1, len(rows[r]) - 1)
            
            old_val = rows[r][pos]
            
            # Try nearby values
            found = False
            for delta in range(1, 10):
                for new_val in [old_val - delta, old_val + delta]:
                    if new_val < 0:
                        continue
                    # Ensure row remains strictly increasing
                    if pos > 0 and new_val <= rows[r][pos - 1]:
                        continue
                    if pos < len(rows[r]) - 1 and new_val >= rows[r][pos + 1]:
                        continue
                    
                    rows[r][pos] = new_val
                    new_conflicts = get_conflicts(rows)
                    new_scope = get_scope(rows)
                    
                    # Accept if fewer conflicts, or same conflicts but smaller scope
                    if new_conflicts < current_conflicts:
                        current_conflicts = new_conflicts
                        current_scope = new_scope
                        found = True
                        break
                    elif new_conflicts == current_conflicts and new_scope < current_scope:
                        current_conflicts = new_conflicts
                        current_scope = new_scope
                        found = True
                        break
                    else:
                        # Revert
                        rows[r][pos] = old_val
                
                if found:
                    break
            
            if not found:
                rows[r][pos] = old_val
                no_improve += 1
            else:
                no_improve = 0
            
            # Update best local solution
            if current_conflicts < best_local_conflicts or \
               (current_conflicts == best_local_conflicts and current_scope < best_local_scope):
                best_local = [row[:] for row in rows]
                best_local_scope = current_scope
                best_local_conflicts = current_conflicts
            
            if no_improve > 5000:
                break
        
        return best_local, best_local_scope, best_local_conflicts
    
    # Run local search with random restarts
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        
        # Create a perturbed version of the best solution
        if restarts == 1:
            rows = [row[:] for row in best_rows]
        else:
            # Perturb: randomly change some entries
            rows = [row[:] for row in best_rows]
            for _ in range(random.randint(1, 5)):
                r = random.randint(0, n_rows - 1)
                if len(rows[r]) <= 1:
                    continue
                pos = random.randint(1, len(rows[r]) - 1)
                delta = random.randint(-5, 5)
                new_val = rows[r][pos] + delta
                if new_val > 0 and (pos == 0 or new_val > rows[r][pos - 1]) and \
                   (pos == len(rows[r]) - 1 or new_val < rows[r][pos + 1]):
                    rows[r][pos] = new_val
        
        result_rows, result_scope, result_conflicts = local_search(rows)
        
        if result_conflicts < best_conflicts or \
           (result_conflicts == best_conflicts and result_scope < best_scope):
            best_rows = result_rows
            best_scope = result_scope
            best_conflicts = result_conflicts
        
        if best_conflicts == 0:
            # Valid solution found, keep trying to reduce scope
            pass
    
    return best_rows
