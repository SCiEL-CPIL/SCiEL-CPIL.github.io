import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Number of entries per row
    n_entries = k + 1
    # Total differences needed: n_rows * (k*(k+1)//2)
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    # Greedy construction: build all rows simultaneously
    # At each step, pick the row with the smallest current max to extend
    # This balances the growth across rows
    
    def greedy_simultaneous(seed=0):
        rng = random.Random(seed)
        rows = [[0]] * n_rows
        used = set()
        
        # Track how many entries each row has
        counts = [1] * n_rows
        
        while any(c < n_entries for c in counts):
            # Find rows that still need entries
            candidates = [i for i in range(n_rows) if counts[i] < n_entries]
            if not candidates:
                break
            
            # Pick the row with the smallest current max (to balance)
            # Add some randomness to explore
            if seed > 0 and rng.random() < 0.3:
                row_idx = rng.choice(candidates)
            else:
                row_idx = min(candidates, key=lambda i: rows[i][-1])
            
            row = rows[row_idx]
            last = row[-1]
            
            # Find the smallest x > last such that all new differences are unused
            x = last + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
                # Safety: if x gets too large, break to avoid infinite loop
                if x > 500:
                    return None
            
            row.append(x)
            used.update(new_diffs)
            counts[row_idx] += 1
        
        return rows
    
    # Try multiple strategies and seeds
    best_rows = None
    best_max = float('inf')
    
    # Strategy 1: Pure greedy (seed=0)
    rows = greedy_simultaneous(0)
    if rows is not None:
        m = max(max(r) for r in rows)
        if m < best_max:
            best_max = m
            best_rows = [r[:] for r in rows]
    
    # Strategy 2: Greedy with randomization
    for seed in range(1, 200):
        if time.time() > deadline - 10:
            break
        rows = greedy_simultaneous(seed)
        if rows is not None:
            m = max(max(r) for r in rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
    
    # Local search: try to reduce the maximum
    # For each row, try to shift the last element down if possible
    # Or try alternative values for elements
    
    def try_improve(rows, used, max_val):
        """Try to reduce max_val by modifying elements."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Find the row with the largest max
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            current_max = row[-1]
            
            # Try to reduce the last element
            # We need to find a smaller value that maintains distinct differences
            # and doesn't conflict with used differences
            # This is complex; try a simpler approach:
            # Try removing the last element and re-adding with a smaller value
            
            # Actually, let's try a different approach:
            # For each row, try to find an alternative construction with smaller max
            
            pass
        return rows
    
    # Alternative local search: for each row, try to find a smaller valid ending
    # by backtracking the last few elements
    
    def local_search(rows):
        """Try to improve the solution by modifying individual rows."""
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Compute used differences from all rows except the one being modified
            for row_idx in range(n_rows):
                # Remove this row's differences from used set
                row = rows[row_idx]
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        row_diffs.add(row[j] - row[i])
                
                # Build used set without this row
                used_without = set()
                for other_idx in range(n_rows):
                    if other_idx == row_idx:
                        continue
                    other = rows[other_idx]
                    for i in range(len(other)):
                        for j in range(i+1, len(other)):
                            used_without.add(other[j] - other[i])
                
                # Try to reconstruct this row with a smaller max
                # Start from the current row and try to reduce
                best_row = None
                best_row_max = row[-1]
                
                # Try greedy reconstruction with a cap
                cap = best_row_max - 1
                new_row = [0]
                x = 0
                while len(new_row) < n_entries:
                    x += 1
                    if x > cap:
                        break
                    new_diffs = {x - a for a in new_row}
                    if not (new_diffs & used_without) and not (new_diffs & set()):
                        # Also need to check against other elements in new_row
                        # Actually new_diffs already includes all diffs to existing elements
                        # But we need to make sure they don't conflict with each other
                        # They can't since they're all positive and from distinct pairs
                        # Wait, we need them to be distinct from each other too
                        # Check for duplicates within new_diffs
                        if len(new_diffs) == len(new_row):
                            new_row.append(x)
                
                if len(new_row) == n_entries and new_row[-1] < best_row_max:
                    best_row = new_row
                    best_row_max = new_row[-1]
                
                if best_row is not None:
                    rows[row_idx] = best_row
                    improved = True
        
        return rows
    
    if best_rows is not None:
        best_rows = local_search(best_rows)
        new_max = max(max(r) for r in best_rows)
        if new_max < best_max:
            best_max = new_max
    
    # Verify validity
    def is_valid(rows):
        used = set()
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            used.update(diffs)
        return True
    
    if best_rows is None or not is_valid(best_rows):
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_entries:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
    
    return best_rows
