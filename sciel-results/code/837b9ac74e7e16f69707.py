"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    total_diffs = n_rows * k * (k + 1) // 2
    # Counting bound: M-1 >= total_diffs, so M >= total_diffs + 1
    counting_bound = total_diffs + 1
    
    def get_all_diffs(rows):
        """Get all pairwise differences from all rows."""
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are distinct."""
        diffs = get_all_diffs(rows)
        return len(diffs) == len(set(diffs))
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_row(used_diffs, k, start_val=0):
        """Build a single row greedily given already-used differences."""
        row = [start_val]
        for _ in range(k):
            found = False
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                break
        return row
    
    def greedy_construct(n_rows, k, randomize=False):
        """Construct rows greedily, with optional randomization."""
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            if randomize:
                # Randomize the order of trying values
                row = [0]
                for _ in range(k):
                    candidates = []
                    x = row[-1] + 1
                    # Look ahead a bit
                    max_look = 20
                    for trial in range(max_look):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            candidates.append(x)
                            if len(candidates) >= 3:
                                break
                        x += 1
                    if candidates:
                        if randomize:
                            choice = random.choice(candidates[:3])
                        else:
                            choice = candidates[0]
                        row.append(choice)
                        used_diffs |= {choice - a for a in row[:-1]}
                    else:
                        # Fallback: just keep incrementing
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used_diffs:
                                row.append(x)
                                used_diffs |= new_diffs
                                break
                            x += 1
            else:
                row = greedy_row(used_diffs, k)
            rows.append(row)
        return rows
    
    # Initial construction
    best_rows = greedy_construct(n_rows, k)
    best_score = max_entry(best_rows)
    
    # Try randomized initializations
    for attempt in range(5):
        if time.time() > deadline:
            break
        rows = greedy_construct(n_rows, k, randomize=True)
        score = max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    # Local search to improve
    def local_search(rows, max_time=110):
        """Local search to minimize the maximum entry."""
        rows = [list(r) for r in rows]
        current_score = max_entry(rows)
        best_rows = [list(r) for r in rows]
        best_score = current_score
        
        start_time = time.time()
        temperature = 100.0
        cooling = 0.9995
        min_temperature = 0.1
        
        while time.time() < start_time + (max_time - (time.time() - start_time)) and temperature > min_temperature:
            # Pick a random row and a random position to modify
            r_idx = random.randint(0, len(rows) - 1)
            row = rows[r_idx]
            
            if len(row) <= 1:
                continue
            
            # Try to reduce a random element
            pos = random.randint(1, len(row) - 1)  # Don't change position 0 (must be 0)
            old_val = row[pos]
            
            # Try to decrease the value
            new_val = old_val - random.randint(1, 5)
            if new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            
            if new_val == old_val:
                continue
            
            # Check if changing row[r_idx][pos] to new_val keeps validity
            # Remove old diffs for this row
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_diffs.add(row[j] - row[i])
            
            # Create modified row
            mod_row = list(row)
            mod_row[pos] = new_val
            
            # Check if mod_row is strictly increasing
            if not all(mod_row[i] < mod_row[i+1] for i in range(len(mod_row) - 1)):
                continue
            
            # Get new diffs for modified row
            new_diffs = set()
            for i in range(len(mod_row)):
                for j in range(i + 1, len(mod_row)):
                    new_diffs.add(mod_row[j] - mod_row[i])
            
            # Get all diffs from other rows
            other_diffs = set()
            for i, other_row in enumerate(rows):
                if i == r_idx:
                    continue
                for a in range(len(other_row)):
                    for b in range(a + 1, len(other_row)):
                        other_diffs.add(other_row[b] - other_row[a])
            
            # Check for conflicts
            conflict = new_diffs & other_diffs
            if conflict:
                # Accept with probability based on temperature
                delta = len(conflict)
                if random.random() < min(1.0, (1.0 / (1.0 + delta)) * temperature / 10.0):
                    rows[r_idx] = mod_row
                    current_score = max_entry(rows)
                    if current_score < best_score:
                        best_score = current_score
                        best_rows = [list(r) for r in rows]
            else:
                # No conflict, accept
                rows[r_idx] = mod_row
                current_score = max_entry(rows)
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [list(r) for r in rows]
            
            temperature *= cooling
        
        return best_rows, best_score
    
    # Run local search
    if time.time() < deadline - 5:
        best_rows, best_score = local_search(best_rows)
    
    # Final check: ensure validity
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows
# EVOLVE-BLOCK-END
