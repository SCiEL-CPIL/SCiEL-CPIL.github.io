"""Difference triangle set (7,5): minimize the largest mark (scope).

Strategy: Simulated Annealing / Local Search.
1. Initialize a valid configuration using a greedy method.
2. Iteratively modify one element of a random row.
3. Accept moves that maintain validity (distinct differences globally and within row)
   and improve or occasionally worsen the scope (largest element) based on temperature.
4. Return the best valid configuration found within the time limit.
"""
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # State:
    # rows: list of n_rows lists, each of length k+1, rows[r][0]=0, rows[r][1..k] > 0
    # used_diffs: set of all positive differences found in all rows
    
    def get_diffs(row):
        """Return list of differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid_row(row):
        """Check if a row has distinct internal differences and is strictly increasing."""
        if row[0] != 0:
            return False
        if any(row[i] >= row[i+1] for i in range(len(row)-1)):
            return False
        diffs = get_diffs(row)
        return len(diffs) == len(set(diffs))

    def calculate_scope(rows):
        return max(max(row) for row in rows)

    def greedy_init():
        """Generate an initial valid set of rows."""
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            # Greedily pick smallest available numbers
            while len(row) < k + 1:
                found = False
                # Try numbers from 1 upwards
                for x in range(1, 500):
                    # Check if x can be appended
                    # Must be > row[-1]
                    if x <= row[-1]:
                        continue
                    # Check new diffs
                    new_diffs = {x - a for a in row}
                    # Check against used_diffs
                    if new_diffs & used_diffs:
                        continue
                    # Check internal distinctness of new row
                    test_row = row + [x]
                    if is_valid_row(test_row):
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback: just pick a large random number to avoid infinite loop
                    x = random.randint(100, 200)
                    while x <= row[-1] or ({x - a for a in row} & used_diffs):
                        x += 1
                    row.append(x)
                    used_diffs |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows, used_diffs

    # Initialize
    rows, used_diffs = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = calculate_scope(best_rows)
    
    # Simulated Annealing Parameters
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.01
    
    def check_move_validity(r_idx, pos_idx, new_val):
        """
        Check if replacing rows[r_idx][pos_idx] with new_val is valid.
        Returns (is_valid, old_diffs, new_diffs)
        """
        old_val = rows[r_idx][pos_idx]
        if new_val == old_val:
            return True, set(), set()
        
        # 1. Check strictly increasing
        if pos_idx > 0 and new_val <= rows[r_idx][pos_idx - 1]:
            return False, set(), set()
        if pos_idx < k and new_val >= rows[r_idx][pos_idx + 1]:
            return False, set(), set()
            
        # 2. Check differences
        old_diffs = set()
        new_diffs = set()
        
        for i in range(k + 1):
            if i != pos_idx:
                old_diffs.add(abs(rows[r_idx][i] - old_val))
                new_diffs.add(abs(rows[r_idx][i] - new_val))
        
        # Check internal distinctness of new row
        test_row = rows[r_idx][:]
        test_row[pos_idx] = new_val
        if not is_valid_row(test_row):
            return False, set(), set()

        # Check global collisions
        remaining_used = used_diffs - old_diffs
        if new_diffs & remaining_used:
            return False, set(), set()
            
        return True, old_diffs, new_diffs

    def apply_move(r_idx, pos_idx, new_val, old_diffs, new_diffs):
        """Apply a valid move."""
        used_diffs.difference_update(old_diffs)
        rows[r_idx][pos_idx] = new_val
        used_diffs.update(new_diffs)

    def revert_move(r_idx, pos_idx, old_val, new_val, old_diffs, new_diffs):
        """Revert a move."""
        used_diffs.difference_update(new_diffs)
        rows[r_idx][pos_idx] = old_val
        used_diffs.update(old_diffs)

    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k) # Never change the 0
        
        # Generate a new value
        current_val = rows[r_idx][pos_idx]
        
        # 50% chance to try decreasing, 50% random
        if random.random() < 0.5 and current_val > 1:
            new_val = random.randint(1, current_val - 1)
        else:
            # Random value, biased towards lower values to minimize scope
            max_range = best_scope + 20
            new_val = random.randint(1, max_range)
            
        if new_val == current_val:
            continue
            
        # Check validity
        valid, old_diffs, new_diffs = check_move_validity(r_idx, pos_idx, new_val)
        
        if valid:
            old_val = rows[r_idx][pos_idx]
            
            # Apply move
            apply_move(r_idx, pos_idx, new_val, old_diffs, new_diffs)
            
            # Calculate new scope
            new_scope = calculate_scope(rows)
            
            # Acceptance criteria
            if new_scope <= best_scope or random.random() < pow(2.71828, -(new_scope - best_scope) / T):
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                revert_move(r_idx, pos_idx, old_val, new_val, old_diffs, new_diffs)
                
        T *= cooling_rate

    # Return best valid solution
    return best_rows
