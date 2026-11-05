import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry (simple counting bound)
    lower_bound = total_diffs
    
    # We need to find rows such that all differences are distinct.
    # Strategy: Simulated Annealing on the positions of the marks.
    # State: list of rows, each row is a list of k+1 integers.
    # Move: 
    #   1. Pick a random row and a random index (not 0).
    #   2. Perturb the value at that index.
    #   3. If the row becomes invalid (internal diffs collide), reject.
    #   4. If the set becomes invalid (cross-row diffs collide), accept with probability based on energy.
    #   5. Special "Compaction" move: Try to shift a row or specific elements left to reduce max entry.
    
    # Helper: Get all diffs for a row
    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # Helper: Check if a row is internally valid (Golomb ruler)
    def is_golomb(row):
        return len(get_row_diffs(row)) == len(row) * (len(row) - 1) // 2

    # Helper: Compute the "cost" or energy.
    # We want to minimize the largest entry, but we must maintain validity.
    # If invalid, energy = infinity (or a large penalty).
    # If valid, energy = max_entry.
    # Actually, for SA, we can define a penalty for collisions.
    # Let's use a cost function:
    # Cost = max_entry + P * (number_of_conflicts)
    # But we want valid solutions. So if conflicts > 0, cost is very high.
    # Better: 
    # If valid: cost = max_entry
    # If invalid: cost = max_entry + 1000 * (number_of_conflicts)
    
    def compute_cost(rows, used_diffs_count, total_conflicts):
        # This is a bit stateful. Let's compute from scratch or maintain state.
        # For speed, we will maintain state.
        pass

    # We will implement a stateful SA.
    # State:
    # rows: list of lists
    # used_diffs: set of all differences used
    # conflicts: dictionary mapping diff -> count (if count > 1, it's a conflict)
    # max_entry: current max entry
    
    # Initialize with a valid greedy solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    
    # Compute initial state
    def compute_state(rows):
        used_diffs = {}
        max_entry = 0
        for row in rows:
            max_entry = max(max_entry, max(row))
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    used_diffs[d] = used_diffs.get(d, 0) + 1
        
        conflicts = 0
        for d, c in used_diffs.items():
            if c > 1:
                conflicts += c - 1 # or just count the number of conflicting pairs? 
                # Actually, if c=2, there is 1 conflict. If c=3, there are 3 pairs? 
                # Let's just say conflict_count = sum(max(0, c-1))
        
        return used_diffs, max_entry, conflicts

    used_diffs, max_entry, conflicts = compute_state(rows)
    current_cost = max_entry + (1000 if conflicts > 0 else 0)
    best_rows = [row[:] for row in rows]
    best_cost = current_cost
    best_max = max_entry
    
    # SA Parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Move generators
    def move_perturb():
        """Randomly change one element in one row."""
        r_idx = random.randint(0, n_rows - 1)
        # Don't change index 0 (must be 0)
        c_idx = random.randint(1, k)
        
        # Proposed new value
        # Try to move it towards a random position, or shift it
        # Shift by a small random amount
        delta = random.randint(-10, 10)
        new_val = rows[r_idx][c_idx] + delta
        
        # Constraints:
        # 1. Must be > previous element
        # 2. Must be < next element
        # 3. Must be integer
        # 4. Must keep row strictly increasing
        
        if c_idx > 0:
            if new_val <= rows[r_idx][c_idx - 1]:
                return False, 0, 0
        if c_idx < k:
            if new_val >= rows[r_idx][c_idx + 1]:
                return False, 0, 0
        
        # Check if this move creates internal conflicts in the row
        # We need to check diffs involving c_idx
        old_val = rows[r_idx][c_idx]
        
        # Diffs to remove
        diffs_to_remove = []
        for i in range(len(rows[r_idx])):
            if i != c_idx:
                d = abs(rows[r_idx][c_idx] - rows[r_idx][i])
                diffs_to_remove.append((d, i, c_idx if i < c_idx else i, c_idx if i < c_idx else i))
        
        # Diffs to add
        diffs_to_add = []
        for i in range(len(rows[r_idx])):
            if i != c_idx:
                d = abs(new_val - rows[r_idx][i])
                # Check internal validity: new diffs must not collide with other diffs in the row
                # This is a pre-check. If it fails, we reject immediately.
                # But we also need to check if d collides with *other* diffs in the row that are NOT involving c_idx.
                # Let's just do the full check for simplicity first, or a fast check.
                # Fast check: 
                # The set of diffs in the row is D.
                # D' = (D - D_remove) U D_add
                # We need D' to be a set (no duplicates).
                # So, for each d in D_add, if d is in (D - D_remove) or if d is in D_add (duplicate in add), it's a conflict.
                
                pass # Logic too complex for inline, let's just apply and check global state, or do a local check.
        
        # Let's apply the move and check validity.
        rows[r_idx][c_idx] = new_val
        
        # Check internal validity of row r_idx
        if not is_golomb(rows[r_idx]):
            rows[r_idx][c_idx] = old_val
            return False, 0, 0
        
        # Update global state
        # Remove old diffs
        for i in range(len(rows[r_idx])):
            if i != c_idx:
                d = abs(old_val - rows[r_idx][i])
                used_diffs[d] -= 1
                if used_diffs[d] == 0:
                    del used_diffs[d]
        
        # Add new diffs
        for i in range(len(rows[r_idx])):
            if i != c_idx:
                d = abs(new_val - rows[r_idx][i])
                used_diffs[d] = used_diffs.get(d, 0) + 1
        
        # Recalculate conflicts and max_entry
        # This is O(N^2) if done from scratch. We can update incrementally?
        # Conflicts: 
        # A diff d is in conflict if count[d] > 1.
        # When we remove/add diffs, we can update a counter of conflicts.
        # Let's just recalculate conflicts for the affected diffs.
        # Actually, recalculating max_entry is O(N*K). Recalculating conflicts is O(Unique_Diffs).
        # For k=8, N=4, this is small.
        
        new_max = 0
        for row in rows:
            if max(row) > new_max:
                new_max = max(row)
        
        new_conflicts = 0
        for d, c in used_diffs.items():
            if c > 1:
                new_conflicts += c - 1
                
        return True, new_max, new_conflicts

    def move_compact():
        """Try to shift a row to the left or reduce a specific gap."""
        # Pick a row
        r_idx = random.randint(0, n_rows - 1)
        # Try to reduce the last element? Or shift the whole row?
        # Shifting the whole row is dangerous if it hits 0 or other rows? 
        # No, rows are independent except for diff collisions.
        # If we shift row r_idx by -1, all diffs remain the same.
        # So if the row is valid and doesn't contain 1 (so shift doesn't make it negative or zero-collision? No, 0 is fixed).
        # Wait, row[0] is 0. We cannot shift the whole row.
        # We can only change relative positions.
        # So "compact" means reducing the span by moving an inner point left or outer point left.
        # Let's try to move the last element (max) left.
        c_idx = k
        old_val = rows[r_idx][c_idx]
        
        # Try to decrease old_val by 1, 2, ...
        for step in range(1, 10):
            new_val = old_val - step
            if new_val <= rows[r_idx][c_idx - 1]:
                break
            
            # Apply
            rows[r_idx][c_idx] = new_val
            
            if not is_golomb(rows[r_idx]):
                continue
            
            # Update global state
            for i in range(len(rows[r_idx])):
                if i != c_idx:
                    d = abs(old_val - rows[r_idx][i])
                    used_diffs[d] -= 1
                    if used_diffs[d] == 0:
                        del used_diffs[d]
            
            for i in range(len(rows[r_idx])):
                if i != c_idx:
                    d = abs(new_val - rows[r_idx][i])
                    used_diffs[d] = used_diffs.get(d, 0) + 1
            
            new_max = 0
            for row in rows:
                if max(row) > new_max:
                    new_max = max(row)
            
            new_conflicts = 0
            for d, c in used_diffs.items():
                if c > 1:
                    new_conflicts += c - 1
            
            return True, new_max, new_conflicts
        
        return False, 0, 0

    # Main SA Loop
    T = 50.0
    while time.time() < deadline:
        # Choose move
        if random.random() < 0.5:
            success, new_max, new_conflicts = move_perturb()
        else:
            success, new_max, new_conflicts = move_compact()
        
        if not success:
            continue
            
        new_cost = new_max + (1000 if new_conflicts > 0 else 0)
        delta_cost = new_cost - current_cost
        
        # Acceptance criterion
        if delta_cost < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_cost / T)
            
        if accept:
            current_cost = new_cost
            max_entry = new_max
            conflicts = new_conflicts
            
            if conflicts == 0 and max_entry < best_max:
                best_max = max_entry
                best_rows = [row[:] for row in rows]
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_rows
