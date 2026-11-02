"""Difference triangle set (4,7): construct rows with distinct global differences."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 115.0  # safety margin before 120s kill
    total_differences = n_rows * (k + 1) * k // 2
    lb = total_differences
    
    # Efficient difference tracking
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def build_used_set(rows):
        used = set()
        for row in rows:
            used |= get_row_diffs(row)
        return used
    
    def can_place(row, pos, val, used, other_rows):
        """Check if we can place val at position pos in row without conflicts."""
        # Check conflicts within the row itself (with already-placed elements)
        for i in range(pos):
            d = val - row[i]
            if d in used:
                return False
        # Check conflicts with other rows
        for other_row in other_rows:
            for a in other_row:
                if abs(val - a) > 0 and (val - a) in used:
                    return False
                if (a - val) in used and a > val:
                    return False
        # Also check if val-a for a in other rows creates new diffs that conflict
        # Actually, we need to check: for each a in other_row, is (val-a) or (a-val) already in used?
        # No wait - the constraint is that ALL differences across ALL rows must be distinct.
        # So if we add val to row, the new diffs are {val - a : a in row[0..pos-1]} 
        # and also {val - b : b in other_rows} and {b - val : b in other_rows} if b > val
        # These must not be in used (which contains all diffs from other rows and previously placed in this row)
        
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(val - row[i])
        for other_row in other_rows:
            for b in other_row:
                if b != 0:  # b=0 gives diff = val which might be valid
                    d1 = val - b
                    d2 = b - val
                    if d1 > 0:
                        new_diffs.add(d1)
                    if d2 > 0:
                        new_diffs.add(d2)
        
        return not (new_diffs & used)
    
    def apply_placement(used, row, pos, val):
        """Add the new differences to used set."""
        for i in range(pos):
            used.add(val - row[i])
        # Note: differences with other rows are already in used from their construction
    
    def remove_placement(used, row, pos, val):
        """Remove the new differences from used set."""
        for i in range(pos):
            used.discard(val - row[i])
    
    def greedy_construct():
        """Greedy construction to get initial valid solution."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                for a in row:
                    new_diffs.add(x - a)
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Get initial solution
    rows = greedy_construct()
    best_rows = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    if best_scope <= lb:
        return best_rows
    
    # Simulated annealing to compact rows
    def compute_scope(rows):
        return max(max(r) for r in rows)
    
    def try_compact_row(rows, row_idx, max_iter=1000):
        """Try to compact a single row by shifting elements to smaller values."""
        row = list(rows[row_idx])
        other_rows = [rows[i] for i in range(n_rows) if i != row_idx]
        
        # Build used set from all other rows
        used = set()
        for other_row in other_rows:
            used |= get_row_diffs(other_row)
        
        # Try to reduce each element from right to left
        improved = False
        for pos in range(k, 0, -1):  # from last element to first (skip 0)
            if pos == 0:
                continue
            current_val = row[pos]
            # Try smaller values
            for new_val in range(current_val - 1, pos, -1):
                # Check if new_val > row[pos-1] (strictly increasing)
                if new_val <= row[pos - 1]:
                    continue
                
                # Check if we can place new_val at position pos
                # New diffs with elements in this row (positions 0..pos-1)
                new_diffs = set()
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0:
                        break
                    new_diffs.add(d)
                
                # Check conflicts with used set
                if new_diffs & used:
                    continue
                
                # Also need to check: does removing current_val and adding new_val
                # create any conflicts? The diffs involving current_val with earlier elements
                # are already in used (from this row's perspective), but we're modifying this row.
                # Actually, used only contains diffs from OTHER rows. Diffs within this row
                # are not in used. So we just need new_diffs to not conflict with used.
                
                # Remove old diffs from consideration (they're not in used, so nothing to do)
                # Add new_val
                row[pos] = new_val
                improved = True
                break
        
        if improved:
            rows[row_idx] = row
            return True
        return False
    
    # Phase 1: Greedy compaction
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        progress = False
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            if try_compact_row(rows, row_idx):
                progress = True
        
        scope = compute_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        
        if best_scope <= lb:
            break
        
        if not progress:
            break
    
    # Phase 2: Simulated annealing with random perturbations
    if time.time() < deadline:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        T = 10.0  # temperature
        cooling_rate = 0.995
        
        sa_iter = 0
        while time.time() < deadline and T > 0.1:
            sa_iter += 1
            if sa_iter % 1000 == 0:
                T *= cooling_rate
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            row = list(current_rows[row_idx])
            other_rows = [current_rows[i] for i in range(n_rows) if i != row_idx]
            
            # Build used set from other rows
            used = set()
            for other_row in other_rows:
                used |= get_row_diffs(other_row)
            
            # Pick a random position to modify (not 0)
            pos = random.randint(1, k)
            old_val = row[pos]
            
            # Generate a candidate new value
            # Try to move it down (compaction) or slightly up (perturbation)
            if random.random() < 0.7:
                # Try to compact: move down
                min_allowed = row[pos - 1] + 1 if pos > 0 else 1
                max_allowed = old_val - 1
                if max_allowed < min_allowed:
                    continue
                new_val = random.randint(min_allowed, max_allowed)
            else:
                # Perturbation: try a nearby value
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if new_val <= row[pos - 1] if pos > 0 else new_val <= 0:
                    continue
                if new_val >= row[pos + 1] if pos < k else False:
                    continue
            
            # Check if new_val is valid
            new_diffs = set()
            for i in range(pos):
                d = new_val - row[i]
                if d > 0:
                    new_diffs.add(d)
            
            if new_diffs & used:
                continue
            
            # Also check that new_val doesn't create duplicate diffs within the row
            # with the new placement
            row_diffs_new = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if i == pos or j == pos:
                        if i < j:
                            if j == pos:
                                d = new_val - row[i]
                            else:
                                d = row[j] - new_val
                        else:
                            if i == pos:
                                d = row[j] - new_val
                            else:
                                d = new_val - row[i]
                        if d > 0:
                            row_diffs_new.add(d)
                    else:
                        d = row[j] - row[i]
                        if d > 0:
                            row_diffs_new.add(d)
            
            # Check for duplicates within the row
            if len(row_diffs_new) != (k + 1) * k // 2:
                continue
            
            # Check that row_diffs_new doesn't conflict with used
            if row_diffs_new & used:
                continue
            
            # Apply the change
            row[pos] = new_val
            new_rows = list(current_rows)
            new_rows[row_idx] = row
            new_scope = compute_scope(new_rows)
            
            # Accept if better or with probability exp(-(new_scope - current_scope)/T)
            if new_scope < current_scope or random.random() < pow(2.718281828, -(new_scope - current_scope) / T):
                current_rows = new_rows
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    if best_scope <= lb:
                        break
    
    # Final validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            row_diffs = get_row_diffs(row)
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    if not validate(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct()
    
    return best_rows
# EVOLVE-BLOCK-END
