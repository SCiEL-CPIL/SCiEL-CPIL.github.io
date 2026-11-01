import random
import math

def construct(n_rows, k):
    # Initial greedy construction
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
                used.update(new_diffs)
        rows.append(row)
    
    # Calculate initial max span
    current_max = max(max(row) for row in rows)
    
    # Local Search / Simulated Annealing to reduce max span
    # We will try to move points to smaller positions if possible.
    
    # Pre-calculate diffs for each row for faster removal/addition?
    # Since k is small (8), recalculating is fine.
    
    def get_diffs(row, idx):
        """Get all differences involving row[idx]."""
        val = row[idx]
        diffs = set()
        for j, v in enumerate(row):
            if i != j:
                diffs.add(abs(val - v))
        return diffs

    # Actually, it's faster to just re-check the whole row against global used set
    # when proposing a move.
    
    def is_valid_move(row, idx, new_val):
        """Check if placing new_val at idx in row is valid with respect to 
        the rest of the row and the global used set (excluding current diffs of this point)."""
        # 1. Check sorted order
        if idx > 0 and new_val <= row[idx-1]:
            return False
        if idx < len(row)-1 and new_val >= row[idx+1]:
            return False
        
        # 2. Check internal collisions (diffs with other points in same row)
        # We need to ensure new_val - row[j] != row[j] - new_val (always true for distinct)
        # and new_val - row[j] not in {row[a]-row[b] for other a,b}
        
        # Let's construct the new row temporarily
        old_val = row[idx]
        row[idx] = new_val
        
        # Check internal validity of this row
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    row[idx] = old_val
                    return False
                row_diffs.add(d)
        
        # Check against global used set
        # We must exclude the differences that were previously contributed by this point
        # Actually, it's easier to just check if any new diff is in (used - old_diffs)
        
        # Let's just do a full check for simplicity, given small size.
        # But we need to know which diffs are "free".
        
        # Better approach:
        # 1. Remove old diffs of this point from a temporary copy of used? No, too slow.
        # 2. Calculate new_diffs.
        # 3. Calculate old_diffs.
        # 4. Check if any diff in new_diffs is in used AND not in old_diffs.
        
        row[idx] = old_val # Restore
        
        old_diffs = set()
        for j, v in enumerate(row):
            if j != idx:
                old_diffs.add(abs(old_val - v))
        
        new_diffs = set()
        for j, v in enumerate(row):
            if j != idx:
                new_diffs.add(abs(new_val - v))
        
        # Check for internal conflicts in the new row state
        # We already checked sorted order.
        # We need to ensure that for any pair (i,j) in the new row, the diff is unique.
        # This is equivalent to:
        # - Diffs involving the moved point are unique among themselves (guaranteed by distinct values)
        # - Diffs involving the moved point are not equal to any diff NOT involving the moved point.
        
        # Diffs not involving the moved point remain unchanged and were valid.
        # So we only need to check:
        # 1. new_diffs are distinct (yes, values distinct)
        # 2. new_diffs do not intersect with (used - old_diffs)
        
        if new_diffs & (used - old_diffs):
            return False
            
        # Also check if new_diffs have internal duplicates? 
        # No, because row values are distinct, differences from one point to others are distinct 
        # IF the other points are distinct. 
        # Wait: |new_val - a| == |new_val - b| implies a == b or a + b == 2*new_val.
        # Since a, b are distinct integers, it's possible that |new_val - a| == |new_val - b|.
        # Example: row [0, 2, 4], move 2 to 1. Diffs: |1-0|=1, |1-4|=3. Distinct.
        # Example: row [0, 4, 8], move 4 to 2. Diffs: |2-0|=2, |2-8|=6. Distinct.
        # Is it possible to have |new_val - a| == |new_val - b|?
        # Yes, if new_val is the midpoint of a and b.
        # So we MUST check if new_diffs has duplicates.
        
        if len(new_diffs) != len([abs(new_val - v) for j, v in enumerate(row) if j != idx]):
            return False
            
        return True

    # Optimization Loop
    # We try to reduce the maximum span.
    # Strategy: Iterate through rows, try to move the last element to a smaller value.
    # If stuck, try other elements.
    
    # To avoid getting stuck in local minima, we can add some randomness.
    # Let's do a simple hill climbing with random restarts or random moves.
    
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Time limit check: 120s. Let's run for a fixed number of iterations or time.
    import time
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    while time.time() - start_time < max_time:
        # Pick a row to modify
        # Prefer the row with the largest max
        row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[row_idx]
        
        # Pick a point to move
        # Try moving the last point first
        point_idx = len(row) - 1
        
        # Try to find a smaller valid position
        current_val = row[point_idx]
        # Search downwards
        found = False
        for new_val in range(current_val - 1, 0, -1):
            if is_valid_move(row, point_idx, new_val):
                # Accept move
                # Update used set
                old_diffs = set(abs(current_val - v) for j, v in enumerate(row) if j != point_idx)
                new_diffs = set(abs(new_val - v) for j, v in enumerate(row) if j != point_idx)
                
                used -= old_diffs
                used |= new_diffs
                
                row[point_idx] = new_val
                found = True
                break
        
        if not found:
            # Try other points in this row or other rows
            # For simplicity, let's just pick a random row and random point
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            point_idx = random.randint(1, len(row) - 1) # Don't move 0
            
            current_val = row[point_idx]
            # Try a random smaller value
            if current_val > 1:
                new_val = random.randint(1, current_val - 1)
                if is_valid_move(row, point_idx, new_val):
                    old_diffs = set(abs(current_val - v) for j, v in enumerate(row) if j != point_idx)
                    new_diffs = set(abs(new_val - v) for j, v in enumerate(row) if j != point_idx)
                    
                    used -= old_diffs
                    used |= new_diffs
                    
                    row[point_idx] = new_val
                    found = True
            
        if not found:
            # If we can't make any improvement, maybe we are stuck.
            # We can try a "kick" move: move a point to a larger value to escape local minimum?
            # But that increases span, which is bad.
            # Instead, let's just break if we've been stuck for a while?
            # Or just continue trying random moves.
            pass
        
        # Update best
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in rows]

    return best_rows
