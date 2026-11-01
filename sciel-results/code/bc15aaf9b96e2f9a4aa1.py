import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Greedy Initialization
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
        
    # Incremental state
    current_rows = [row[:] for row in rows]
    current_used = set(used)
    current_max = max(max(row) for row in current_rows)
    
    best_rows = [row[:] for row in current_rows]
    best_max = current_max
    
    def get_row_max(r):
        return max(r)

    T = 50.0
    min_T = 0.01
    cooling = 0.995
    
    while time.time() < deadline and T > min_T:
        new_rows = [row[:] for row in current_rows]
        new_used = set(current_used)
        
        r_idx = random.randint(0, n_rows - 1)
        row = new_rows[r_idx]
        
        # Pick a random non-zero index
        j_idx = random.randint(1, len(row) - 1)
        old_val = row[j_idx]
        
        # Generate new value
        # Bias towards reducing max
        if random.random() < 0.7:
            # Try to decrease
            delta = random.randint(1, 10)
            new_val = old_val - delta
            if new_val <= row[j_idx - 1]:
                new_val = row[j_idx - 1] + 1
        else:
            # Random move
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val <= row[j_idx - 1]:
                new_val = row[j_idx - 1] + 1
            if new_val >= row[j_idx + 1]:
                new_val = row[j_idx + 1] - 1
                
        if new_val == old_val:
            T *= cooling
            continue
            
        # Check validity incrementally
        # Remove old diffs
        valid_move = True
        old_diffs = []
        for i in range(len(row)):
            if i == j_idx: continue
            d = abs(row[i] - old_val)
            old_diffs.append(d)
            # We need to know if this diff appears elsewhere
            # For now, just remove from new_used
            if d in new_used:
                # It might appear multiple times? No, in a valid state, it appears exactly once.
                # But if we are in an invalid state, it might appear multiple times.
                # To be safe, we should count occurrences.
                pass
        
        # This incremental set logic is tricky if the current state is invalid (duplicates exist).
        # If current state is valid, each diff appears exactly once.
        # If we allow invalid states in SA, we need to handle counts.
        
        # Let's stick to a simpler approach: 
        # Only accept moves that result in a VALID state?
        # That's Hill Climbing, not SA.
        # But SA with invalid states is complex to implement correctly with sets.
        
        # Alternative: Use a cost function that counts conflicts, but use a fast check.
        # Given the small size, maybe just recalculating conflicts is fast enough?
        # N=7, K=6. Pairs per row = 15. Total pairs = 105.
        # Checking 105 pairs is very fast.
        
        # Let's revert to the full conflict check but optimize the loop.
        
        # Actually, let's try a different move: 
        # "Swap" two elements in the same row? No, order must be preserved.
        # "Shift" a suffix of the row?
        
        # Let's try a simpler heuristic: 
        # Randomly pick a row. 
        # Try to reduce the largest element by 1. 
        # If valid, accept. 
        # If not, try reducing by 2, etc.
        # If we can't reduce, try a random perturbation.
        
        # This is essentially a local search.
        
        # Let's implement the full conflict check version as it's safer.
        
        row[j_idx] = new_val
        
        # Calculate new conflicts
        # We can just check if the new differences introduced by new_val conflict with the rest of the set.
        # But "rest of the set" includes other rows.
        
        # Let's just do the full count. It's O(100). 100 * 10000 steps = 1M ops. Fine.
        
        conflicts = 0
        seen = set()
        for r in new_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
                        
        new_max = max(max(r) for r in new_rows)
        new_cost = new_max + 100 * conflicts
        current_cost = current_max + 100 * (0 if current_used == seen else 1) # This is wrong, current_used is a set, not a count.
        
        # I need to track current conflicts count.
        
        break # This block is messy. Let's restart the logic in the final code.
        
    return best_rows
