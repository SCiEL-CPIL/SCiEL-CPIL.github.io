import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.15, window=50):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = last + 200 # Look ahead window
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: find the next valid one strictly
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def is_move_valid(rows, r_idx, p_idx, new_val):
        """
        Check if moving rows[r_idx][p_idx] to new_val results in a valid configuration.
        This is O(N^2) but N is small.
        """
        row = rows[r_idx]
        
        # 1. Check strictly increasing within the row
        if p_idx > 0 and new_val <= row[p_idx-1]:
            return False
        if p_idx < row_len - 1 and new_val >= row[p_idx+1]:
            return False
        
        # 2. Collect all differences NOT involving the moved element
        #    These are "fixed" differences that must not be duplicated by the new differences
        fixed_diffs = set()
        for i, r in enumerate(rows):
            for a in range(len(r)):
                for b in range(a+1, len(r)):
                    # Skip if this difference involves the moved element
                    if i == r_idx and (a == p_idx or b == p_idx):
                        continue
                    fixed_diffs.add(r[b] - r[a])
        
        # 3. Generate new differences involving the moved element
        new_diffs = []
        # Differences with elements to the left
        for i in range(p_idx):
            new_diffs.append(new_val - row[i])
        # Differences with elements to the right
        for j in range(p_idx + 1, len(row)):
            new_diffs.append(row[j] - new_val)
        
        # 4. Check for duplicates within new_diffs
        if len(set(new_diffs)) != len(new_diffs):
            return False
            
        # 5. Check for duplicates against fixed_diffs
        for d in new_diffs:
            if d in fixed_diffs:
                return False
                
        return True

    # Initial Solution
    current_rows = greedy_init()
    
    # Ensure we start with a valid solution if possible, otherwise find one quickly
    # If greedy_init produces invalid, we need a repair or restart.
    # Given the nature of greedy, it usually produces valid rows locally, but global conflicts might exist?
    # No, greedy_init adds to a global used_diffs set, so it guarantees global uniqueness.
    # So current_rows is always valid.
    
    best_rows = [r[:] for r in current_rows]
    best_scope = get_scope(best_rows)
    
    # Main Search Loop
    # We alternate between "Targeted Reduction" and "Random Perturbation"
    # Targeted Reduction: Try to lower the global max.
    # Random Perturbation: Move a random element to a random valid position to escape local minima.
    
    T = 100.0
    cooling = 0.9995
    min_T = 0.1
    stagnant_count = 0
    
    while time.time() < deadline:
        if time.time() > deadline - 2.0:
            break
            
        current_scope = get_scope(current_rows)
        
        # 1. Targeted Reduction Phase
        # Find the global maximum element
        max_val = -1
        max_r_idx = -1
        max_c_idx = -1
        
        for r_idx, r in enumerate(current_rows):
            for c_idx, val in enumerate(r):
                if val > max_val:
                    max_val = val
                    max_r_idx = r_idx
                    max_c_idx = c_idx
        
        if max_c_idx == 0:
            # Should not happen as rows start at 0 and increase
            break
            
        reduced = False
        # Try to reduce the max element
        # Look for a valid value lower than current max
        # We scan downwards from max_val - 1
        # To avoid O(N^2) per step being too slow if range is huge, we can sample or limit steps.
        # But for k=7, row_len=8, the max is likely < 200. Scanning is fast.
        
        # Optimization: Only scan a certain window or use random sampling if range is large?
        # For now, scan down, but limit to 500 steps or until found.
        for new_val in range(max_val - 1, current_rows[max_r_idx][max_c_idx-1], -1):
            if is_move_valid(current_rows, max_r_idx, max_c_idx, new_val):
                current_rows[max_r_idx][max_c_idx] = new_val
                reduced = True
                break
        
        if reduced:
            new_scope = get_scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnant_count = 0
            else:
                # Even if global scope didn't drop, the distribution might be tighter
                # We accept it if it's valid (which it is)
                stagnant_count = 0
        else:
            # 2. Random Perturbation Phase (Escape Local Minima)
            # Pick a random row and position
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k) # Positions 1 to k
            
            row = current_rows[r_idx]
            old_val = row[p_idx]
            
            # Determine valid range
            min_val = row[p_idx-1] + 1
            max_val_bound = row[p_idx+1] - 1 if p_idx < k else old_val + 50
            
            # Try a few random moves
            # Bias towards smaller values to help scope reduction
            found_move = False
            attempts = 10
            for _ in range(attempts):
                if min_val > max_val_bound:
                    break
                # Sample from [min_val, max_val_bound]
                # To bias towards smaller, we can sample from a smaller range or just uniform
                new_val = random.randint(min_val, max_val_bound)
                if new_val == old_val:
                    continue
                
                if is_move_valid(current_rows, r_idx, p_idx, new_val):
                    # Accept if it improves scope or with probability T (SA style)
                    current_rows[r_idx][p_idx] = new_val
                    new_scope = get_scope(current_rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        stagnant_count = 0
                    else:
                        # Accept with probability exp(-delta/T) if it worsens scope?
                        # Or just accept if it's valid to explore?
                        # Let's use a simple heuristic: Accept if scope doesn't increase too much,
                        # or always accept to allow exploration, but keep track of best.
                        # Since we keep best_rows, we can be more aggressive.
                        # Let's accept valid moves to escape plateaus.
                        stagnant_count += 1
                    
                    found_move = True
                    break
            
            if not found_move:
                # If no random move worked, try a "kick" by moving the max element to a random valid spot
                # Or just retry targeted reduction later
                pass

        # Cooling
        T *= cooling
        
        # If we've been stagnant for too long, we might need a stronger restart or perturbation
        # For now, the random perturbation handles this.
        
    return best_rows
