import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Greedy Initialization
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Precompute row diffs for quick access? 
    # We will maintain `used` set dynamically.
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    def get_max(rows):
        return max(max(r) for r in rows)
        
    # Helper to get diffs for a specific element in a row
    # row is sorted
    # returns list of diffs involving index j
    def get_element_diffs(row, j):
        x = row[j]
        diffs = []
        for i in range(j):
            diffs.append(x - row[i])
        for i in range(j+1, len(row)):
            diffs.append(row[i] - x)
        return diffs

    def is_valid_move(row, j, new_val, used_set):
        # Check order
        if j > 0 and new_val <= row[j-1]:
            return False
        if j < len(row) - 1 and new_val >= row[j+1]:
            return False
            
        # Check diffs
        # Remove old diffs from a temporary view? 
        # Better: calculate new diffs and check intersection with used_set 
        # BUT used_set contains the old diffs of this element.
        # So we must exclude the old diffs of this element from the check.
        
        old_diffs = set(get_element_diffs(row, j))
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - row[i])
        for i in range(j+1, len(row)):
            new_diffs.add(row[i] - new_val)
            
        # The conflict is if new_diffs intersects with (used_set - old_diffs)
        # Also new_diffs must not contain 0 (guaranteed by strict inequality)
        
        # Optimization: 
        # if any d in new_diffs is in used_set and d not in old_diffs -> invalid
        # Note: d can be in both old and new if the relative distance to some other element didn't change? 
        # e.g. x=10, a=0, d=10. x'=11, a=0, d=11. Different.
        # If x=10, a=2, d=8. x'=12, a=2, d=10.
        # It's possible for a diff to be in both? 
        # x - a = x' - b? No, we are changing x to x'. The diffs are x-a and x'-a.
        # They are different unless a changes, which it doesn't.
        # So new_diffs and old_diffs are disjoint?
        # x - a = x' - a => x = x'. So yes, disjoint if x != x'.
        
        # So we just check: is there any d in new_diffs such that d in used_set?
        # Wait, used_set contains old_diffs.
        # So we must check: (new_diffs & used_set) == set() is NOT sufficient if new_diffs overlaps with old_diffs?
        # We established they are disjoint.
        # So if d in new_diffs and d in used_set, it must be from another element.
        # So simply: if new_diffs & used_set: return False
        
        if new_diffs & used_set:
            return False
            
        return True

    step = 0
    while time.time() < deadline:
        if T < min_T:
            T = 1.0 # Restart or stop? Let's just keep going with low T, basically hill climbing
            # Or break if we want pure local search at end
            
        # Pick a random row and element
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        j_idx = random.randint(1, k) # 1 to k, index 0 is fixed at 0
        
        old_val = row[j_idx]
        
        # Propose new value
        # Bias towards smaller values to reduce max?
        # Or random walk.
        # Let's try small random deltas
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure new_val is not 0 (since row[0]=0 and increasing)
        if new_val <= 0:
            continue
            
        if is_valid_move(row, j_idx, new_val, used):
            # Apply move
            old_diffs = set(get_element_diffs(row, j_idx))
            used -= old_diffs
            
            row[j_idx] = new_val
            
            new_diffs = set(get_element_diffs(row, j_idx))
            used |= new_diffs
            
            new_max = get_max(current_rows)
            
            if new_max < current_max:
                # Accept
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
            else:
                # Reject or Accept with prob
                # Delta cost is new_max - current_max (which is >= 0)
                # If new_max == current_max, we might accept to explore
                if new_max == current_max:
                    # Accept with high prob to allow moves that keep max same
                    if random.random() < 0.5: 
                        pass # Accept
                    else:
                        # Undo
                        used -= new_diffs
                        row[j_idx] = old_val
                        used |= old_diffs
                        continue
                else:
                    # new_max > current_max
                    # Accept with prob exp(-delta / T)
                    delta_cost = new_max - current_max
                    prob = math.exp(-delta_cost / T)
                    if random.random() < prob:
                        pass # Accept
                    else:
                        # Undo
                        used -= new_diffs
                        row[j_idx] = old_val
                        used |= old_diffs
                        continue
                        
        T *= cooling
        step += 1
        
    return best_rows
