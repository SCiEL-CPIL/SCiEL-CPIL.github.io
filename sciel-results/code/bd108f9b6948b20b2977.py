import random
import math
import time

def construct(n_rows, k):
    # 1. Initial Solution: Greedy
    used = {} # diff -> count
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = [x - a for a in row]
            # Check if any new diff is already used
            conflict = False
            for d in new_diffs:
                if used.get(d, 0) > 0:
                    conflict = True
                    break
            if not conflict:
                for d in new_diffs:
                    used[d] = used.get(d, 0) + 1
                row.append(x)
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # State for incremental cost
    # current_conflicts: number of pairs of differences that are equal
    # used: dict of diff -> count
    # To calculate conflicts from used: sum(count * (count-1) / 2) for each diff
    # But we want to update incrementally.
    
    def update_diffs(row_idx, new_row):
        # Remove old diffs of row_idx
        old_row = rows[row_idx]
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                d = old_row[j] - old_row[i]
                used[d] -= 1
                if used[d] <= 0:
                    del used[d]
        
        # Add new diffs of new_row
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                d = new_row[j] - new_row[i]
                used[d] = used.get(d, 0) + 1
        
        rows[row_idx] = new_row

    def get_conflicts():
        c = 0
        for count in used.values():
            if count > 1:
                c += count * (count - 1) // 2
        return c

    def get_max_val():
        return max(max(r) for r in rows)

    current_conflicts = get_conflicts()
    current_max = get_max_val()
    
    # SA Parameters
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    start_time = time.time()
    time_limit = 110
    
    # Helper to calculate cost for SA acceptance
    # We want to minimize MaxVal, but we need to be valid (Conflicts=0).
    # Cost = MaxVal if Conflicts==0 else MaxVal + Conflicts * Large_Penalty
    
    def calc_cost():
        if current_conflicts == 0:
            return current_max
        else:
            return current_max + current_conflicts * 10000

    current_cost = calc_cost()
    
    while T > min_T and (time.time() - start_time) < time_limit:
        r_idx = random.randint(0, n_rows - 1)
        
        # Generate a new candidate row for r_idx
        # Strategy: Perturb the existing row
        old_row = rows[r_idx][:]
        new_row = old_row[:]
        
        # Randomly change one or more elements
        # Let's try changing one random element (index 1 to k)
        if k > 0:
            j_idx = random.randint(1, k)
            # Choose a new value
            # Range: (old_row[j_idx-1], old_row[j_idx+1]) or similar
            lower = old_row[j_idx-1] + 1
            upper = old_row[j_idx+1] - 1 if j_idx < k else old_row[j_idx] + 50
            
            if lower < upper:
                new_val = random.randint(lower, upper)
                new_row[j_idx] = new_val
            else:
                # If range is too small, try shifting the suffix
                delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                # Shift all elements from j_idx to end
                for j in range(j_idx, k+1):
                    new_row[j] = old_row[j] + delta
                # Check validity of new_row (strictly increasing)
                valid = True
                for j in range(1, len(new_row)):
                    if new_row[j] <= new_row[j-1]:
                        valid = False
                        break
                if not valid:
                    continue # Skip this move if it breaks monotonicity
                
        # If k==0, row is always [0], no move possible
        if k == 0:
            break

        # Apply move temporarily to calculate new cost
        # We need to update `used` and `current_conflicts`
        
        # 1. Remove old diffs
        old_diffs = []
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                d = old_row[j] - old_row[i]
                old_diffs.append(d)
                
        # 2. Add new diffs
        new_diffs = []
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                d = new_row[j] - new_row[i]
                new_diffs.append(d)
        
        # Calculate change in conflicts
        # This is tricky to do incrementally without touching the global `used` dict 
        # unless we actually apply the change.
        
        # Let's apply the change to `used` and `rows`, calculate cost, then revert if rejected.
        
        # Remove old diffs from `used`
        for d in old_diffs:
            used[d] -= 1
            if used[d] == 0:
                del used[d]
                
        # Add new diffs to `used`
        for d in new_diffs:
            used[d] = used.get(d, 0) + 1
            
        # Update rows
        rows[r_idx] = new_row
        
        # Calculate new conflicts and max
        new_conflicts = get_conflicts()
        new_max = get_max_val()
        new_cost = new_max + new_conflicts * 10000 if new_conflicts > 0 else new_max
        
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        elif T > 0:
            # Avoid overflow in exp
            if delta_cost / T < 500:
                if random.random() < math.exp(-delta_cost / T):
                    accept = True
            
        if accept:
            current_conflicts = new_conflicts
            current_max = new_max
            current_cost = new_cost
            
            if new_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            # Remove new diffs
            for d in new_diffs:
                used[d] -= 1
                if used[d] == 0:
                    del used[d]
                    
            # Add old diffs
            for d in old_diffs:
                used[d] = used.get(d, 0) + 1
                
            # Revert rows
            rows[r_idx] = old_row

        T *= cooling

    return best_rows
