<<<<<<< SEARCH
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
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
    
    # Calculate initial max
    current_max = max(max(row) for row in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    best_used = set(used) # Not strictly needed if we recompute or maintain carefully, but for SA we maintain state
    
    # State for SA
    # We maintain `used` as a set of integers.
    # We also need to be able to undo moves or recalculate.
    # Given k is small, recalculating diffs for a row is fast.
    # But checking global conflict is O(1) if we have the set.
    
    # Let's define a function to get the cost
    # Cost = max_value * 1000 + num_conflicts
    # Actually, we want valid solutions first.
    # If we restrict moves to only valid ones, we can't escape local minima easily.
    # So we allow invalid moves with a penalty.
    
    def get_cost(rows, used_diffs):
        max_val = max(max(r) for r in rows)
        # Count conflicts? 
        # It's easier to just check validity in the move generation if we maintain the set correctly.
        # But for SA, we need to accept invalid states.
        # So we need a way to count conflicts.
        # Re-counting conflicts is O(n_rows * k^2). 7 * 36 = 252. This is fine.
        
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        
        # Count duplicates
        from collections import Counter
        c = Counter(all_diffs)
        conflicts = sum(count - 1 for count in c.values() if count > 1)
        
        return max_val * 1000 + conflicts

    current_rows = [r[:] for r in rows]
    current_max = best_max
    current_conflicts = 0 # Greedy is valid
    
    # Precompute indices for moves
    # Move: pick row r, index i (1..k), new value v
    # v must be > row[i-1] and < row[i+1] (if i < k)
    # If i == k, v can be anything > row[k-1] and < current_max + some bound?
    # To reduce max, we should try v < current_max.
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)
        
        row = current_rows[r_idx]
        old_val = row[i_idx]
        
        # Determine range for new value
        low = row[i_idx - 1] + 1
        high = row[i_idx + 1] - 1 if i_idx < k else current_max + 10 # Allow expansion? No, we want to shrink.
        # Let's bias towards smaller values to reduce max.
        # If we want to reduce max, we should only try values < current_max.
        # But sometimes we need to move things around to allow reduction.
        # Let's try a range [low, min(high, current_max)]
        
        if low > high:
            T *= cooling
            continue
            
        # Bias: 80% chance to pick a value < current_max if possible
        # If i_idx == k, high is effectively infinity, but we cap at current_max to try to shrink.
        # If i_idx < k, high is fixed by next element.
        
        if i_idx == k:
            # Try to move the largest element down
            if low <= current_max:
                new_val = random.randint(low, current_max)
            else:
                new_val = random.randint(low, low + 5) # Fallback
        else:
            new_val = random.randint(low, high)
            
        if new_val == old_val:
            continue
            
        # Perform move
        row[i_idx] = new_val
        
        # Calculate new cost
        # We need to check if the row is still strictly increasing.
        # low/high logic ensures row[i-1] < new_val < row[i+1].
        
        new_cost = get_cost(current_rows, None) # We pass None to ignore used_diffs in get_cost, it recalcs
        old_cost = current_max * 1000 + current_conflicts
        
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            current_max = max(current_max, new_val) # Actually max of all rows
            # Recalc max properly
            current_max = max(max(r) for r in current_rows)
            
            # Recalc conflicts for state tracking? 
            # get_cost already calculated it. Let's just store the cost components or re-calc.
            # For simplicity, let's just track best valid solution.
            # We don't strictly need current_conflicts for acceptance if we use get_cost, 
            # but we need it for old_cost. 
            # Let's just re-calc old_cost every time? No, that's slow.
            # Let's maintain current_conflicts.
            
            # To maintain current_conflicts efficiently is hard with this simple get_cost.
            # Let's just use get_cost for both old and new? 
            # get_cost is ~250 ops. 100k steps * 500 ops = 50M ops. Python might be slow.
            # 120s is plenty for 100k-1M steps in Python if optimized.
            
            # Let's optimize get_cost or just use it.
            
            # Update state
            # We don't maintain 'used' set in this loop version, we just check validity via get_cost.
            # This is O(N^2) per step.
            
            # Check if valid and better
            if new_cost < 1000: # No conflicts
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
            
        else:
            # Reject move, revert
            row[i_idx] = old_val
            
        T *= cooling
        
    # Return best valid solution
    return best_rows

# The above draft has a flaw: get_cost is called twice per iteration (once for new, once for old if we don't store it).
# And get_cost is O(N^2).
# Let's refine the cost calculation to be faster or maintain the state.
# Actually, for N=7, K=5, N*K=35 elements. Pairs ~ 7*15 = 105 diffs.
# 105 diffs is very small. Counter on 105 items is fast.
# So get_cost is very fast.

# Let's rewrite cleanly.

=======
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # k is 5, n_rows is 7.
    # Each row has k+1 = 6 elements.
    
    def greedy_init():
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
        return rows

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Initial solution
    best_rows = greedy_init()
    best_scope = get_scope(best_rows)
    
    # Iterated Local Search
    # We run a local search (SA) for a fixed time chunk, then perturb, then repeat.
    # Chunk size: 10 seconds? Or dynamic?
    # Let's use a chunk of 5 seconds for ILS to allow many restarts.
    
    chunk_time = 5.0
    T_start = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        # Run local search from current best
        current_rows = [r[:] for r in best_rows]
        
        # Initialize state for local search
        used_diffs = set()
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        current_scope = get_scope(current_rows)
        best_local_scope = current_scope
        best_local_rows = [r[:] for r in current_rows]
        
        T = T_start
        
        # Local Search Loop
        while time.time() < start_time + (time.time() - start_time) + chunk_time and time.time() < deadline and T > min_T:
            # Actually, simpler:
            # local_deadline = min(time.time() + chunk_time, deadline)
            # while time.time() < local_deadline and T > min_T:
            
            # Let's just use a simple time check
            pass # Placeholder for loop structure
            
        # Corrected Local Search Loop
        local_deadline = min(time.time() + chunk_time, deadline)
        T = T_start
        while time.time() < local_deadline and T > min_T:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, k)
            
            row = current_rows[r_idx]
            old_val = row[i_idx]
            
            low = row[i_idx - 1] + 1
            high = row[i_idx + 1] - 1 if i_idx < k else current_scope + 5
            
            if low > high:
                T *= cooling
                continue
            
            if i_idx == k:
                max_target = min(high, current_scope)
                if low <= max_target and random.random() < 0.8:
                    new_val = random.randint(low, max_target)
                else:
                    new_val = random.randint(low, high)
            else:
                new_val = random.randint(low, high)
                
            if new_val == old_val:
                T *= cooling
                continue
            
            new_row = row[:]
            new_row[i_idx] = new_val
            
            # Check strictly increasing
            if any(new_row[j] <= new_row[j-1] for j in range(1, len(new_row))):
                T *= cooling
                continue
                
            new_row_diffs = set()
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    new_row_diffs.add(new_row[j] - new_row[i])
            
            old_row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    old_row_diffs.add(row[j] - row[i])
            
            # Remove old diffs
            used_diffs.difference_update(old_row_diffs)
            
            conflicts = 0
            for d in new_row_diffs:
                if d in used_diffs:
                    conflicts += 1
            
            if conflicts == 0:
                used_diffs.update(new_row_diffs)
                current_rows[r_idx] = new_row
                new_scope = get_scope(current_rows)
                
                if new_scope < current_scope:
                    current_scope = new_scope
                    if new_scope < best_local_scope:
                        best_local_scope = new_scope
                        best_local_rows = [r[:] for r in current_rows]
                else:
                    delta = new_scope - current_scope
                    if delta == 0 or random.random() < math.exp(-delta / T):
                        current_scope = new_scope
                    else:
                        # Revert
                        current_rows[r_idx] = row[:]
                        used_diffs.difference_update(new_row_diffs)
                        used_diffs.update(old_row_diffs)
            else:
                # Revert
                used_diffs.update(old_row_diffs)
                
            T *= cooling
        
        # End of local search chunk
        if best_local_scope < best_scope:
            best_scope = best_local_scope
            best_rows = best_local_rows
            
        # Perturb best_rows for next ILS iteration
        # Change 1-2 random values in a random row
        if time.time() < deadline - 2.0: # Leave time for final check
            p_rows = [r[:] for r in best_rows]
            for _ in range(random.randint(1, 2)):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, k)
                row = p_rows[r_idx]
                low = row[i_idx - 1] + 1
                high = row[i_idx + 1] - 1 if i_idx < k else best_scope + 10
                if low <= high:
                    p_rows[r_idx][i_idx] = random.randint(low, high)
            # We don't immediately update best_rows to p_rows, we use p_rows as start for next local search?
            # My loop structure restarts from `best_rows`.
            # To use perturbation, I should set `best_rows` to a perturbed version?
            # No, `best_rows` is the BEST found so far. We should search from there.
            # But ILS usually searches from a perturbed version of the best.
            # Let's modify the start of the next loop to use a perturbed version of best_rows.
            
            # I will restructure the loop to store `current_search_start`.
            pass

    return best_rows

>>>>>>> REPLACE

The logic above is a bit tangled. I will write a clean version.

