import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # Lower bound on scope

    # Helper: Check if a row is valid given a set of used diffs
    def is_row_valid(row, used_diffs):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs or d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    # Fast Greedy Construction to get initial solution
    def greedy_construct(seed, max_scope):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Try to find next element
                # Bias towards smaller values to keep scope low
                candidates = range(last + 1, max_scope + 1)
                # Randomize slightly to avoid getting stuck
                if rng.random() < 0.2:
                    candidates = list(candidates)
                    rng.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Backtracking Solver for a single row
    # used_bitset: bytearray or int representing used differences
    # max_val: maximum allowed value for elements in the row
    def find_row_backtrack(used_bitset, max_val, k, node_limit=100000):
        """
        Finds a valid row of length k+1 starting with 0, max <= max_val,
        such that its differences do not conflict with used_bitset.
        Returns the row or None.
        """
        row = [0]
        nodes_visited = 0
        
        # Precompute max possible diff index
        max_diff = max_val
        
        def dfs(current_row, last_val):
            nonlocal nodes_visited
            nodes_visited += 1
            if nodes_visited > node_limit:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            # Determine range for next value
            # Must be > last_val
            # Must be <= max_val
            # Also, we need to place (k+1 - len(current_row)) - 1 more elements
            # So next_val + (remaining) <= max_val is not strictly required, 
            # but next_val + 1 <= max_val is.
            
            # Heuristic: Try values in a randomized order, but biased towards smaller values
            # to keep the row compact.
            
            # Generate candidates
            min_next = last_val + 1
            max_next = max_val
            
            # Optimization: If we have R slots left, the smallest possible sequence is 
            # last+1, last+2, ...
            # So min_next is last+1.
            
            # To avoid checking too many invalid candidates early, we can check
            # if the diff (x - a) is in used_bitset.
            
            # Randomized selection for diversity
            # But we want to find *any* valid row.
            # Let's try a few random starts and then greedy?
            # No, pure DFS with randomization.
            
            candidates = list(range(min_next, max_next + 1))
            # Shuffle with bias: smaller values are better for scope, but we are bounded by max_val.
            # Actually, for a fixed max_val, any valid row is equally good for the *current* step.
            # However, a row with smaller max might be better for *future* rows?
            # Since we are reconstructing one row at a time, and we check against fixed other rows,
            # the internal structure matters for the diffs used.
            # A "compact" row uses smaller diffs, which might be more likely to be blocked.
            # A "spread" row uses larger diffs, which might be less blocked.
            # Let's just shuffle.
            random.shuffle(candidates)
            
            for x in candidates:
                # Check conflicts
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d > max_diff:
                        continue # Should not happen if x <= max_val and a >= 0
                    # Check bitset
                    # d is index
                    if used_bitset[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check internal conflicts
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Commit
                current_row.append(x)
                for d in new_diffs:
                    used_bitset[d] = 1
                
                if dfs(current_row, x):
                    return True
                
                # Backtrack
                for d in new_diffs:
                    used_bitset[d] = 0
                current_row.pop()
            
            return False
        
        # We need a mutable bitset for the search. 
        # The input used_bitset is the state of OTHER rows.
        # We need to add the diffs of the row we are building.
        # So we pass a copy or manage it carefully.
        # Since we backtrack, we can modify a local copy.
        
        # Create a local copy of the used_bitset to allow backtracking
        local_used = bytearray(used_bitset) 
        
        if dfs(row, 0):
            return row
        return None

    # Initial Solution
    best_rows = None
    best_scope = float('inf')
    
    # Try greedy with increasing scope limits
    for scope_limit in range(lb, lb + 50):
        if time.time() > deadline - 10:
            break
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            rows = greedy_construct(seed, scope_limit)
            if rows:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    if best_scope == lb:
                        break
        if best_scope == lb:
            break
        # If greedy failed for this scope, it might still be possible with backtracking,
        # but we start with a valid solution first.
        if best_rows is not None:
            break

    if best_rows is None:
        # Fallback: very loose greedy
        rows = greedy_construct(0, 500)
        if rows:
            best_rows = rows
            best_scope = get_scope(rows)
        else:
            return None # Should not happen

    # LNS with Backtracking
    # We will try to improve the scope by reconstructing rows.
    
    # Precompute used set for best_rows
    current_used = compute_used_set(best_rows)
    
    # Convert to bitset for speed
    # Max diff is best_scope
    max_diff_size = best_scope + 1
    used_bitset = bytearray(max_diff_size)
    for d in current_used:
        if d < max_diff_size:
            used_bitset[d] = 1

    # LNS Loop
    iters = 0
    while time.time() < deadline - 2:
        iters += 1
        if iters % 100 == 0:
            # Periodic check
            pass
        
        # Pick a random row to reconstruct
        row_idx = random.randint(0, n_rows - 1)
        
        # Save the original row
        orig_row = best_rows[row_idx]
        
        # Remove this row from the used set
        # We need to clear the diffs of orig_row from used_bitset
        for i in range(len(orig_row)):
            for j in range(i + 1, len(orig_row)):
                d = orig_row[j] - orig_row[i]
                if d < max_diff_size:
                    used_bitset[d] = 0
        
        # Try to find a new row for this slot
        # Target scope: try to keep it <= best_scope, or slightly less if possible
        # We can try to find a row with max <= best_scope - 1 to improve global scope
        # But if we only reduce one row's max, the global max might not change if another row has it.
        
        # Strategy: 
        # 1. Try to find a row with max <= best_scope - 1. If found, and if this row was the unique max, 
        #    the global scope drops.
        # 2. If not, try to find any row with max <= best_scope.
        
        # Let's try to find a row with max <= best_scope - 1 first (if best_scope > lb)
        target_scope = best_scope - 1
        if target_scope < lb:
            target_scope = lb
            
        new_row = None
        
        # Use backtracking to find a row
        # To speed up, we can limit the search space.
        # If we fail, we revert.
        
        # We need to be careful: if we change the row, the used_bitset changes.
        # The backtracking function modifies a local copy, so it's safe.
        
        # To improve diversity, we can randomize the backtracking order (done inside).
        
        new_row = find_row_backtrack(used_bitset, target_scope, k, node_limit=50000)
        
        if new_row:
            # Check if this improves the global scope
            new_max = max(new_row)
            
            # Calculate new global scope
            # We need to check if any other row has max > new_max?
            # Actually, the global scope is max(max(row) for row in rows).
            # If we replace row_idx with new_row, the new global scope is:
            # max( max(row) for row in best_rows if row is not best_rows[row_idx] ) vs new_max
            
            other_maxes = [max(r) for i, r in enumerate(best_rows) if i != row_idx]
            current_other_max = max(other_maxes) if other_maxes else 0
            new_global_scope = max(current_other_max, new_max)
            
            if new_global_scope < best_scope:
                # Update
                best_rows[row_idx] = new_row
                best_scope = new_global_scope
                # Update used_bitset to reflect the new row
                # We already cleared the old row's diffs.
                # Now add the new row's diffs.
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d < max_diff_size:
                            used_bitset[d] = 1
                            
                # If scope dropped, we might be able to shrink the bitset?
                # Not necessary, just keep it large enough.
                
            else:
                # Even if global scope didn't drop, maybe the new row is "better" 
                # (e.g., uses different diffs) allowing future improvements?
                # For now, only accept if scope improves to avoid drifting.
                # However, accepting equivalent scope solutions can help escape local minima.
                # Let's accept with some probability if scope is same, to explore.
                if new_global_scope == best_scope and random.random() < 0.1:
                    best_rows[row_idx] = new_row
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d < max_diff_size:
                                used_bitset[d] = 1
                else:
                    # Revert: re-add original row's diffs
                    for i in range(len(orig_row)):
                        for j in range(i + 1, len(orig_row)):
                            d = orig_row[j] - orig_row[i]
                            if d < max_diff_size:
                                used_bitset[d] = 1
        else:
            # Revert
            for i in range(len(orig_row)):
                for j in range(i + 1, len(orig_row)):
                    d = orig_row[j] - orig_row[i]
                    if d < max_diff_size:
                        used_bitset[d] = 1

    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
