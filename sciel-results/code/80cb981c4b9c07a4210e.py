import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0,
    such that all pairwise differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on max entry (counting bound: we need total_diffs distinct positive integers)
    # The largest difference is at least total_diffs.
    # Actually, the differences are distinct positive integers. The set of all differences has size total_diffs.
    # The maximum difference is at least total_diffs.
    # Since max diff <= max_entry, max_entry >= total_diffs.
    lower_bound = total_diffs
    
    best_result = None
    best_max = float('inf')
    
    rng = random.Random(42)
    
    def check_valid(rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_construct(seed_offset=0):
        """Greedy construction to get an initial valid solution."""
        r = random.Random(seed_offset)
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

    def compute_diffs_except(rows, skip_idx):
        """Compute the set of all differences in rows, excluding the row at skip_idx."""
        used = set()
        for idx, row in enumerate(rows):
            if idx == skip_idx:
                continue
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def backtrack_row(used_other, max_scope, rng):
        """
        Attempt to construct a single row of length k+1 starting with 0,
        such that:
        1. It is strictly increasing.
        2. Max entry <= max_scope.
        3. No difference in this row is in used_other.
        4. All differences in this row are distinct (implied by 3 if we check locally too, 
           but we must ensure internal distinctness as well).
        
        Returns a list if successful, None otherwise.
        """
        row = [0]
        
        # To speed up, we can pre-compute valid candidates or just use backtracking with pruning.
        # Given max_scope is usually < 200, and k=7, depth is 8.
        
        def bt(pos, last_val):
            if pos == k + 1:
                return True
            
            # Heuristic: try smaller values first to keep scope low, 
            # but randomize slightly to avoid getting stuck in the same trap
            start_val = last_val + 1
            if pos == 0:
                start_val = 0 # Actually row[0] is 0, so we start filling pos 1
            
            # Determine candidate range
            # We want to find a value x such that {x - a for a in row} are all new and not in used_other
            # And x <= max_scope
            
            # Optimization: Pre-compute conflicts for a range? 
            # For k=7, row size is small.
            
            # Randomized selection of candidates to escape local minima in backtracking
            candidates = []
            # We can look ahead a bit. 
            # Let's just iterate from start_val to max_scope, but shuffle the order slightly?
            # No, standard backtracking with MRV (Minimum Remaining Values) is hard here.
            # Let's try values in a randomized order within a window, or just sequential with random restarts.
            
            # Simple approach: iterate sequentially. If we fail, we backtrack.
            # To improve diversity, we can shuffle the candidate list.
            # However, generating the list up to max_scope (e.g., 150) every step is cheap.
            
            # Let's use a hybrid: try small values first, but with some randomization.
            # Actually, for finding *any* valid row within scope, sequential is often fast if a solution exists.
            # The bottleneck is if no solution exists, we explore the whole tree.
            # To mitigate this, we can add a "random jump" in the candidate selection.
            
            # Let's stick to a simple backtracking but with a time limit per call?
            # No, the caller (SA) will manage time.
            
            # To speed up, we can check if the remaining capacity is sufficient.
            # Remaining elements: rem = k + 1 - pos
            # Min possible next value: last_val + 1
            # Max possible last value: max_scope
            # We need to fit 'rem' elements in [last_val+1, max_scope].
            # This requires max_scope - last_val >= rem.
            if max_scope - last_val < (k + 1 - pos):
                return False

            # Generate candidates
            # To improve search, we can try candidates in a random order, 
            # but biased towards smaller values to keep the row "tight".
            # Let's generate a list of valid candidates first? That might be expensive.
            # Let's just iterate.
            
            # To avoid deterministic traps, we can shuffle the range if it's small, 
            # or just rely on the SA loop calling this with different 'used_other' sets.
            
            for val in range(start_val, max_scope + 1):
                # Check conflicts with current row
                valid = True
                new_diffs = set()
                for a in row:
                    d = val - a
                    if d <= 0:
                        continue
                    if d in used_other:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Forward checking: 
                # If we place val, can we still complete the row?
                # This is hard to check strictly. 
                
                row.append(val)
                # Note: We don't update 'used_other' because this is local to the row construction.
                # The 'used_other' is constant for this call.
                
                if bt(pos + 1, val):
                    return True
                
                row.pop()
            
            return False
        
        # Start backtracking
        # We need to handle pos=0 separately since row[0] is always 0.
        if bt(1, 0):
            return row[:]
        return None

    def local_search_step(rows, max_scope_target, rng):
        """
        Perform a single step of local search / simulated annealing.
        Returns new_rows if a move was accepted, else None.
        """
        current_scope = get_scope(rows)
        
        # Decide move type
        # 1. Rebuild a row (Structural)
        # 2. Perturb an element (Local)
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Structural Move: Rebuild one row
            idx = rng.randint(0, n_rows - 1)
            
            # Calculate constraints from other rows
            used_other = compute_diffs_except(rows, idx)
            
            # Try to find a new row for idx with scope <= current_scope (or slightly less)
            # To allow escape, we might try up to current_scope + small_delta
            # But we want to minimize scope. 
            # Let's try to find a row with max entry <= current_scope.
            # If found, and its max entry is smaller than the current row's max, it's an improvement.
            # If found with same max, it's neutral (might help future moves).
            
            # To be more aggressive in minimizing, we can try to find a row with max entry <= best_max_found_so_far
            # But we pass current_scope as the limit to ensure validity relative to the current state.
            
            new_row = backtrack_row(used_other, current_scope, rng)
            
            if new_row is not None:
                new_rows = [r[:] for r in rows]
                new_rows[idx] = new_row
                
                # Check if this move improves the global scope
                new_scope = get_scope(new_rows)
                
                # Acceptance criteria:
                # 1. If new_scope < current_scope, accept.
                # 2. If new_scope == current_scope, accept with some probability (to explore).
                # 3. If new_scope > current_scope (shouldn't happen if limit is current_scope, 
                #    unless the new row has max > current_scope? No, limit is current_scope).
                
                if new_scope < current_scope:
                    return new_rows
                elif new_scope == current_scope:
                    # Accept with probability 0.5 to allow exploration
                    if rng.random() < 0.5:
                        return new_rows
                    else:
                        return None
                else:
                    return None
            else:
                # Couldn't rebuild within current scope.
                # Try rebuilding with a slightly larger scope to escape?
                # This would increase the global scope, so we only accept if we later reduce it.
                # For simplicity in this step, if we can't fit in current scope, skip.
                # Or, try to fit in current_scope + 5, and accept only if it leads to a dead end escape?
                # Let's stick to strict improvement or neutral for now to ensure monotonic decrease in scope 
                # when possible.
                return None
                
        else:
            # Local Move: Perturb one element
            idx = rng.randint(0, n_rows - 1)
            row = rows[idx]
            pos = rng.randint(1, len(row) - 1) # Don't touch 0
            
            old_val = row[pos]
            
            # Try a new value
            # Range: (row[pos-1], row[pos+1]) if pos+1 exists, else (row[pos-1], max_entry)
            lower = row[pos-1] + 1
            upper = row[pos+1] - 1 if pos + 1 < len(row) else get_scope(rows)
            
            if lower > upper:
                return None
                
            new_val = rng.randint(lower, upper)
            if new_val == old_val:
                return None
            
            # Check validity
            # Remove old diffs
            old_diffs = set()
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(row[i] - old_val)
                if d > 0:
                    old_diffs.add(d)
            
            # Check new diffs
            new_diffs = set()
            conflict = False
            # We need to check against ALL other diffs in the system except the old ones from this row/pos
            # This is expensive. 
            # Optimization: Compute global used set once? 
            # For now, let's just check against the specific row and other rows.
            
            # Actually, a full validity check is O(N^2). 
            # Let's build the set of all diffs in the system excluding the current row's diffs involving pos.
            # This is still O(N^2).
            # Given N is small (4 rows, 8 entries), N^2 is 64. It's fast.
            
            global_used = set()
            for r_idx, r in enumerate(rows):
                if r_idx == idx:
                    # Skip diffs involving pos in this row
                    for i in range(len(r)):
                        if i == pos:
                            continue
                        for j in range(i + 1, len(r)):
                            if j == pos:
                                continue
                            global_used.add(r[j] - r[i])
                else:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            global_used.add(r[j] - r[i])
            
            # Now check new_val
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(row[i] - new_val)
                if d == 0:
                    conflict = True
                    break
                if d in global_used:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                return None
            
            # Apply move
            new_rows = [r[:] for r in rows]
            new_rows[idx][pos] = new_val
            new_scope = get_scope(new_rows)
            current_scope = get_scope(rows)
            
            if new_scope < current_scope:
                return new_rows
            elif new_scope == current_scope:
                if rng.random() < 0.3:
                    return new_rows
            else:
                # Only accept if we are in a "high temperature" phase? 
                # We don't have explicit temperature here, but we can accept worse moves
                # if they are slightly worse, to allow escape.
                # Let's accept if new_scope <= current_scope + 2
                if new_scope <= current_scope + 2 and rng.random() < 0.1:
                    return new_rows
            
            return None

    # Phase 1: Initial Solution
    initial_rows = greedy_construct(0)
    if initial_rows and check_valid(initial_rows):
        best_result = initial_rows
        best_max = get_scope(initial_rows)
    else:
        # Fallback if greedy fails (unlikely)
        best_result = None
        best_max = float('inf')

    # Phase 2: Simulated Annealing / Local Search
    if best_result:
        current_rows = [r[:] for r in best_result]
        current_max = best_max
        
        # We will run until deadline
        # To allow escaping local minima, we can occasionally allow "worse" solutions
        # But the primary driver is the row-rebuild move which strictly tries to fit in current scope.
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            
            # Perform a move
            new_rows = local_search_step(current_rows, current_max, rng)
            
            if new_rows is not None:
                new_max = get_scope(new_rows)
                
                if new_max < best_max:
                    best_max = new_max
                    best_result = [r[:] for r in new_rows]
                    current_rows = new_rows
                    current_max = new_max
                    no_improve_count = 0
                elif new_max == best_max:
                    # Accept neutral moves to explore
                    current_rows = new_rows
                    current_max = new_max
                    no_improve_count += 1
                else:
                    # Worse move (accepted by local_search_step)
                    current_rows = new_rows
                    current_max = new_max
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            # If we are stuck, we can try a "kick"
            # A kick involves randomly rebuilding a row with a larger scope, 
            # then trying to reduce it.
            if no_improve_count % 1000 == 0 and no_improve_count > 0:
                # Kick: Rebuild a random row with a larger scope
                idx = rng.randint(0, n_rows - 1)
                used_other = compute_diffs_except(current_rows, idx)
                # Try to find a row with scope up to current_max + 20
                kick_row = backtrack_row(used_other, current_max + 20, rng)
                if kick_row is not None:
                    current_rows = [r[:] for r in current_rows]
                    current_rows[idx] = kick_row
                    current_max = get_scope(current_rows)
                    # Reset no_improve to give it a chance
                    no_improve_count = 0
                    # Note: This might have increased current_max. 
                    # The next steps will try to reduce it.

    if best_result is None:
        # Final fallback
        best_result = greedy_construct(0)
    
    return best_result

if __name__ == "__main__":
    # Test with the specific task parameters
    rows = construct(4, 7)
    print("Result:")
    for r in rows:
        print(r)
    print("Scope:", max(max(r) for r in rows))
