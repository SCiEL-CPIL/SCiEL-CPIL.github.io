import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All differences between entries in the same row must be distinct over all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Precompute all possible pairs (i, j) with i < j for k+1 elements
    # k=5 means 6 elements, 15 pairs
    pairs = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]
    num_pairs = len(pairs)
    
    def get_row_diffs(row):
        """Return a frozenset of differences for a row."""
        return frozenset(row[j] - row[i] for i, j in pairs)
    
    def get_scope(rows):
        return max(max(r) for r in rows) if rows else 0
    
    def is_valid_rows(rows):
        """Check if all rows are valid and differences are distinct across rows."""
        used = set()
        for row in rows:
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            if row[0] != 0:
                return False
            diffs = get_row_diffs(row)
            if diffs & used:
                return False
            used |= diffs
        return True
    
    def greedy_construct(seed, max_scope_hint=None):
        """
        Greedy construction with a specific seed.
        Tries to build rows one by one, choosing the smallest possible values.
        """
        random.seed(seed)
        rows = []
        used = set()
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if max_scope_hint is not None and x > max_scope_hint:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used
    
    def local_search(rows, used, time_budget):
        """
        Local search to minimize the scope.
        Tries to reduce values of elements, especially the maximums.
        """
        rows = [r.copy() for r in rows]
        # Recalculate used set to be safe
        used = set()
        for r in rows:
            used |= get_row_diffs(r)
        
        current_scope = get_scope(rows)
        best_scope = current_scope
        best_rows = [r.copy() for r in rows]
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() - start_time < time_budget:
            # Select a candidate to move
            # Strategy: Pick the element with the largest value, or a random non-zero element
            if random.random() < 0.7:
                # Find global max
                max_val = -1
                max_r = 0
                max_j = 0
                for r_idx, row in enumerate(rows):
                    for j_idx, val in enumerate(row):
                        if val > max_val:
                            max_val = val
                            max_r = r_idx
                            max_j = j_idx
                r_idx, j_idx = max_r, max_j
            else:
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
            
            if j_idx == 0:
                continue
                
            old_val = rows[r_idx][j_idx]
            new_val = old_val - 1
            
            if new_val < 0:
                continue
            
            # Check validity of move
            # We need to ensure:
            # 1. rows[r_idx] remains strictly increasing
            # 2. No difference conflict with other rows
            
            # Check strict increase
            valid = True
            if j_idx > 0 and new_val <= rows[r_idx][j_idx - 1]:
                valid = False
            if valid and j_idx < k and new_val >= rows[r_idx][j_idx + 1]:
                valid = False
            
            if valid:
                # Calculate new diffs for this row
                temp_row = rows[r_idx].copy()
                temp_row[j_idx] = new_val
                new_diffs = get_row_diffs(temp_row)
                
                # Calculate old diffs for this row
                old_diffs = get_row_diffs(rows[r_idx])
                
                # The set of diffs used by OTHER rows is used - old_diffs
                # (Since old_diffs are only in this row, assuming valid state)
                other_diffs = used - old_diffs
                
                # Check if any new diff conflicts with other_diffs
                # Note: new_diffs might contain diffs that were in old_diffs, those are fine.
                # We only care if new_diffs contains something in other_diffs.
                if new_diffs & other_diffs:
                    valid = False
            
            if valid:
                # Apply move
                rows[r_idx][j_idx] = new_val
                old_diffs = get_row_diffs(rows[r_idx]) # This is now new_diffs effectively, but let's be precise
                # Actually, we should update used set properly.
                # used = (used - old_diffs_before_move) | new_diffs
                # We have old_diffs from before the change? No, we calculated it above but didn't store it cleanly.
                # Let's re-calculate old_diffs before applying? No, we already changed the row in temp.
                # Let's do it cleanly:
                # 1. Remove old_diffs from used
                # 2. Update row
                # 3. Add new_diffs to used
                
                # We need old_diffs. Let's recompute from original row state?
                # It's faster to just recompute the whole used set if it's small, but n_rows=7, k=5, used size ~100.
                # Recomputing used set is O(n_rows * k^2) = 7 * 15 = 105 ops. Very fast.
                # Let's just recompute used set to avoid bugs.
                
                # But wait, we already modified rows[r_idx][j_idx] in the check? No, we used temp_row.
                # So rows is still in old state.
                
                # Update rows
                rows[r_idx][j_idx] = new_val
                
                # Recompute used
                used = set()
                for r in rows:
                    used |= get_row_diffs(r)
                
                new_scope = get_scope(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    no_improve = 0
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r.copy() for r in rows]
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            if no_improve > max_no_improve:
                break
                
        return best_rows, best_scope

    # Phase 1: Generate initial solutions
    # The counting bound is 105. Best known is 112.
    # Greedy might produce scopes around 120-150.
    
    initial_candidates = []
    for seed in range(50):
        rows, used = greedy_construct(seed)
        if rows is not None:
            scope = get_scope(rows)
            initial_candidates.append((scope, rows, used))
            
    if not initial_candidates:
        # Fallback: return a trivial (but likely large) solution
        # Just return zeros? No, must be increasing.
        # Return a simple arithmetic progression?
        # a_i = i * 1000
        rows = [[i * 1000 for i in range(k + 1)] for _ in range(n_rows)]
        return rows

    # Sort by scope
    initial_candidates.sort(key=lambda x: x[0])
    
    best_rows = initial_candidates[0][1]
    best_scope = initial_candidates[0][0]
    
    # Phase 2: Local Search
    # Run local search on the top few candidates
    # Budget allocation:
    # - 50% time for the best candidate
    # - 25% for the second best
    # - 25% for the third best (if time permits)
    
    time_spent = time.time() - start_time
    remaining = time_limit - time_spent
    
    # Run on best
    if remaining > 5:
        budget1 = remaining * 0.5
        rows_ls, scope_ls = local_search(initial_candidates[0][1], initial_candidates[0][2], start_time + budget1)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls
            
    # Run on second best if distinct and time remains
    time_spent = time.time() - start_time
    remaining = time_limit - time_spent
    if remaining > 5 and len(initial_candidates) > 1:
        # Check if distinct
        if initial_candidates[1][0] != initial_candidates[0][0]:
            budget2 = remaining * 0.3
            rows_ls, scope_ls = local_search(initial_candidates[1][1], initial_candidates[1][2], start_time + budget2)
            if scope_ls < best_scope:
                best_scope = scope_ls
                best_rows = rows_ls
                
    # Run on third best if distinct and time remains
    time_spent = time.time() - start_time
    remaining = time_limit - time_spent
    if remaining > 5 and len(initial_candidates) > 2:
        if initial_candidates[2][0] != initial_candidates[0][0] and initial_candidates[2][0] != initial_candidates[1][0]:
            budget3 = remaining * 0.3
            rows_ls, scope_ls = local_search(initial_candidates[2][1], initial_candidates[2][2], start_time + budget3)
            if scope_ls < best_scope:
                best_scope = scope_ls
                best_rows = rows_ls
                
    # Final check
    if not is_valid_rows(best_rows):
        # Fallback to first valid candidate
        for scope, rows, used in initial_candidates:
            if is_valid_rows(rows):
                return rows
        return [[i * 1000 for i in range(k + 1)] for _ in range(n_rows)]
        
    return best_rows

# EVOLVE-BLOCK-START
# (No additional code needed here as construct is defined above)
# EVOLVE-BLOCK-END
