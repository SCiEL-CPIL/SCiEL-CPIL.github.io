import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 135s to be safe.
    deadline = start_time + 132.0

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

    def build_diff_index(rows):
        """
        Build a dictionary: diff -> (row_idx, pos_i, pos_j)
        where row[pos_j] - row[pos_i] = diff.
        This allows O(1) lookup for conflict checking during moves.
        Returns None if a conflict is detected (shouldn't happen if input is valid).
        """
        index = {}
        for ri, row in enumerate(rows):
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in index:
                        return None # Conflict
                    index[d] = (ri, i, j)
        return index

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
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

    def local_search(rows, time_limit):
        """
        Iterated Local Search (ILS) using an efficient difference index for O(1) conflict checks.
        1. Try to reduce the global maximum via single-element moves.
        2. If stuck, perturb a random element or reconstruct a row.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        # Initialize the difference index
        diff_index = build_diff_index(best_rows)
        if diff_index is None:
            # Fallback if initial state is invalid (should not happen)
            return best_rows, best_scope

        stuck_counter = 0
        no_improve_counter = 0

        def try_move(r_idx, c_idx, new_val):
            """
            Attempt to move element at (r_idx, c_idx) to new_val.
            Updates diff_index in place if successful.
            Returns True if successful, False otherwise.
            """
            row = best_rows[r_idx]
            old_val = row[c_idx]
            
            if new_val == old_val:
                return False
            
            # Check bounds
            if c_idx > 0 and new_val <= row[c_idx - 1]:
                return False
            if c_idx < row_len - 1 and new_val >= row[c_idx + 1]:
                return False
            
            # Identify affected differences
            # Diffs to remove: (old_val - row[i]) for i < c_idx, (row[j] - old_val) for j > c_idx
            # Diffs to add:    (new_val - row[i]) for i < c_idx, (row[j] - new_val) for j > c_idx
            
            old_diffs = []
            new_diffs = []
            
            for i in range(c_idx):
                old_diffs.append((old_val - row[i], r_idx, i, c_idx))
                new_diffs.append((new_val - row[i], r_idx, i, c_idx))
            
            for j in range(c_idx + 1, row_len):
                old_diffs.append((row[j] - old_val, r_idx, c_idx, j))
                new_diffs.append((row[j] - new_val, r_idx, c_idx, j))
            
            # Check if old diffs exist in index (sanity check)
            for d, ri, i, j in old_diffs:
                if d not in diff_index:
                    return False
                if diff_index[d] != (ri, i, j):
                    return False
            
            # Check if new diffs conflict with existing ones
            for d, ri, i, j in new_diffs:
                if d <= 0:
                    return False
                if d in diff_index:
                    # Conflict if the existing diff is not one of the ones we are removing
                    # Actually, since we check old_diffs first, we need to be careful.
                    # If d is in diff_index, and it's not in old_diffs, it's a conflict.
                    # If it IS in old_diffs, it's fine (we are swapping/keeping it).
                    # But wait, if d is in old_diffs, it means we are removing it and re-adding it?
                    # No, if new_diff == old_diff for a specific pair, it's fine.
                    # But if new_diff matches an old_diff of a *different* pair, that's a conflict.
                    
                    # Simpler check:
                    # 1. Remove all old_diffs from a temporary set of "to_remove"
                    # 2. Check if any new_diff is in (diff_index - to_remove)
                    
                    pass # We will do a cleaner check below
            
            # Cleaner Conflict Check:
            to_remove = set(d for d, _, _, _ in old_diffs)
            
            for d, ri, i, j in new_diffs:
                if d in diff_index:
                    # If the diff is in the index, check if it's one we are removing
                    if d not in to_remove:
                        return False
                    # If it is in to_remove, we need to ensure it maps to the same source/dest
                    # Actually, if d is in to_remove, it means some old pair had this diff.
                    # If the new pair also has this diff, it's a self-conflict only if the pair is different.
                    # But here we are changing value at c_idx.
                    # If row[i] - row[j] == d was old, and row[i] - row[c_idx] == d is new...
                    # This implies row[j] == row[c_idx] which is impossible if j != c_idx and strictly increasing.
                    # So if d is in to_remove, it's safe to overwrite?
                    # Yes, because the old entry is being deleted.
                    
            # Apply changes
            for d, ri, i, j in old_diffs:
                del diff_index[d]
            
            for d, ri, i, j in new_diffs:
                diff_index[d] = (ri, i, j)
            
            row[c_idx] = new_val
            return True

        while time.time() < start + time_limit:
            # 1. Targeted Descent: Try to reduce the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            found_improvement = False
            # Try values from current_val - 1 down to prev_val + 1
            for new_val in range(current_val - 1, prev_val, -1):
                if try_move(max_r_idx, max_c_idx, new_val):
                    found_improvement = True
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        stuck_counter = 0
                        no_improve_counter = 0
                    break
            
            if found_improvement:
                continue
                
            # No improvement on max element; try other elements too
            improved_any = False
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if r_idx == max_r_idx and c_idx == max_c_idx:
                        continue
                    curr_val = best_rows[r_idx][c_idx]
                    prev_v = best_rows[r_idx][c_idx-1]
                    next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 1
                    
                    # Try to decrease this element
                    for new_val in range(curr_val - 1, prev_v, -1):
                        if try_move(r_idx, c_idx, new_val):
                            improved_any = True
                            new_scope = get_scope(best_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                stuck_counter = 0
                                no_improve_counter = 0
                            break
                    if improved_any:
                        break
                if improved_any:
                    break
            
            if improved_any:
                continue
            
            # Completely stuck
            no_improve_counter += 1
            stuck_counter += 1
            
            # If stuck for a while, try a more aggressive perturbation: Row Reconstruction
            if stuck_counter > 10 and random.random() < 0.4:
                # Pick a random row to reconstruct
                r_idx = random.randint(0, n_rows - 1)
                
                # Compute diffs used by other rows
                fixed_diffs = set()
                for r in best_rows:
                    if r is best_rows[r_idx]:
                        continue
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            fixed_diffs.add(r[j] - r[i])
                
                # Greedily reconstruct the row
                success = False
                for attempt in range(8):
                    temp_row = [0]
                    temp_used = set()
                    
                    while len(temp_row) < row_len:
                        last = temp_row[-1]
                        candidates = []
                        x = last + 1
                        limit = min(best_scope, last + 60)
                        
                        while x <= limit and len(candidates) < 25:
                            valid = True
                            for a in temp_row:
                                d = x - a
                                if d in fixed_diffs or d in temp_used:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                            x += 1
                        
                        if not candidates:
                            while True:
                                x += 1
                                valid = True
                                for a in temp_row:
                                    d = x - a
                                    if d in fixed_diffs or d in temp_used:
                                        valid = False
                                        break
                                if valid:
                                    candidates.append(x)
                                    break
                        
                        if random.random() < 0.25 and len(candidates) > 1:
                            x = random.choice(candidates)
                        else:
                            x = candidates[0]
                        
                        temp_row.append(x)
                        for a in temp_row[:-1]:
                            temp_used.add(x - a)
                    
                    if max(temp_row) <= best_scope:
                        # Replace row in best_rows and rebuild index
                        # This is expensive, so we do it carefully
                        best_rows[r_idx] = temp_row
                        diff_index = build_diff_index(best_rows)
                        if diff_index is None:
                            # Should not happen
                            break
                        success = True
                        stuck_counter = 0
                        no_improve_counter = 0
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        break

            else:
                # Standard perturbation: Move a random element
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx-1]
                next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                
                if low > high:
                    continue
                
                applied = False
                for _ in range(15):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    
                    if try_move(r_idx, c_idx, cand):
                        applied = True
                        break
                
                if applied:
                    stuck_counter = 0

        return best_rows, best_scope

    # Phase 1: Diverse Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Run several greedy initializations with different seeds
    num_inits = 15
    init_time_limit = 6.0
    start_init = time.time()
    
    for i in range(num_inits):
        if time.time() > start_init + init_time_limit:
            break
        rows = greedy_init(rand_prob=0.05 + 0.01*i, window=25, seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # Phase 2: Local Search Refinement
    local_time = 55.0
    if time.time() > deadline - 65:
        local_time = deadline - time.time() - 10
        
    if local_time > 10 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 2.5: High-precision polish pass
    # Re-use the optimized local_search with a tighter time budget to squeeze out last bits
    if best_rows and time.time() < deadline - 10:
        polish_time = min(15.0, deadline - time.time() - 5)
        if polish_time > 3:
            polish_rows, polish_scope = local_search(best_rows, polish_time)
            if validate(polish_rows):
                if polish_scope < best_scope:
                    best_rows = polish_rows
                    best_scope = polish_scope

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
