import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct(seed, max_scope=None):
        """
        Fast greedy construction with optional scope limit.
        Uses randomized candidate ordering to explore different local optima.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Determine search range
                if max_scope is not None:
                    upper = max_scope + 1
                else:
                    upper = last + 200
                    
                # Generate candidates: bias towards smaller values but allow some randomness
                # We want to keep the row compact
                candidates = list(range(last + 1, upper))
                
                # If the range is large, we can't just shuffle it all (too slow/memory).
                # But for (4,7) and scope ~134, last+200 is ~334. Shuffling 334 items is fine.
                # However, we want to prioritize smaller values.
                # Strategy: try the first few in random order, then expand.
                
                # Phase 1: Try small increments (last+1 to last+20) in random order
                small_range = min(20, upper - last - 1)
                if small_range > 0:
                    small_cands = list(range(last + 1, last + small_range + 1))
                    rng.shuffle(small_cands)
                    for x in small_cands:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                # Phase 2: If not found, try wider range
                if not found:
                    # Start from where we left off or just scan from last+1 to upper
                    # To save time, we scan linearly but check in a semi-random way?
                    # Actually, linear scan from last+1 is fast if we use a set for 'used'.
                    for x in range(last + 1, upper):
                        new_diffs = [x - a for a in row]
                        # Quick check: if any diff is in used, skip
                        conflict = False
                        for d in new_diffs:
                            if d in used:
                                conflict = True
                                break
                        if conflict:
                            continue
                        
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    return None
            rows.append(row)
        return rows

    # 1. Get initial valid solution
    best_rows = None
    best_scope = float('inf')
    
    # Try a few greedy seeds to get a good starting point
    for i in range(20):
        if time.time() > deadline - 10:
            break
        rows = greedy_construct(i)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback: should rarely happen
        return [[0] * (k + 1) for _ in range(n_rows)]

    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    used_set = compute_used_set(current_rows)

    # 2. Compaction-based Local Search
    # This is the "refinement machinery" from the current program, which is strong.
    # We iterate elements and try to move them to the smallest valid position.
    
    def try_compact(rows, used_set):
        """
        Attempts to reduce the scope of rows by moving elements to smaller values.
        Returns True if any change was made.
        """
        changed = False
        for r_idx in range(n_rows):
            row = rows[r_idx]
            # Process indices from last to first (excluding 0) to allow left-shifting
            for e_idx in range(len(row) - 1, 0, -1):
                old_val = row[e_idx]
                min_val = row[e_idx - 1] + 1
                
                if old_val <= min_val:
                    continue
                
                # We want to find the smallest v in [min_val, old_val) that is valid
                # To speed this up, we can iterate v from min_val upwards.
                # If we find a valid v, we set row[e_idx] = v and break.
                
                found_valid = False
                for new_val in range(min_val, old_val):
                    # Check bounds with next element
                    if e_idx < len(row) - 1 and new_val >= row[e_idx + 1]:
                        break # Since we're increasing new_val, it will only get larger
                    
                    # Calculate diffs
                    # Removed diffs: |row[i] - old_val| for i != e_idx
                    # Added diffs: |row[i] - new_val| for i != e_idx
                    
                    removed_diffs = set()
                    added_diffs = set()
                    for i in range(len(row)):
                        if i == e_idx:
                            continue
                        removed_diffs.add(abs(row[i] - old_val))
                        added_diffs.add(abs(row[i] - new_val))
                    
                    # Check internal duplicates in added_diffs
                    # Number of pairs involving e_idx is len(row) - 1
                    if len(added_diffs) != len(row) - 1:
                        continue
                        
                    # Check against global used set
                    # new_used = used_set - removed_diffs
                    # if added_diffs & new_used: invalid
                    
                    # Optimization: Check intersection
                    # We can do: if added_diffs & (used_set - removed_diffs)
                    # But set subtraction creates a new set.
                    # Alternative: 
                    # for d in added_diffs:
                    #     if d in used_set and d not in removed_diffs:
                    #         conflict = True
                    #         break
                    
                    conflict = False
                    for d in added_diffs:
                        if d in used_set:
                            if d not in removed_diffs:
                                conflict = True
                                break
                    if conflict:
                        continue
                        
                    # Valid!
                    row[e_idx] = new_val
                    used_set = (used_set - removed_diffs) | added_diffs
                    changed = True
                    found_valid = True
                    break # Found the smallest valid value for this element
                
        return changed, used_set

    # 3. Perturbation
    def perturb(rows, used_set, rng):
        """
        Randomly increases a random element to a valid value to escape local minima.
        """
        r_idx = rng.randint(0, n_rows - 1)
        e_idx = rng.randint(1, len(rows[r_idx]) - 1)
        old_val = rows[r_idx][e_idx]
        
        # Try to find a valid new value > old_val
        # Search range: old_val + 1 to old_val + 50 (or up to current max scope + some buffer)
        max_search = old_val + 50
        
        for _ in range(10): # Try a few random jumps
            new_val = rng.randint(old_val + 1, max_search)
            
            if e_idx < len(rows[r_idx]) - 1 and new_val >= rows[r_idx][e_idx + 1]:
                continue
            
            removed_diffs = set()
            added_diffs = set()
            for i in range(len(rows[r_idx])):
                if i == e_idx:
                    continue
                removed_diffs.add(abs(rows[r_idx][i] - old_val))
                added_diffs.add(abs(rows[r_idx][i] - new_val))
            
            if len(added_diffs) != len(rows[r_idx]) - 1:
                continue
                
            conflict = False
            for d in added_diffs:
                if d in used_set:
                    if d not in removed_diffs:
                        conflict = True
                        break
            if conflict:
                continue
                
            rows[r_idx][e_idx] = new_val
            used_set = (used_set - removed_diffs) | added_diffs
            return True
            
        return False

    # 4. Backtracking Search for Tighter Scope
    # This is the "structural idea" from Inspiration 1/2: try to construct a solution
    # with a specific, smaller max_scope.
    
    def backtracking_search(target_scope, seed, deadline):
        """
        Attempts to construct a solution where all elements are <= target_scope.
        Uses randomized backtracking.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        def build_row(row_idx, pos, current_mark, row_so_far):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return row_so_far
            
            # Candidates must be > current_mark and <= target_scope
            # Also, we need enough room to place remaining elements?
            # Simple check: if current_mark + (k+1 - pos) > target_scope, fail?
            # Not strictly, but if current_mark + 1 > target_scope, fail.
            
            if current_mark + 1 > target_scope:
                return None
            
            valid_candidates = []
            # We want to try smaller candidates first to keep scope low, but randomize order slightly
            # Or just iterate from current_mark+1 to target_scope
            # For efficiency, we can collect a few candidates and shuffle
            
            # To avoid iterating too many, we can break if we have enough?
            # No, we need to check all if we want to be thorough, but for speed,
            # we can sample.
            
            # Let's collect candidates up to a limit, or just iterate.
            # Given target_scope ~134, iterating 134 items is fast.
            
            # We will iterate from current_mark + 1 to target_scope
            # But we will shuffle the order of attempts to escape deterministic traps.
            # However, we must respect the increasing order constraint.
            # So we can't just shuffle the whole range.
            # We can pick a random candidate from the valid ones, or try the first few in random order.
            
            # Strategy: 
            # 1. Find all valid candidates in [current_mark+1, target_scope]
            # 2. If none, return None
            # 3. Pick a subset (e.g., first 10) and shuffle them
            # 4. Try them in that order
            
            candidates = []
            for x in range(current_mark + 1, target_scope + 1):
                new_diffs = {x - a for a in row_so_far}
                if new_diffs & used:
                    continue
                candidates.append(x)
                
                # Optimization: if we have many candidates, we can stop collecting?
                # No, because a later candidate might be better? 
                # Actually, for backtracking, we just need *any* valid path.
                # Smaller candidates are generally better for future flexibility?
                # Not necessarily. 
                # But collecting all is O(M^2) which is fine for M=134.
            
            if not candidates:
                return None
            
            # Shuffle a portion of the candidates to randomize the search path
            # We try the first 20 candidates in random order
            sample_pool = candidates[:20]
            rng.shuffle(sample_pool)
            
            for x in sample_pool:
                new_diffs = {x - a for a in row_so_far}
                
                # Add to used
                used.update(new_diffs)
                
                # Recurse
                res = build_row(row_idx, pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                
                # Undo
                used.difference_update(new_diffs)
                
            return None

        for i in range(n_rows):
            row = build_row(i, 1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            
        return rows

    # Main Loop
    rng = random.Random(12345)
    
    # Phase A: Local Search (Compaction + Perturbation)
    # Run this for a while to polish the greedy solution
    iterations = 0
    max_ls_iterations = 500
    no_improve_count = 0
    
    while time.time() < deadline - 30 and iterations < max_ls_iterations:
        iterations += 1
        
        # Compact
        changed, used_set = try_compact(current_rows, used_set)
        
        if changed:
            new_scope = get_scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # If stuck, perturb
        if no_improve_count > 5:
            if perturb(current_rows, used_set, rng):
                # After perturbation, compact again immediately?
                # The next loop iteration will compact.
                pass
            no_improve_count = 0
            
    # Phase B: Backtracking Search for Lower Scope
    # Try to find a solution with scope < best_scope
    # We decrement target_scope and try to find a solution.
    
    # Start from best_scope - 1
    # We can also try to jump down by a few units if we have time.
    
    current_target = best_scope - 1
    bt_attempts = 0
    
    while time.time() < deadline - 5 and current_target > lower_bound:
        bt_attempts += 1
        
        # Try a few random seeds for this target
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            solution = backtracking_search(current_target, seed, deadline)
            if solution:
                new_scope = get_scope(solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in solution]
                    found = True
                    # Update current_rows for further local search?
                    # Yes, let's restart local search with this better solution
                    current_rows = [list(r) for r in best_rows]
                    used_set = compute_used_set(current_rows)
                    
                    # If we hit the lower bound, we are done
                    if best_scope <= lower_bound:
                        return best_rows
                    break
        
        if found:
            # If we found a better solution, we might have dropped the scope by more than 1?
            # No, backtracking_search ensures max element <= current_target.
            # So new_scope <= current_target.
            # We can set current_target = new_scope - 1 to try to go even lower.
            current_target = best_scope - 1
        else:
            # If we failed to find a solution for current_target, 
            # it's possible that no solution exists for this target.
            # We should decrement target and try again.
            current_target -= 1

    # Final Polish: Run local search again on the best solution found
    if time.time() < deadline - 2:
        current_rows = [list(r) for r in best_rows]
        used_set = compute_used_set(current_rows)
        
        for _ in range(100): # Quick polish
            if time.time() > deadline - 2:
                break
            changed, used_set = try_compact(current_rows, used_set)
            if changed:
                new_scope = get_scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]

    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
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
