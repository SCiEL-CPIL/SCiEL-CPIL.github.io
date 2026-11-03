import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 140s to be safe.
    deadline = start_time + 140.0 

    total_diffs = n_rows * k * (k + 1) // 2
    # Theoretical lower bound on scope
    # Each row has k*(k+1)/2 diffs. Total diffs = n_rows * k*(k+1)/2.
    # These diffs must be distinct integers from 1..M.
    # So M >= total_diffs.
    # Also, the largest element in a row is at least the largest diff in that row.
    # A simple bound is M >= total_diffs.
    lb = total_diffs + 1
    
    # Heuristic upper bound for initial search
    # The best known is 134. Let's start a bit higher to ensure we find *something*.
    initial_upper_bound = 160 

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate_rows(rows):
        """Check all differences are distinct across all rows."""
        used = bytearray(initial_upper_bound + 100)  # Slightly larger than expected
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d > len(used) - 1:
                        return False
                    if used[d]:
                        return False
                    used[d] = 1
        return True

    def _greedy_construct(seed, target_scope_hint=200):
        """
        Fast greedy construction to get an initial valid solution.
        Uses a bitset for speed.
        """
        rng = random.Random(seed)
        # We don't know the exact scope yet, so use a generous bitset size
        # We will resize if we exceed it, but for greedy, we just grow.
        # Actually, for speed, let's just use a set first for the greedy phase, 
        # then convert to bitset for the optimization phase.
        # Or, we can use a large enough bytearray.
        # Max scope likely < 200. Let's use 300.
        used = bytearray(300)
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                # Try small jumps first, then random
                last = row[-1]
                # Randomized search window
                search_end = last + 50 + rng.randint(0, 50)
                if search_end > 250:
                    search_end = 250
                
                # Try candidates
                candidates = list(range(last + 1, search_end))
                rng.shuffle(candidates)
                
                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d >= len(used):
                            # This shouldn't happen if we sized correctly, but just in case
                            valid = False
                            break
                        if used[d]:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        for d in new_diffs:
                            used[d] = 1
                        row.append(x)
                        found = True
                        break
                
                if not found:
                    # Fallback: linear search
                    for x in range(last + 1, 300):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d >= len(used):
                                valid = False
                                break
                            if used[d]:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            for d in new_diffs:
                                used[d] = 1
                            row.append(x)
                            found = True
                            break
                    if not found:
                        return None
            rows.append(row)
        
        return rows

    def _local_compact(rows, seed, time_limit):
        """
        Local improvement: try to move each element to a smaller position.
        Uses bitset for speed.
        """
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        
        # Build bitset
        max_val = scope + 10
        used = bytearray(max_val + 1)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used[d] = 1

        for _ in range(5000): # Max iterations
            if time.time() > time_limit:
                break
                
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            
            if old_val <= 1:
                continue

            # Remove old diffs
            for a in row[:elem_idx]:
                used[old_val - a] = 0
            for a in row[elem_idx + 1:]:
                used[a - old_val] = 0
            
            row.pop(elem_idx)
            
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val_try = old_val - 1
            
            # Try to find a smaller valid value
            for new_val in range(min_val, max_val_try + 1):
                valid = True
                new_diffs = []
                for i, a in enumerate(row):
                    if i < elem_idx:
                        d = new_val - a
                    else:
                        d = a - new_val
                    if d >= len(used):
                        valid = False
                        break
                    if used[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    for d in new_diffs:
                        used[d] = 1
                    row.insert(elem_idx, new_val)
                    found = True
                    break
            
            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                # Restore
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used[old_val - a] = 1
                for a in row[elem_idx + 1:]:
                    used[a - old_val] = 1

        return rows, scope

    def _backtracking_search(target_scope, deadline, seed_offset=0):
        """
        Try to find a valid set of rows with max entry <= target_scope.
        Uses DFS with bitset pruning.
        """
        rng = random.Random(seed_offset * 1000 + 7)
        L = k + 1
        rows = []
        used = bytearray(target_scope + 1)
        
        def build_row(row_idx, current_row, pos):
            if time.time() > deadline:
                return False
            
            if pos == L:
                rows.append(list(current_row))
                return True
            
            last = current_row[-1]
            
            # Pruning: if we can't fit the remaining elements within target_scope
            # The remaining elements must be at least 1 apart.
            # So the next element must be <= target_scope - (L - pos - 1)
            max_next = target_scope - (L - pos - 1)
            
            if max_next <= last:
                return False

            # Candidates: last+1 to max_next
            # Heuristic: try smaller values first to keep the row compact
            # But randomize slightly to escape local minima?
            # For exact search, we try all. But we can order them.
            
            # To speed up, we can skip candidates that create immediate conflicts.
            # We iterate x from last+1 to max_next
            
            # Optimization: If row_idx is 0, we can try to fix the first few elements
            # to be as small as possible to constrain the search space for other rows.
            
            for x in range(last + 1, max_next + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if used[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    for d in new_diffs:
                        used[d] = 1
                    current_row.append(x)
                    
                    if build_row(row_idx, current_row, pos + 1):
                        return True
                    
                    # Backtrack
                    for d in new_diffs:
                        used[d] = 0
                    current_row.pop()
            
            return False

        # Start building rows
        # We can randomize the order of rows or the starting points to find solutions faster?
        # No, rows are symmetric in terms of validity, but the search space is different.
        # Let's just build them in order.
        
        # Heuristic: Try to make the first row as compact as possible.
        # This reduces the available differences for subsequent rows.
        
        # We'll run this search multiple times with different random seeds if it fails?
        # No, it's a deterministic search for a fixed target_scope.
        # If it fails, it fails.
        
        # However, we can randomize the *order* of candidates if we want to be stochastic.
        # For now, let's stick to deterministic "smallest first" which is usually best for finding *a* solution.
        
        # One issue: If we always pick the smallest valid x, we might get stuck.
        # We can add a small randomization: with probability p, pick a random valid x from the top 10.
        
        # Let's implement a slight randomization to avoid getting stuck in a "greedy trap".
        
        def build_row_rand(row_idx, current_row, pos):
            if time.time() > deadline:
                return False
            
            if pos == L:
                rows.append(list(current_row))
                return True
            
            last = current_row[-1]
            max_next = target_scope - (L - pos - 1)
            
            if max_next <= last:
                return False

            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, max_next + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if used[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    valid_candidates.append((x, new_diffs))
            
            if not valid_candidates:
                return False
            
            # Randomize order if we have many candidates
            if len(valid_candidates) > 5:
                # Shuffle a subset or all?
                # Let's shuffle the whole list
                rng.shuffle(valid_candidates)
            
            for x, new_diffs in valid_candidates:
                for d in new_diffs:
                    used[d] = 1
                current_row.append(x)
                
                if build_row_rand(row_idx, current_row, pos + 1):
                    return True
                
                for d in new_diffs:
                    used[d] = 0
                current_row.pop()
            
            return False

        for r in range(n_rows):
            if not build_row_rand(r, [0], 1):
                return None
        
        return rows

    # Phase 1: Get a baseline solution
    # Try a few greedy seeds
    for i in range(20):
        if time.time() > deadline - 30:
            break
        rows = _greedy_construct(i)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: Very loose greedy
        rows = _greedy_construct(999)
        if rows:
            best_rows = rows
            best_scope = get_scope(rows)
        else:
            return [] # Should not happen

    # Phase 2: Local Compaction to tighten the baseline
    if time.time() < deadline - 20:
        improved, imp_scope = _local_compact(best_rows, 42, deadline - 10)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 3: Targeted Backtracking Search
    # Try to find a solution with scope < best_scope
    # We iterate target_scope downwards from best_scope - 1
    # We stop if we hit the lower bound or time runs out.
    
    # The backtracking search is expensive. We should only try it for a few steps.
    # Also, we can use the current best_rows as a "hint" or starting point?
    # No, the backtracking search builds from scratch.
    
    # Optimization: We can limit the number of backtracking attempts.
    # Each attempt is a full DFS.
    
    # Let's try scopes: best_scope - 1, best_scope - 2, ... down to max(lb, best_scope - 10)
    # If we find one, we update best_scope and continue.
    
    # To make the search faster, we can use the fact that the rows are symmetric.
    # We can fix the first row to be lexicographically smallest?
    # The build_row_rand function already tries to keep it compact.
    
    # We will run the backtracking search for a limited time per scope.
    # If it doesn't find a solution in 5 seconds, we move to the next scope.
    
    current_target = best_scope - 1
    min_target = max(lb, best_scope - 15) # Don't go too far down if it's hard
    
    while current_target >= min_target and time.time() < deadline - 5:
        # Allocate time for this search
        search_deadline = time.time() + 5.0
        if search_deadline > deadline - 2:
            search_deadline = deadline - 2
            
        # Try a few random seeds for the backtracking search
        found = False
        for seed in range(5):
            if time.time() > search_deadline:
                break
            rows = _backtracking_search(current_target, search_deadline, seed)
            if rows:
                # Verify
                if validate_rows(rows):
                    scope = get_scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                        found = True
                        break
        
        if found:
            # If we found a solution, we can try to compact it further
            if time.time() < deadline - 10:
                improved, imp_scope = _local_compact(best_rows, 123, deadline - 5)
                if imp_scope < best_scope:
                    best_scope = imp_scope
                    best_rows = improved
            
            # Continue searching for even lower scopes
            current_target -= 1
        else:
            # If we didn't find a solution for this target, it might be impossible
            # or too hard. We can break or continue?
            # Usually, if scope S is not possible, S-1 is also not.
            # But due to search limitations, we might have missed it.
            # Let's break to save time for other strategies if we fail a few times in a row.
            # Actually, let's just break if we fail. The greedy+compaction is more reliable for finding *some* solution.
            # The backtracking is for the "last mile" improvement.
            break

    return best_rows

# Quick self-test
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
