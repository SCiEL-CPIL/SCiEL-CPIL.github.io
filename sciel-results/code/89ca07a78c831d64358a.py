import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution with multiple random restarts for better upper bound
    def greedy_random(n_rows, k, seed):
        rng = random.Random(seed)
        used_diffs = set()
        greedy_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
            greedy_rows.append(row)
        return greedy_rows
    
    # Try multiple greedy seeds to get a good initial bound
    for seed in range(20):
        if time.time() > deadline:
            break
        greedy_rows = greedy_random(n_rows, k, seed)
        g_max = max(max(r) for r in greedy_rows)
        if g_max < best_max:
            best_max = g_max
            best_rows = greedy_rows
    
    # Bitset-based CSP with forward checking and randomized restarts
    # We use bitmasks for efficient difference checks.
    # Max possible diff is bounded by best_max initially, but we search for lower targets.
    # Let's define a maximum search bound. The problem size is small (k=8, n_rows=4).
    # Max entry ~134. Diffs are 1..133.
    # We can use an integer bitmask where bit d is set if diff d is used.
    
    def try_construct(target_max, n_rows, k, random_state):
        used_diff_mask = 0
        rows = []
        
        # Precompute valid candidates for each row position given current mask?
        # No, mask changes. But we can precompute which x values are valid for a given row prefix and mask?
        # That's too expensive.
        
        # Instead, we check on the fly.
        # To speed up, we can precompute for each x (1..target_max) the bitmask of diffs it introduces
        # relative to a row prefix? No, it depends on the row prefix.
        
        # Let's just use sets for clarity first, then optimize if needed.
        # Actually, bitmask operations are faster.
        # diff d is represented as bit (d-1) or just d?
        # Let's use bit d for diff d.
        
        def get_diff_mask_for_row(row):
            mask = 0
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
            return mask

        def backtrack_row(idx, row, used_diff_mask):
            if idx == k:
                return row[:]
            
            last = row[-1]
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                # Check if adding x creates any conflict
                # Diffs are x - a for a in row
                # We need to check if any of these diffs are already in used_diff_mask
                # And they must be distinct (they are, since row is strictly increasing)
                
                conflict = False
                for a in row:
                    d = x - a
                    if (used_diff_mask >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    valid_candidates.append(x)
                    
            if not valid_candidates:
                return None
            
            # Heuristic: Prefer smaller x, but randomize among a window
            W = min(len(valid_candidates), 15)
            candidates_to_try = valid_candidates[:W]
            random_state.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                # Calculate new diffs mask
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                
                res = backtrack_row(idx + 1, new_row, used_diff_mask | new_diffs_mask)
                if res is not None:
                    return res
                    
            return None

        # Main loop for rows
        for i in range(n_rows):
            row = [0]
            res = backtrack_row(0, row, used_diff_mask)
            if res is None:
                return None
            # Update global mask
            used_diff_mask |= get_diff_mask_for_row(res)
            rows.append(res)
            
        return rows

    # Local improvement: try to reduce the max entry by rebuilding rows
    def try_local_improve(rows, n_rows, k, random_state):
        """Try to rebuild individual rows to reduce the maximum entry."""
        current_max = max(max(r) for r in rows)
        for _ in range(30):
            if time.time() > deadline:
                break
            # Pick a random row to rebuild
            row_idx = random_state.randint(0, n_rows - 1)
            
            # Compute used diffs excluding this row
            all_diffs = set()
            for i, r in enumerate(rows):
                if i == row_idx:
                    continue
                for j in range(len(r)):
                    for l in range(j+1, len(r)):
                        all_diffs.add(r[l] - r[j])
            
            # Try to rebuild row_idx with a smaller max
            target = current_max - 1
            if target < k:
                target = k
            
            # Backtrack to rebuild this row
            def rebuild_row(target_max, used_diffs_set):
                def bt(row, used_diffs):
                    if len(row) == k + 1:
                        return row[:]
                    last = row[-1]
                    for x in range(last + 1, target_max + 1):
                        new_diffs = {x - a for a in row}
                        if new_diffs & used_diffs:
                            continue
                        res = bt(row + [x], used_diffs | new_diffs)
                        if res is not None:
                            return res
                    return None
                return bt([0], set(used_diffs_set))
            
            new_row = rebuild_row(target, all_diffs)
            if new_row is not None:
                rows[row_idx] = new_row
                new_max = max(max(r) for r in rows)
                if new_max < current_max:
                    current_max = new_max
        
        return rows, current_max

    # Main search loop
    # We want to minimize the maximum entry.
    # Start from best_max - 1 and go down.
    # If we find a solution, update best_max and continue.
    
    current_target = best_max - 1
    
    while time.time() < deadline and current_target >= k:
        found = False
        # Try multiple restarts for this target
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            random_state = random.Random()
            res = try_construct(current_target, n_rows, k, random_state)
            if res is not None:
                current_max = max(max(r) for r in res)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in res]
                    # Try local improvement
                    if time.time() < deadline:
                        improved_rows, improved_max = try_local_improve(
                            [r[:] for r in best_rows], n_rows, k, random_state
                        )
                        if improved_max < best_max:
                            best_max = improved_max
                            best_rows = improved_rows
                found = True
                break
        
        if found:
            current_target = best_max - 1
        else:
            # If we can't find a solution for current_target, 
            # it might be impossible or just hard.
            # Stop searching for lower targets.
            break

    return best_rows
