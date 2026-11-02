import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # 10s margin

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i):
                diffs.add(row[i] - row[j])
        return diffs

    def generate_candidate_row():
        row = [0]
        used_diffs = set()
        # Randomized greedy
        # We want to keep max small.
        # At each step, find the smallest valid x, but sometimes pick a slightly larger one to allow future flexibility?
        # Actually, for difference sets, greedy smallest is usually best for minimizing max.
        # Randomization helps in finding *different* rows, but they will all be locally optimal.
        # This might lead to a pool of very similar rows.
        # To get diverse rows, we can introduce "noise":
        # Pick x from {smallest_valid, smallest_valid + 1, ..., smallest_valid + noise}
        
        max_noise = 2
        while len(row) < k + 1:
            # Find smallest valid x
            x = row[-1] + 1
            smallest_valid = None
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    smallest_valid = x
                    break
                x += 1
            
            if smallest_valid is None:
                return None # Should not happen
            
            # Pick from a small range around smallest_valid
            # But ensure we don't pick invalid ones
            valid_options = []
            x = smallest_valid
            for _ in range(3 + max_noise): # Check a few
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_options.append(x)
                x += 1
            
            if not valid_options:
                return None
            
            # Weighted choice: prefer smaller
            # Simple: random.choice
            x = random.choice(valid_options)
            
            row.append(x)
            # Update used_diffs
            # The new diffs are x - a for a in row[:-1]
            for a in row[:-1]:
                used_diffs.add(x - a)
                
        return row

    best_rows = None
    best_scope = float('inf')

    # Multiple attempts to generate better pools
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate pool
        pool_size = 1000 + attempts * 500 # Increase pool size in later attempts?
        # Actually, keep it constant but vary randomness.
        pool_size = 1500
        
        candidates = []
        # Use a local time limit for generation
        gen_deadline = time.time() + (deadline - time.time()) * 0.5
        while time.time() < gen_deadline and len(candidates) < pool_size:
            r = generate_candidate_row()
            if r:
                candidates.append(r)
        
        if len(candidates) < n_rows:
            continue

        # Precompute diffs
        cand_info = []
        for r in candidates:
            d = get_diffs(r)
            cand_info.append((r[-1], r, d))
        
        # Sort by max entry
        cand_info.sort(key=lambda x: x[0])
        
        # Backtrack to find n_rows non-conflicting rows
        # We want to minimize the max entry of the selected rows.
        # Since sorted, the max entry of the set is the max entry of the last selected row.
        
        selected = []
        used_diffs_global = set()
        
        def backtrack(idx, count):
            nonlocal best_rows, best_scope
            
            if count == n_rows:
                # Update best
                curr_max = selected[-1][1][-1]
                if curr_max < best_scope:
                    best_scope = curr_max
                    best_rows = [r[1][:] for r in selected]
                return True
            
            # Pruning: if we have picked count rows, and the current best scope is S,
            # any row we pick now must have max < S?
            # Not necessarily, the max of the *set* is the max of the last picked row.
            # If we pick a row with max M, the set max is at least M.
            # So if M >= best_scope, we can prune.
            
            for i in range(idx, len(cand_info)):
                max_val, r, d = cand_info[i]
                
                if max_val >= best_scope:
                    break
                
                if d & used_diffs_global:
                    continue
                
                # Select
                selected.append(cand_info[i])
                # Update used_diffs_global
                # We need to restore it after recursion
                old_used = used_diffs_global.copy()
                used_diffs_global |= d
                
                if backtrack(i + 1, count + 1):
                    # Restore and return if we found a better one?
                    # Actually, if we found a solution, we might want to continue to find a better one?
                    # Or just return?
                    # Since we sorted by max, the first solution found in this branch might not be the absolute best
                    # if a later branch allows a smaller max? No, later branches have larger max rows.
                    # Wait.
                    # If we pick row i, the max is max_val.
                    # If we skip i and pick i+1, the max is max_val_{i+1} >= max_val.
                    # So the first valid combination found in a DFS that iterates i in increasing order
                    # will have the smallest possible max for that specific prefix of choices?
                    # Yes. So we can return True immediately upon finding a valid set.
                    used_diffs_global = old_used
                    selected.pop()
                    return True
                
                used_diffs_global = old_used
                selected.pop()
                
            return False

        # Run backtrack
        # To avoid deep recursion issues or long runs, we can add a time check inside backtrack?
        # Python recursion limit might be an issue if n_rows is large, but n_rows is small (e.g. 4).
        # However, the branching factor is large.
        # We can limit the search time per attempt.
        
        search_start = time.time()
        # We'll just run it. If it takes too long, the outer loop will stop.
        # But backtrack might hang.
        # Let's add a global time check in backtrack?
        # It's hard to break out of deep recursion easily without exceptions.
        # Given n_rows=4, depth is 4. Branching factor is ~1500.
        # 1500^4 is too big.
        # But pruning is strong: `if d & used_diffs_global: continue`.
        # Most rows will conflict.
        
        # To be safe, let's limit the number of nodes visited in backtrack.
        nodes_visited = 0
        max_nodes = 100000
        
        def backtrack_limited(idx, count):
            nonlocal best_rows, best_scope, nodes_visited
            
            if time.time() > deadline - 2:
                return False
            if nodes_visited > max_nodes:
                return False
            nodes_visited += 1

            if count == n_rows:
                curr_max = selected[-1][1][-1]
                if curr_max < best_scope:
                    best_scope = curr_max
                    best_rows = [r[1][:] for r in selected]
                return True
            
            for i in range(idx, len(cand_info)):
                max_val, r, d = cand_info[i]
                
                if max_val >= best_scope:
                    break
                
                if d & used_diffs_global:
                    continue
                
                selected.append(cand_info[i])
                old_used = used_diffs_global.copy()
                used_diffs_global |= d
                
                if backtrack_limited(i + 1, count + 1):
                    used_diffs_global = old_used
                    selected.pop()
                    return True
                
                used_diffs_global = old_used
                selected.pop()
                
            return False

        backtrack_limited(0, 0)

        if best_rows:
            # We found a solution. Can we do better?
            # The outer loop will continue if time permits, with new random candidates.
            # This allows us to potentially find a smaller scope.
            pass

    if best_rows:
        return best_rows
    else:
        # Fallback: Simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
