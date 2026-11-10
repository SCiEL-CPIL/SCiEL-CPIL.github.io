import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound estimate: sqrt(2 * total_diffs)
    # This is a rough estimate. The actual scope M must satisfy M*(M+1)/2 >= total_diffs
    # So M >= sqrt(2 * total_diffs)
    import math
    lower_bound = int(math.ceil(math.sqrt(2 * total_diffs)))
    
    best_solution = None
    best_score = float('inf')
    
    def validate(rows):
        """Check if rows are valid: all positive differences distinct across all rows."""
        all_diffs = set()
        for row in rows:
            # Must start at 0 and be strictly increasing
            if not row or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    def get_score(rows):
        return max(max(row) for row in rows)

    def try_backtracking(max_scope, seed):
        """
        Attempt to find a solution with max entry <= max_scope.
        Uses backtracking on the difference sets.
        """
        random.seed(seed)
        
        # State:
        # used_diffs: set of differences already used
        # current_rows: list of lists, each list is a set of differences for that row
        # We don't construct the row elements directly, but the set of differences.
        # A valid row of size k+1 starting at 0 corresponds to a set D of differences
        # such that there exists a set A = {a_0=0, ..., a_k} with A-A = D U {-D} U {0}.
        # This is equivalent to A being a Sidon set (B2 sequence) of size k+1?
        # No, Sidon set means all sums a_i+a_j (i<j) are distinct.
        # Here we need all differences a_j - a_i (i<j) to be distinct globally.
        # Within a row, if differences are distinct, it's a Golomb ruler.
        # So each row is a Golomb ruler of length k (k+1 marks).
        # And the sets of differences from different rows must be disjoint.
        
        # Strategy:
        # 1. Generate a pool of candidate Golomb rulers of length k with max mark <= max_scope.
        # 2. Select n_rows disjoint rulers from this pool.
        
        # Generating all Golomb rulers up to max_scope might be expensive if max_scope is large.
        # But max_scope is likely small (around lower_bound).
        # For k=7, max_scope ~ 112-134.
        # Number of Golomb rulers of length 7 is not huge?
        # Actually, we can generate them on the fly or use a pre-computed list if possible.
        # Since we can't pre-compute for arbitrary k, we must generate.
        
        # Alternative: Backtracking on the rows directly, but choosing the "next difference"
        # globally.
        
        # Let's stick to the "Difference Set Packing" idea.
        # We need to assign each of the total_diffs differences to a row.
        # But the differences in a row are not independent; they must form a valid Golomb ruler.
        
        # Heuristic:
        # 1. Find a valid Golomb ruler for row 0.
        # 2. Find a valid Golomb ruler for row 1 disjoint from row 0.
        # ...
        # This is what the current program does, but greedily.
        # Backtracking allows us to undo choices if we get stuck.
        
        # However, backtracking over the entire row construction is expensive.
        # Let's try a different angle:
        # Randomized Local Search / Simulated Annealing on the set of rows?
        # Start with a valid set of rows (from greedy).
        # Move: Pick a row, remove it, re-insert a different valid Golomb ruler.
        # Objective: Minimize max entry.
        
        # Let's implement a simple Simulated Annealing.
        
        # 1. Generate initial solution using greedy.
        def greedy_solve():
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                rows.append(row)
            return rows, used

        try:
            current_rows, current_used = greedy_solve()
            current_max = max(max(r) for r in current_rows)
        except:
            return None

        if current_max <= max_scope:
            return current_rows

        # Simulated Annealing
        # We need a way to generate a random Golomb ruler disjoint from a set of used diffs.
        def generate_random_ruler(used_diffs, max_val, seed_val):
            """
            Generate a random Golomb ruler of length k (k+1 marks) 
            with max mark <= max_val, disjoint from used_diffs.
            """
            local_rng = random.Random(seed_val)
            attempts = 0
            max_attempts = 1000
            while attempts < max_attempts:
                attempts += 1
                # Try to build a ruler
                row = [0]
                last = 0
                success = True
                local_used = set()
                # We must ensure new diffs don't conflict with global used_diffs
                # And don't conflict with each other within the row.
                
                # To make it random, we can pick the next mark from a random valid range
                # or use a randomized greedy.
                
                # Let's use randomized greedy:
                # At each step, pick the smallest valid x with prob p, 
                # or a random valid x in [last+1, last+window] with prob 1-p.
                
                while len(row) < k + 1:
                    # Find valid options
                    valid_options = []
                    # Scan from last+1 up to max_val
                    # Optimization: only scan up to a certain limit?
                    # If we scan up to max_val, it might be slow.
                    # But max_val is small.
                    
                    # To speed up, we can check validity quickly.
                    # new_diffs = {x - a for a in row}
                    # Must be disjoint from used_diffs AND local_used.
                    
                    # We'll scan x from last+1 to max_val.
                    # If we find many, we pick randomly.
                    # If we find none, fail.
                    
                    found = False
                    for x in range(last + 1, max_val + 1):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs) and not (new_diffs & local_used):
                            # Valid
                            if not found:
                                valid_options = [x]
                                found = True
                            else:
                                # Limit options to avoid huge lists
                                if len(valid_options) < 20:
                                    valid_options.append(x)
                                else:
                                    # If we have enough, we can break?
                                    # No, we want to explore larger x too for diversity?
                                    # Actually, for minimizing max, smaller is better.
                                    # So we prefer smaller.
                                    # Let's just collect the first 20 valid ones.
                                    pass
                    
                    if not found:
                        success = False
                        break
                    
                    # Choose from valid_options
                    # Bias towards smaller
                    if local_rng.random() < 0.7:
                        next_x = valid_options[0]
                    else:
                        next_x = local_rng.choice(valid_options)
                    
                    # Update
                    row.append(next_x)
                    new_diffs = {next_x - a for a in row[:-1]}
                    local_used.update(new_diffs)
                    last = next_x
                
                if success:
                    # Check if max(row) <= max_val (it is, by construction)
                    # And disjoint from used_diffs (checked)
                    return row
            
            return None

        # SA Loop
        T = 10.0
        cooling = 0.99
        best_rows = current_rows
        best_max = current_max
        current_max_val = current_max
        
        # We need to be able to remove a row and re-add it.
        # used_diffs must be updated.
        
        steps = 0
        max_steps = 5000
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            
            # Pick a random row to modify
            idx = random.randint(0, n_rows - 1)
            old_row = current_rows[idx]
            
            # Remove old row from used_diffs
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            current_used.difference_update(old_diffs)
            
            # Try to generate a new row for this slot
            # We want to keep the max scope small.
            # Try to generate a row with max <= current_max_val
            new_row = generate_random_ruler(current_used, current_max_val, seed_val=steps + 1000)
            
            if new_row is None:
                # Restore old row
                current_used.update(old_diffs)
                current_rows[idx] = old_row
                continue
            
            # Check if new row improves or maintains max
            new_max = max(new_row)
            
            # Calculate energy difference
            # We want to minimize max(max(rows)).
            # If new_max < current_max_val, it's an improvement.
            # If new_max == current_max_val, it's neutral (but might help future moves).
            # If new_max > current_max_val, it's worse.
            
            # But we also need to ensure the set of rows is valid (disjoint).
            # The generation ensures disjointness from current_used (which excludes old_row).
            # So the new set is valid.
            
            # Acceptance criterion
            accept = False
            if new_max < current_max_val:
                accept = True
            elif new_max == current_max_val:
                # Accept with some probability to escape local minima?
                # Or always accept to explore?
                # Let's accept with prob 0.5
                accept = random.random() < 0.5
            else:
                # Worse
                delta_E = new_max - current_max_val
                if random.random() < math.exp(-delta_E / T):
                    accept = True
            
            if accept:
                # Commit change
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                
                current_used.update(new_diffs)
                current_rows[idx] = new_row
                current_max_val = new_max
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                    
                # Cool down
                T *= cooling
            else:
                # Reject, restore
                current_used.update(old_diffs)
                current_rows[idx] = old_row
                T *= cooling
        
        if best_max <= max_scope:
            return best_rows
        else:
            return None

    # Run backtracking/SA with increasing max_scope
    # Start from lower_bound
    current_scope = lower_bound
    seed = 42
    
    while time.time() < deadline:
        # Try to find a solution with this scope
        result = try_backtracking(current_scope, seed)
        
        if result is not None:
            # Validate
            if validate(result):
                score = get_score(result)
                if score < best_score:
                    best_score = score
                    best_solution = result
                    # If we hit the lower bound, we can't do better
                    if score <= lower_bound:
                        return result
                # If we found a solution, we can stop searching for larger scopes
                # unless we want to try to improve further?
                # The task is to minimize the largest entry.
                # If we found a solution with scope S, we can return it.
                # But maybe a smaller scope exists?
                # We are iterating scope from low to high.
                # So the first valid solution found is the best possible for this scope.
                # But wait, try_backtracking returns a solution if one exists for that scope.
                # So if it returns non-None, we have a solution with max <= current_scope.
                # Since we start from lower_bound, this is the best possible.
                return result
        else:
            # No solution found for this scope.
            # Increase scope and try again.
            current_scope += 1
            seed += 1
            
            # Safety: don't go too high
            if current_scope > 500:
                break

    if best_solution is not None:
        return best_solution
    
    # Fallback to deterministic greedy if SA failed
    def greedy_deterministic():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return greedy_deterministic()
