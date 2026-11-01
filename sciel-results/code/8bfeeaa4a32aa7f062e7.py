import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # The evaluator kills at 150s. Use 145s as hard deadline for logic, 
    # but aim to return well before to avoid overhead.
    deadline = start_time + 140.0
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_init(rand_prob=0.1, window=20):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
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

    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        Returns rows if found, None otherwise.
        """
        random.seed(seed)
        start = time.time()
        
        # State
        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()
        
        # We will build row by row.
        # To allow randomization in value selection, we'll collect a few valid candidates.
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == row_len:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1]
            # Lower bound is prev_val + 1
            # Upper bound is target_scope
            # Note: We don't strictly need to check upper bound against next element 
            # because we build left to right, and we will enforce increasing order 
            # in the next step. However, we must ensure we don't exceed target_scope.
            
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            # Collect valid candidates
            # We bias towards smaller values to keep scope low, 
            # but we take a random sample to escape local minima.
            # Strategy: Find the first 5-10 valid values, pick one randomly.
            # If no valid values in [lower, lower+50], expand search.
            
            candidates = []
            # Scan a window first
            scan_limit = min(upper, lower + 50)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                # If none found in the small window, scan the rest
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 10:
                            break
            
            if not candidates:
                return False
            
            # Shuffle candidates to randomize the search path
            # But we want to bias towards smaller values.
            # We can sort and then pick a random one from the first half, or just shuffle.
            # Shuffling is simpler and explores more.
            random.shuffle(candidates)
            
            for val in candidates:
                # Check validity again (in case we picked a bad one? No, we pre-validated)
                # But we need to check against ALL previous elements in the row, not just the immediate previous.
                # The pre-validation loop above did `for i in range(pos)`, so it's correct.
                
                # Add differences
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                # Backtrack
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Phase 1: Get a baseline
    best_rows = greedy_init(0.1, 20)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Iterative Improvement using Backtracking
    # We try to reduce the scope by 1 each time.
    # If backtracking fails, we might try a "jolt" to the current best and retry,
    # but since backtracking is global (builds from scratch), it doesn't depend on previous rows' specific values, 
    # only on the target scope. 
    # Wait, the backtracking search builds ALL rows from scratch. 
    # This is good because it finds a globally consistent set for the target scope.
    # It doesn't get stuck in "bad row 0" local minima because it backtracks row 0 too.
    
    # However, pure backtracking might be slow if the target scope is tight.
    # We will allocate time to each attempt.
    
    current_target = best_scope - 1
    seed_counter = 0
    
    while time.time() < deadline - 5:
        if current_target < 112: # Theoretical bound
            break
            
        # Allocate time for this search attempt
        # Start with 2s, increase if we are close to the best?
        # Let's use a fixed chunk of time, e.g., 5s, or adaptive.
        # If we find a solution, we reduce target by 1 and continue.
        # If we fail, we might try a slightly higher target? No, we want to minimize.
        # If we fail, we should probably stop trying to reduce by 1 and try to see if we can find ANY solution 
        # with a slightly higher scope to confirm validity? No, we already have a valid one.
        
        # The risk is that backtracking is too slow for tight scopes.
        # We will run the search with a time limit.
        search_time = 5.0
        if time.time() > deadline - 10:
            search_time = deadline - time.time() - 2
        
        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1
        
        if result:
            new_scope = get_scope(result)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = result
                current_target = best_scope - 1
            else:
                # Should not happen if search is correct, but just in case
                break
        else:
            # Failed to find solution with current_target
            # This means current_target is likely too low.
            # We should stop trying to reduce further.
            # However, due to randomness, we might have failed by bad luck.
            # We can retry once with a different seed.
            if time.time() < deadline - 10:
                result_retry = backtracking_search(n_rows, k, current_target, 5.0, seed=seed_counter)
                seed_counter += 1
                if result_retry:
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    # Confirmed failure
                    break
            else:
                break

    return best_rows

# Fallback in case construct is called with unexpected parameters or errors
if __name__ == "__main__":
    # Simple test
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    # Verify
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
