import time
import random
import numpy as np
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Lower bound estimate
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = int(math.ceil(math.sqrt(2 * total_diffs)))
    
    # If the problem is trivial, just return a known small solution if possible
    # For n_rows=4, k=7, lower_bound is 112. Best known is 134.
    
    # We will use a global set of used differences for fast checking
    # But for vectorized SA, we need a different approach.
    
    # Strategy:
    # 1. Generate a valid initial solution using a fast greedy method.
    # 2. Use Simulated Annealing where we propose a new row for a random index.
    # 3. To speed up, we will batch-generate many candidate rows and check them.
    #    However, checking disjointness with the *other* rows is the bottleneck.
    
    # Let's refine: 
    # The constraint is that all differences in all rows are distinct.
    # This is equivalent to finding a set of n_rows Golomb rulers (length k) 
    # such that their difference sets are pairwise disjoint.
    
    # We will maintain:
    # rows: list of lists of ints
    # used_diffs: set of all differences used so far (excluding the row being modified)
    
    # To make SA faster, we can pre-compute a pool of "good" candidate rows 
    # that are small (low max entry) and have few "conflicts" with common patterns?
    # No, conflicts depend on the specific used_diffs.
    
    # Alternative Vectorized Approach:
    # Represent the state as a 2D array of row values.
    # Objective: Minimize max(rows).
    # Constraint: All pairwise differences are unique.
    
    # Let's stick to a highly optimized sequential SA but with a "Candidate Pool" strategy.
    # Generate a large pool of random Golomb rulers with small max values.
    # For each step, pick a random row to replace.
    # Check which pool candidates are valid (disjoint from used_diffs).
    # This allows us to try many candidates per step without generating them from scratch.
    
    # Step 1: Generate initial solution
    def greedy_initial():
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

    rows, used_diffs = greedy_initial()
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Step 2: Generate a pool of candidate rows
    # We want candidates with max value close to lower_bound or current_max.
    # Let's generate candidates in a range [lower_bound, current_max + 10]
    
    def generate_random_ruler(max_val, rng):
        """
        Generate a random Golomb ruler of length k (k+1 marks) with max <= max_val.
        Returns None if failed.
        """
        row = [0]
        last = 0
        local_used = set()
        
        # To make it random but efficient, we try a few strategies
        # 1. Greedy with randomization
        # 2. Backtracking with random choice
        
        # We'll use a randomized greedy that picks from a window of valid options
        for _ in range(k):
            valid_options = []
            # Scan from last+1 to max_val
            # Optimization: If we can't find a valid option in the first 50, break early?
            # No, we must check all up to max_val to ensure we don't miss a valid small one.
            # But max_val is small (~134). So scanning is fast.
            
            found = False
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used):
                    # Check against global used? No, this function only generates a valid Golomb ruler.
                    # The disjointness check happens later.
                    # Wait, if we generate a ruler that conflicts with global used, it's useless.
                    # But we don't know global used inside this function if we want it generic.
                    # So we just generate a valid Golomb ruler.
                    valid_options.append(x)
                    if len(valid_options) > 50:
                        break # Enough options, pick one
                    found = True
            if not found:
                return None
            
            # Pick randomly from options, bias towards smaller
            if rng.random() < 0.8:
                next_x = valid_options[0]
            else:
                next_x = rng.choice(valid_options)
            row.append(next_x)
            local_used.update({next_x - a for a in row[:-1]})
            last = next_x
            
        return row

    # Pre-generate a pool of candidates
    # We want candidates that are "good" (small max).
    # Let's generate 1000 candidates with max_val = current_max
    # And 1000 with max_val = lower_bound (if current_max > lower_bound)
    
    pool = []
    rng = random.Random(42)
    
    # Generate candidates for the current max scope
    target_scope = current_max
    for _ in range(2000):
        r = generate_random_ruler(target_scope, rng)
        if r:
            # Calculate diffs for quick checking later
            # Actually, we can just store the row. We'll compute diffs when checking.
            pool.append(r)
            
    # If we are above lower bound, try to generate some at lower bound
    if current_max > lower_bound:
        for _ in range(2000):
            r = generate_random_ruler(lower_bound, rng)
            if r:
                pool.append(r)
                
    # If pool is empty, we are in trouble, but greedy should have worked.
    
    # Simulated Annealing Loop
    T = 10.0
    cooling = 0.995
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Pick a random row to replace
        idx = rng.randint(0, n_rows - 1)
        old_row = rows[idx]
        
        # Compute old diffs
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        # Remove old diffs from global set
        used_diffs.difference_update(old_diffs)
        
        # Try to find a valid candidate from the pool
        # We want a candidate that is disjoint from used_diffs
        # And ideally has a smaller max value
        
        best_candidate = None
        best_candidate_max = float('inf')
        
        # Iterate through pool, check validity
        # To speed this up, we can shuffle the pool or pick random samples
        # If we iterate all 4000 candidates, and each check takes ~28 set ops (k=7 -> 28 diffs),
        # 4000 * 28 = 112,000 ops per step. 50,000 steps = 5.6 billion ops. Too slow.
        
        # Optimization: Only check a random subset of the pool, or use a faster check.
        # Let's check a random sample of 50 candidates per step.
        
        sample_size = min(50, len(pool))
        indices = rng.sample(range(len(pool)), sample_size)
        
        found_valid = False
        for i in indices:
            cand = pool[i]
            cand_max = max(cand)
            
            # Quick reject if cand_max > best_candidate_max (we want improvement or neutral)
            # Actually, SA allows worse moves.
            
            # Check disjointness
            valid = True
            # Compute diffs of candidate
            cand_diffs = set()
            for p in range(len(cand)):
                for q in range(p + 1, len(cand)):
                    d = cand[q] - cand[p]
                    if d in used_diffs:
                        valid = False
                        break
                    cand_diffs.add(d)
            if not valid:
                continue
                
            # It's valid!
            # Compare with current best candidate in this step
            # We want to minimize max(cand)
            if cand_max < best_candidate_max:
                best_candidate_max = cand_max
                best_candidate = cand
                # We can break if we find one with max == lower_bound?
                # Or just keep looking for a slightly better one?
                # Let's keep looking in the sample to see if there's a smaller one.
        
        if best_candidate is None:
            # No valid candidate found in sample
            # Restore old row
            used_diffs.update(old_diffs)
            # T *= cooling # Don't cool on failed move? Or do?
            continue
            
        # Acceptance Criterion
        new_max = best_candidate_max
        old_max = current_max
        
        # Calculate energy diff
        # Note: current_max is the max over ALL rows.
        # If we replace row idx, the new global max is max(max(rows excluding idx), new_max)
        
        # Calculate max of other rows
        other_max = 0
        for j in range(n_rows):
            if j != idx:
                if max(rows[j]) > other_max:
                    other_max = max(rows[j])
        
        new_global_max = max(other_max, new_max)
        old_global_max = current_max
        
        delta_E = new_global_max - old_global_max
        
        accept = False
        if delta_E < 0:
            accept = True
        elif delta_E == 0:
            accept = rng.random() < 0.5
        else:
            if T > 0.1: # Avoid overflow/underflow issues
                prob = math.exp(-delta_E / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            # Commit
            rows[idx] = best_candidate
            # Update used_diffs
            cand_diffs = set()
            for p in range(len(best_candidate)):
                for q in range(p + 1, len(best_candidate)):
                    cand_diffs.add(best_candidate[q] - best_candidate[p])
            used_diffs.update(cand_diffs)
            
            current_max = new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                
            T *= cooling
        else:
            # Reject, restore
            used_diffs.update(old_diffs)
            T *= cooling
            
        # Periodically refresh the pool if we are stuck?
        # If steps % 1000 == 0 and current_max > lower_bound:
        #     Generate new pool entries with max = current_max - 1?
        #     Or just add more random ones.
        
        if steps % 500 == 0:
            # Add some new candidates to keep the pool fresh
            # Focus on the current best max or slightly lower
            target = best_max if best_max > lower_bound else lower_bound
            for _ in range(100):
                r = generate_random_ruler(target, rng)
                if r:
                    pool.append(r)
            # Limit pool size to avoid memory bloat
            if len(pool) > 5000:
                pool = pool[-2500:]

    # Final Validation and Return
    # Ensure best_rows is valid
    def validate(rows_to_check):
        all_diffs = set()
        for row in rows_to_check:
            if not row or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    if validate(best_rows):
        return best_rows
    
    # Fallback
    if validate(rows):
        return rows
        
    # Last resort: Greedy
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
