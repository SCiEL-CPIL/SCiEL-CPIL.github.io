"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Strategy: Simulated Annealing / Iterated Local Search.
    1. Initialize with a greedy valid solution.
    2. Iteratively pick a random row, remove it, and re-generate it 
       attempting to minimize its maximum element (and thus potentially the global scope).
    3. Accept worse solutions with a temperature-dependent probability to escape local minima.
    """
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    # Helper to generate a single row given the set of already used differences.
    # Returns the row list or None if failed (should not happen if we are careful, 
    # but we might hit a dead end if the space is too constrained for this specific row 
    # given the others. However, since we are removing a row that *was* there, 
    # a valid replacement *should* exist (the original one), so we can always 
    # fall back to the original if greedy fails to find a better one quickly, 
    # or just return the original if we want to guarantee validity).
    # Actually, for SA, we want to try to find a *better* one. If we can't find 
    # a better one quickly, we might just keep the old one.
    
    def generate_row(k, used_diffs, max_search, randomize):
        """
        Greedily construct a row of k+1 elements starting at 0.
        used_diffs: set of differences already taken by other rows.
        max_search: upper bound for candidate values.
        randomize: if True, adds some randomness to candidate selection.
        Returns: list of k+1 ints, or None if failed.
        """
        row = [0]
        for _ in range(k):
            last_val = row[-1]
            found = False
            # Search for next element
            # We want small values to keep scope low.
            # Candidates are > last_val.
            # Check diffs: cand - x for all x in row.
            
            # To allow for "randomness" in SA, we can shuffle the first N candidates
            # or pick randomly from a window.
            
            # Simple greedy with random tie-breaking among early candidates
            candidates = []
            x = last_val + 1
            # Limit search window to keep it fast
            # If we are re-generating, we know a solution exists (the old one), 
            # so we don't need to search too far, but we might need to go 
            # further than the old max if we want to "swap" differences?
            # No, we want to *minimize* scope. So we search from small to large.
            
            while x <= max_search:
                # Check if x is valid
                valid = True
                for prev in row:
                    d = x - prev
                    if d in used_diffs:
                        valid = False
                        break
                
                if valid:
                    if randomize:
                        # Small chance to skip this valid candidate to explore others?
                        # Or just pick from a small window of valid candidates.
                        # Let's just pick the first valid one for determinism in "greedy" 
                        # but if we want SA, we need variation.
                        # Variation strategy: 
                        # 1. Find first valid candidate 'c'.
                        # 2. Look ahead a few steps. If another valid 'c2' is close, 
                        #    maybe pick it?
                        # Simpler: Shuffle the first 5 valid candidates and pick one.
                        
                        valid_cands = []
                        # We found the first one. Let's find a few more.
                        valid_cands.append(x)
                        x_tmp = x + 1
                        while x_tmp <= max_search and len(valid_cands) < 5:
                            valid2 = True
                            for prev in row:
                                if (x_tmp - prev) in used_diffs:
                                    valid2 = False
                                    break
                            if valid2:
                                valid_cands.append(x_tmp)
                            x_tmp += 1
                            
                        if len(valid_cands) > 1 and random.random() < 0.5:
                            # Pick a random one from the first few
                            x = random.choice(valid_cands)
                        else:
                            x = valid_cands[0]
                    else:
                        x = x # Use the first found
                    
                    row.append(x)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        return row

    # 1. Initialize
    # Use a robust greedy to get a starting point.
    # We'll run the "old" logic once to get a baseline.
    
    def init_greedy():
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used

    initial_rows, initial_used = init_greedy()
    current_rows = [r[:] for r in initial_rows]
    current_used = set(initial_used)
    current_scope = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # 2. Simulated Annealing Loop
    T = 10.0  # Initial temperature
    cooling = 0.995
    min_T = 0.1
    
    # Estimate max value for search
    # The scope is roughly sqrt(total_diffs). 
    # Let's set max_search to best_scope + 10
    max_search = best_scope + 20
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Pick a random row to modify
        r_idx = random.randint(0, n_rows - 1)
        old_row = current_rows[r_idx]
        
        # Remove old row's differences
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i):
                old_diffs.add(old_row[i] - old_row[j])
        
        for d in old_diffs:
            current_used.discard(d)
        
        # Try to generate a new row
        # We want to minimize the max element of this row.
        # We generate a few candidates and pick the best one (smallest max)
        # that is valid.
        
        best_new_row = None
        best_new_max = float('inf')
        
        # Try a few randomizations
        for _ in range(5):
            new_row = generate_row(k, current_used, max_search, randomize=True)
            if new_row:
                new_max = new_row[-1]
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_new_row = new_row
        
        # If we didn't find a better row, maybe try the original one?
        # Or just keep the original if no improvement.
        # But for SA, we might accept a worse row to escape a local minimum
        # where the *global* scope is determined by this row, but changing it 
        # slightly might allow *other* rows to shrink? 
        # No, rows are independent except for the `used` set.
        # If we change row r, it only affects the `used` set for *other* rows 
        # in subsequent iterations. It doesn't change other rows immediately.
        # So, if we make row r *larger*, it uses larger differences, 
        # which might *free up* smaller differences for other rows?
        # Yes! If row r currently uses small differences 1, 2, 3...
        # and we make it use 10, 11, 12...
        # then 1, 2, 3 become available for other rows, potentially allowing them to shrink.
        # So we MUST allow "worse" moves for individual rows.
        
        if best_new_row is None:
            # Fallback: restore old row
            current_rows[r_idx] = old_row
            current_used |= old_diffs
            continue
        
        # Calculate new global scope if we swap in best_new_row
        # Note: best_new_row is the best *individual* row found.
        # But maybe a *worse* individual row (larger max) is better for the *global* scope?
        # Unlikely to be better for global scope immediately, but might help later.
        # For now, let's just evaluate the swap of the best individual row.
        
        # Actually, to allow exploration, let's pick a random candidate from the 5 generated,
        # not just the best one.
        # Let's re-generate 5, store them, and pick one.
        
        candidates = []
        for _ in range(5):
            new_row = generate_row(k, current_used, max_search, randomize=True)
            if new_row:
                candidates.append(new_row)
        
        if not candidates:
            # Should not happen if old row was valid and we have space
            current_rows[r_idx] = old_row
            current_used |= old_diffs
            continue
        
        # Pick a candidate. 
        # Bias towards smaller max, but allow larger ones.
        # Sort by max element
        candidates.sort(key=lambda r: r[-1])
        
        # With probability p, pick the best. With 1-p, pick a random one.
        if random.random() < 0.7:
            chosen_row = candidates[0]
        else:
            chosen_row = random.choice(candidates)
        
        # Calculate new scope
        # We need to know the max of other rows.
        # To avoid O(N) calculation every time, we can maintain a heap or just compute it.
        # N is small (4-10?), so O(N) is fine.
        other_maxes = [max(row) for i, row in enumerate(current_rows) if i != r_idx]
        new_global_scope = max(chosen_row[-1], max(other_maxes) if other_maxes else 0)
        
        delta = new_global_scope - current_scope
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_rows[r_idx] = chosen_row
            # Update used diffs
            new_diffs = set()
            for i in range(len(chosen_row)):
                for j in range(i):
                    new_diffs.add(chosen_row[i] - chosen_row[j])
            current_used |= new_diffs
            current_scope = new_global_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in current_rows]
        else:
            # Reject: restore old row
            current_rows[r_idx] = old_row
            current_used |= old_diffs
        
        # Cool down
        T = max(min_T, T * cooling)
        
        # If T is very low, we might want to restart or just continue.
        # If we hit the lower bound, break.
        # Lower bound: total_diffs
        total_diffs = n_rows * k * (k + 1) // 2
        if best_scope == total_diffs:
            break
    
    return best_rows
# EVOLVE-BLOCK-END
