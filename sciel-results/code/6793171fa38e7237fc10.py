import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before the 150s kill

    # Problem parameters
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
                # We'll scan up to a reasonable limit to find valid candidates
                # For this problem, values rarely exceed 150
                limit = 200 
                while x <= limit and len(candidates) < window:
                    # Check if x is valid
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: extend search further
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
                
                # Select candidate
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                # Update used_diffs
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def count_conflicts(rows):
        """
        Counts the number of distinct difference collisions.
        A valid solution has 0 conflicts.
        """
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                # If a difference appears 'count' times, there are C(count, 2) pairs of collisions
                # But for local search, simple count of extra occurrences is a good proxy
                # Let's use sum(count - 1) or sum(count^2). 
                # sum(count - 1) is linear. sum(count^2) is quadratic (penalizes large clusters).
                # Let's use sum(count - 1) for simplicity, it's often effective.
                conflicts += count - 1
        return conflicts

    def local_search_cdvr(rows, time_limit):
        """
        Optimizes rows using Conflict-Directed Value Removal (CDVR) style local search.
        1. If conflicts > 0, find the "most critical" position (involved in most conflicts) and change it to a valid value if possible, or a value that reduces conflicts.
        2. If conflicts == 0, focus on reducing scope.
        
        Returns (best_rows, best_scope).
        """
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        
        # Track history of conflicts to detect stagnation
        stagnation_counter = 0
        last_conflict = current_conflicts
        
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        while time.time() < start_time + time_limit:
            if T < T_min:
                # Restart temperature if stuck?
                if stagnation_counter > 1000:
                    T = 5.0
                    stagnation_counter = 0
                else:
                    break
            
            # Choose a row and position to modify
            # Strategy: 
            # If we have conflicts, prioritize positions in rows that have conflicts.
            # Specifically, find the pair of elements (i, j) in a row that creates a difference d that is used by another row.
            # Then pick i or j to move.
            
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            
            # Determine if this row is involved in conflicts
            # For efficiency, let's just pick a random position first, 
            # but if we detect high conflict density, we can be smarter.
            # For now, simple random position.
            pos = random.randint(1, k) # 1 to k
            
            old_val = row[pos]
            
            # Determine new value
            # If we have conflicts, we want to pick a value that is NOT causing a conflict.
            # If we have no conflicts, we want to pick a value that reduces scope (or at least doesn't increase it too much).
            
            # Range for new value
            # Must be > row[pos-1] and < row[pos+1]
            # Or if pos is last, just > row[pos-1]
            # And we want to keep the row increasing.
            
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 10 # Allow some expansion if needed to resolve conflicts
            
            # If lower_bound > upper_bound, this position is stuck in terms of simple swaps?
            # Actually, we can expand the row if pos == k.
            # If pos < k, we are bounded.
            
            if lower_bound > upper_bound:
                # Skip this move
                continue
                
            # Candidate selection
            # If we have conflicts, we want to avoid values that create conflicts.
            # Let's sample a few candidates and pick the one with lowest "local conflict count".
            # Local conflict count: how many other rows use the differences generated by this new value?
            
            best_candidate = None
            best_candidate_score = float('inf')
            
            # Sample a few values
            # Bias towards lower values if we are trying to reduce scope (no conflicts)
            # Bias towards "valid" values if we have conflicts.
            
            num_samples = 15
            candidates = []
            
            # Generate candidates: mix of greedy (small values) and random
            # For scope reduction (no conflicts), bias heavily towards small values
            # For conflict resolution, use more random sampling
            
            if current_conflicts > 0:
                # More random sampling to escape local minima
                for _ in range(num_samples):
                    v = random.randint(lower_bound, upper_bound)
                    candidates.append(v)
            else:
                # Bias towards smaller values to reduce scope
                # 60% greedy (small values), 40% random
                for _ in range(num_samples):
                    if random.random() < 0.6:
                        # Pick from lower portion of range
                        v = lower_bound + random.randint(0, min(upper_bound - lower_bound, 20))
                    else:
                        v = random.randint(lower_bound, upper_bound)
                    candidates.append(v)
            
            # Deduplicate and ensure we have the current value removed
            candidates = list(set(candidates))
            if old_val in candidates:
                candidates.remove(old_val)
            
            if not candidates:
                continue
            
            # Pick the best candidate by evaluating local conflict impact
            # For each candidate, compute how many new conflicts it would create
            # This is approximate but much faster than full global recount
            
            best_v = None
            best_local_score = float('inf')
            
            # Build a set of all currently used diffs for quick lookup
            current_diffs = set()
            for r in current_rows:
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        current_diffs.add(r[j] - r[i])
            
            # Remove the old diffs that involve position pos in row r_idx
            for i in range(row_len):
                if i == pos:
                    continue
                d1 = abs(old_val - row[i])
                if d1 in current_diffs:
                    current_diffs.discard(d1)
            
            for v in candidates:
                # Compute new diffs involving position pos with value v
                new_diffs = set()
                for i in range(row_len):
                    if i == pos:
                        continue
                    d = abs(v - row[i])
                    new_diffs.add(d)
                
                # Count how many of these new diffs are already in current_diffs
                # (excluding the old diffs we removed)
                local_conflicts = sum(1 for d in new_diffs if d in current_diffs)
                
                # Also check if new_diffs have internal conflicts (shouldn't happen since row is increasing)
                # But let's be safe
                if len(new_diffs) != len(new_diffs):
                    local_conflicts += 1
                
                # Prefer lower values if no conflicts (to reduce scope)
                if current_conflicts == 0:
                    score = local_conflicts * 100 + v  # Heavy penalty for conflicts, then prefer small v
                else:
                    score = local_conflicts  # Just minimize conflicts
                
                if score < best_local_score:
                    best_local_score = score
                    best_v = v
            
            if best_v is None or best_v == old_val:
                continue
            
            v = best_v
            
            # Apply move and check global conflict count
            # We need to calculate the new conflict count efficiently?
            # Or just calculate it from scratch?
            # N=4, K=7. Total diffs 112.
            # Calculating conflict count from scratch is O(N * K^2) = 4 * 28 = 112 operations.
            # This is VERY fast. We can afford to calculate it from scratch every step.
            
            # Save old state
            old_val_saved = old_val
            row[pos] = v
            
            new_conflicts = count_conflicts(current_rows)
            new_scope = get_scope(current_rows)
            
            # Acceptance Criteria
            # 1. If new_conflicts < current_conflicts: Accept (and update best)
            # 2. If new_conflicts == current_conflicts:
            #      - If new_scope < current_scope: Accept
            #      - Else: Accept with probability exp(-delta_scope / T)
            # 3. If new_conflicts > current_conflicts:
            #      - Accept with probability exp(-delta_conflict / T)
            
            delta_conflict = new_conflicts - current_conflicts
            delta_scope = new_scope - current_scope
            
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_scope / T):
                        accept = True
            else:
                # Conflict increased
                if random.random() < math.exp(-delta_conflict / T):
                    accept = True
            
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                
                # Update best
                # We prefer valid solutions (conflicts == 0) with lowest scope.
                # If we have conflicts, we prefer lower conflict count, then lower scope.
                
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                    best_conflicts = current_conflicts
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    
                if current_conflicts < last_conflict:
                    last_conflict = current_conflicts
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
            else:
                # Revert
                row[pos] = old_val_saved
                stagnation_counter += 1
                
            T *= cooling
            
        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')
    
    # Multi-start
    # Allocate time for multiple restarts.
    # Total time ~135s.
    # Let's do restarts of ~10s each.
    # Leave 15s for final polish.
    
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Vary parameters
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)
        
        # Run CDVR for a short duration
        # Time limit for this CDVR: 10 seconds
        cdvr_time_limit = 10.0
        optimized_rows, opt_scope = local_search_cdvr(rows, cdvr_time_limit)
        
        # Re-evaluate conflicts for the optimized rows to be sure
        opt_conflicts = count_conflicts(optimized_rows)
        
        if opt_conflicts < best_conflicts or (opt_conflicts == best_conflicts and opt_scope < best_scope):
            best_conflicts = opt_conflicts
            best_scope = opt_scope
            best_rows = optimized_rows
            
        # If we found a valid solution (0 conflicts), we can break early if it's very good
        if best_conflicts == 0 and best_scope < 125:
            break
            
    # Phase 2: Final Polish
    # If we have time left, run CDVR on the best solution found so far
    if best_rows is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Run CDVR again on the best rows
            # We want it to run until `deadline - 2`
            # So time_limit = remaining_time - 2
            final_rows, final_scope = local_search_cdvr(best_rows, remaining_time - 2.0)
            
            final_conflicts = count_conflicts(final_rows)
            
            if final_conflicts < best_conflicts or (final_conflicts == best_conflicts and final_scope < best_scope):
                best_conflicts = final_conflicts
                best_scope = final_scope
                best_rows = final_rows

    if best_rows is None:
        # Fallback
        best_rows = greedy_init(0.1, 20)
        
    return best_rows
