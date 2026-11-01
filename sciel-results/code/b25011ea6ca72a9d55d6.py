import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: 150s limit, leave 10s for overhead/return
    deadline = start_time + 140.0
    
    # Problem parameters
    n_elements = k + 1
    total_diffs_per_row = n_elements * (n_elements - 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    # 1. Initialization: Greedy with a wider search window to avoid early traps
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < n_elements:
                # Search in a window to allow some flexibility, not just next int
                found = False
                # Start from current_max + 1, go up to current_max + 50
                for x in range(current_max + 1, current_max + 50):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    # Check internal uniqueness of new diffs
                    if valid and len(set(new_diffs)) != len(new_diffs):
                        valid = False
                    
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                
                if not found:
                    # Fallback: force it, though this creates conflicts
                    # In a proper init, we should restart or widen window
                    x = current_max + 1
                    while True:
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                # Allow conflict in init to ensure we have a solution structure, 
                                # but this is bad. Better to just pick smallest unused.
                                pass
                            new_diffs.append(d)
                        # Just pick the smallest x that doesn't duplicate internal diffs
                        if len(set(new_diffs)) == len(new_diffs):
                             row.append(x)
                             used_diffs.update(new_diffs)
                             current_max = x
                             break
                        x += 1
            rows.append(row)
        return rows

    def get_scope(rows):
        return max(max(r) for r in rows)

    def evaluate_state(rows):
        """
        Returns:
        conflicts: number of duplicate differences (excess count)
        scope: max element
        conflict_map: dict mapping (row_idx, pos_idx) to count of conflicts involving that position
        """
        all_diffs = []
        # Store diffs with their origin to map back to positions
        # diff (d) comes from row i, pos j (as the larger element) and some pos < j
        # Actually, a difference d = a_j - a_i involves positions i and j.
        # If d is duplicated, both positions i and j in row row_idx are "conflicted".
        
        diff_counter = Counter()
        diff_origins = {} # diff -> list of (row_idx, pos_i, pos_j)
        
        for r_idx, row in enumerate(rows):
            for i in range(n_elements):
                for j in range(i + 1, n_elements):
                    d = row[j] - row[i]
                    diff_counter[d] += 1
                    if d not in diff_origins:
                        diff_origins[d] = []
                    diff_origins[d].append((r_idx, i, j))
        
        total_conflicts = 0
        conflict_map = {} # (r_idx, pos) -> count of conflicts involving this pos
        
        for d, count in diff_counter.items():
            if count > 1:
                excess = count - 1
                total_conflicts += excess
                # Distribute conflict blame to all positions involved in this diff
                # This is a heuristic: if a diff is duplicated, all positions generating it are "bad"
                for (r_idx, i, j) in diff_origins[d]:
                    key1 = (r_idx, i)
                    key2 = (r_idx, j)
                    conflict_map[key1] = conflict_map.get(key1, 0) + 1
                    conflict_map[key2] = conflict_map.get(key2, 0) + 1
        
        scope = get_scope(rows)
        return total_conflicts, scope, conflict_map

    # 2. Pre-compute candidate pools
    # For each row position (r, p), what are the "small" candidates?
    # Since we want to minimize scope, we prefer small values.
    # We will generate candidates dynamically based on current scope during search.
    
    def find_best_replacement(rows, r_idx, p_idx, current_scope_limit):
        """
        Find the smallest value x > rows[r_idx][p_idx-1] (and < rows[r_idx][p_idx+1] if p_idx < k)
        such that replacing rows[r_idx][p_idx] with x reduces or maintains conflicts,
        and ideally reduces scope or keeps it within limit.
        """
        row = rows[r_idx]
        prev_val = row[p_idx - 1] if p_idx > 0 else -1
        next_val = row[p_idx + 1] if p_idx < k else current_scope_limit + 100
        
        # We want to try small values first
        # Range: from prev_val + 1 to min(next_val - 1, current_scope_limit)
        # If p_idx is the last one, we really want to shrink it, so we try down to prev_val+1
        
        # Get current diffs of this row excluding those involving p_idx
        # This is expensive if done naively, but k=7 is small.
        
        # Let's just try a few candidates
        # Candidate 1: The current value (no change) - not useful for search
        # Candidate 2: Smallest valid value in range
        
        # To speed up, we check validity against the *global* used diffs
        # But global used diffs change if we change this row.
        # Simplification: We check if the new value creates *new* conflicts or resolves existing ones.
        
        # For CDVR, we pick the value that minimizes the "conflict cost" locally.
        # Let's just try the smallest few integers in the valid structural range.
        
        candidates = []
        # Try to shrink if possible
        if p_idx == k:
             # Last element, try to make it smaller
             # Range: [row[k-1] + 1, row[k]]
             # We want the smallest one that doesn't create internal dupes
             for x in range(row[p_idx-1] + 1, row[p_idx] + 1):
                 candidates.append(x)
        else:
             # Middle element, try to keep it small or shift slightly
             # Range: [row[p_idx-1] + 1, row[p_idx+1] - 1]
             # We prefer values close to the current one or smaller
             current = row[p_idx]
             # Try current-1, current, current+1, ... up to next-1
             # And also try smaller ones if they fit
             for x in range(row[p_idx-1] + 1, min(row[p_idx+1], current_scope_limit + 1)):
                 candidates.append(x)
        
        # Sort candidates: prefer smaller values (to minimize scope)
        # But also prefer values that are "valid" internally
        # We'll just return the list and let the caller pick the best one based on global evaluation
        # To avoid huge lists, limit candidates
        # Actually, for k=7, the range is small.
        return candidates

    # 3. Main CDVR Loop
    
    # Initialize
    rows = greedy_init()
    
    # Initial Evaluation
    current_conflicts, current_scope, current_conflict_map = evaluate_state(rows)
    best_conflicts = current_conflicts
    best_scope = current_scope
    best_rows = [r[:] for r in rows]
    
    # If already valid, we might just return, but we want to minimize scope.
    # CDVR helps with conflicts. For scope minimization, we can have a secondary mode.
    
    stagnation_counter = 0
    
    while time.time() < deadline:
        if current_conflicts == 0:
            # If no conflicts, we want to minimize scope.
            # Modify objective: try to reduce the max element.
            # Find the row with the max element and try to shrink its last element.
            max_row_idx = -1
            max_val = -1
            for i, r in enumerate(rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            
            # Try to shrink max_row_idx's last element
            improved = False
            r_idx = max_row_idx
            p_idx = k
            row = rows[r_idx]
            
            # Try values from row[k-1]+1 to row[k]-1
            for x in range(row[k-1] + 1, row[k]):
                # Check if x is structurally valid (it is, by range)
                # Check if x creates internal duplicate diffs
                # Check if x creates global duplicate diffs
                
                # Temporarily set
                old_val = row[k]
                row[k] = x
                
                # Quick check: internal diffs
                internal_diffs = set()
                valid_internal = True
                for i in range(k+1):
                    for j in range(i+1, k+1):
                        d = row[j] - row[i]
                        if d in internal_diffs:
                            valid_internal = False
                            break
                        internal_diffs.add(d)
                if not valid_internal:
                    row[k] = old_val
                    continue
                
                # Check global conflicts
                # Remove old diffs involving pos k
                # Add new diffs involving pos k
                # This is complex to do incrementally without full re-eval.
                # Since k=7, full re-eval is fast enough? 
                # 4 rows * 28 diffs = 112 diffs. Counter is fast.
                
                new_conflicts, new_scope, _ = evaluate_state(rows)
                
                if new_conflicts == 0 and new_scope < current_scope:
                    current_scope = new_scope
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    improved = True
                    break
                else:
                    row[k] = old_val
            
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                # If stuck at scope minimum for a while, kick
                if stagnation_counter > 50:
                    # Random kick: change a random element slightly
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k) # Don't touch 0
                    row = rows[r_idx]
                    old_val = row[p_idx]
                    # Try a random nearby value
                    delta = random.randint(-2, 2)
                    new_val = old_val + delta
                    # Ensure structural validity
                    if p_idx > 0 and new_val <= row[p_idx-1]:
                        new_val = row[p_idx-1] + 1
                    if p_idx < k and new_val >= row[p_idx+1]:
                        new_val = row[p_idx+1] - 1
                    
                    if new_val != old_val and new_val > row[p_idx-1] and (p_idx == k or new_val < row[p_idx+1]):
                        row[p_idx] = new_val
                        # Re-evaluate
                        current_conflicts, current_scope, current_conflict_map = evaluate_state(rows)
                        if current_conflicts > 0:
                             # We broke validity, CDVR will fix it
                             pass
                        else:
                             if current_scope < best_scope:
                                 best_scope = current_scope
                                 best_rows = [r[:] for r in rows]
                    stagnation_counter = 0
            continue
        
        # If conflicts > 0, use CDVR
        if not current_conflict_map:
            break
            
        # Find the position with the highest conflict count
        # If tie, pick the one with the largest value (to help reduce scope) or random
        max_conflict = max(current_conflict_map.values())
        # Get all positions with max conflict
        candidates = [key for key, val in current_conflict_map.items() if val == max_conflict]
        
        # Pick one randomly from the worst
        r_idx, p_idx = random.choice(candidates)
        
        # Generate candidates for this position
        row = rows[r_idx]
        prev_val = row[p_idx - 1] if p_idx > 0 else -1
        next_val = row[p_idx + 1] if p_idx < k else current_scope + 100
        
        # We want to try values that might reduce conflicts.
        # Strategy: Try the smallest few values in the valid range, and also values near the current one.
        
        # Determine search range
        # Lower bound: prev_val + 1
        # Upper bound: min(next_val - 1, current_scope) -- we don't want to increase scope unnecessarily
        # But if we are stuck, we might need to increase scope to resolve conflicts.
        # Let's allow up to current_scope + 10
        
        upper_bound = min(next_val - 1, current_scope + 10)
        
        # Create a list of candidate values
        # 1. The current value (skip)
        # 2. Smallest valid value
        # 3. Values near current
        
        candidate_vals = set()
        # Add smallest
        candidate_vals.add(prev_val + 1)
        # Add current +/- small deltas
        for d in range(-3, 4):
            v = row[p_idx] + d
            if v > prev_val and v < next_val and v <= current_scope + 10:
                candidate_vals.add(v)
        # Add a few random ones in range if range is large
        if upper_bound - prev_val > 10:
            for _ in range(5):
                candidate_vals.add(random.randint(prev_val + 1, upper_bound))
        
        # Sort candidates: Prefer smaller values (to minimize scope)
        # But we must ensure structural validity (strict increase)
        valid_candidates = [v for v in candidate_vals if v > prev_val and v < next_val]
        valid_candidates.sort()
        
        found_improvement = False
        best_candidate = None
        best_candidate_score = float('inf')
        
        for x in valid_candidates:
            if x == row[p_idx]:
                continue
                
            # Apply change
            old_val = row[p_idx]
            row[p_idx] = x
            
            # Evaluate
            new_conflicts, new_scope, new_conflict_map = evaluate_state(rows)
            
            # Score: Primary: conflicts. Secondary: scope.
            # We accept if:
            # 1. Conflicts decreased
            # 2. Conflicts same, but scope decreased
            # 3. Conflicts same, scope same, but "conflict distribution" is better? (Hard to measure)
            
            if new_conflicts < current_conflicts:
                found_improvement = True
                best_candidate = x
                best_candidate_score = new_conflicts * 1000 + new_scope
                # We could break here, but maybe a smaller value with same conflict reduction is better?
                # Since we iterate sorted by value, the first one found is the smallest.
                # But a slightly larger value might resolve more conflicts?
                # Let's just take the first one that reduces conflicts.
                break
            elif new_conflicts == current_conflicts and new_scope < current_scope:
                found_improvement = True
                best_candidate = x
                best_candidate_score = new_conflicts * 1000 + new_scope
                break
            else:
                # Revert
                row[p_idx] = old_val
        
        if found_improvement:
            # Apply the best candidate
            row[p_idx] = best_candidate
            # Re-evaluate to get new state
            current_conflicts, current_scope, current_conflict_map = evaluate_state(rows)
            
            if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                best_conflicts = current_conflicts
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
            stagnation_counter = 0
        else:
            # No improvement found with simple CDVR
            stagnation_counter += 1
            
            if stagnation_counter > 20:
                # Random Kick
                # Pick a random position and change it to a random valid value
                r_idx = random.randint(0, n_rows - 1)
                p_idx = random.randint(1, k)
                row = rows[r_idx]
                prev_val = row[p_idx - 1]
                next_val = row[p_idx + 1] if p_idx < k else current_scope + 50
                
                # Pick a random value in range
                if next_val > prev_val + 1:
                    new_val = random.randint(prev_val + 1, next_val - 1)
                    row[p_idx] = new_val
                    current_conflicts, current_scope, current_conflict_map = evaluate_state(rows)
                    stagnation_counter = 0
                else:
                    # Can't kick this position, try another
                    stagnation_counter -= 1

    return best_rows

