"""Difference triangle set (4,7) solver using ILS with fast delta evaluation and row replacement."""
import time
import random
import sys

def construct(n_rows, k):
    # The task specifies (4,7) but the interface takes n_rows and k.
    # We assume the evaluator passes n_rows=4 and k=7.
    
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    size = k + 1
    # Pre-compute pairs for difference calculation
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    
    # Helper to get differences from a row as a list
    def get_diffs(row):
        return [row[j] - row[i] for i, j in pairs]
    
    # Helper to check validity (used only for final verification or initial check)
    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))
    
    # Helper to get scope
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # Initial construction: Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < size:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # State variables for the search
    # diff_counts[d] = number of times difference d appears in all rows
    # We need to know the max possible difference to size the array.
    # A safe upper bound is 2 * best_scope or just dynamically resize.
    # For (4,7), scope is around 130, so max diff is < 130.
    # Let's start with a reasonable size and resize if needed.
    max_diff_est = best_scope * 2
    diff_counts = [0] * (max_diff_est + 1)
    
    def ensure_size(d):
        nonlocal diff_counts
        if d >= len(diff_counts):
            diff_counts.extend([0] * (d - len(diff_counts) + 1))
    
    # Initialize diff_counts and total_conflicts
    def update_diffs_add(row):
        nonlocal diff_counts
        nonlocal total_conflicts
        for d in get_diffs(row):
            ensure_size(d)
            if diff_counts[d] == 1:
                total_conflicts += 1
            diff_counts[d] += 1
            
    def update_diffs_remove(row):
        nonlocal diff_counts
        nonlocal total_conflicts
        for d in get_diffs(row):
            diff_counts[d] -= 1
            if diff_counts[d] == 1:
                total_conflicts -= 1
                
    total_conflicts = 0
    for r in best_rows:
        update_diffs_add(r)
        
    current_rows = [r[:] for r in best_rows]
    current_scope = get_scope(current_rows)
    
    # Objective function: Minimize Conflicts, then Scope
    # We use a composite score for comparison in SA
    # Score = Conflicts * 10000 + Scope
    def get_score(rows, conflicts):
        return conflicts * 10000 + get_scope(rows)
    
    current_score = get_score(current_rows, total_conflicts)
    best_score = current_score
    best_rows_il = [r[:] for r in current_rows]
    
    # Parameters for ILS/SA
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    kick_strength = 3
    
    # Generate a candidate row given the current set of used differences (from other rows)
    # This function tries to build a row that avoids conflicts with 'used_diffs'
    # and has internal distinct differences.
    def generate_candidate_row(used_diffs_set, max_val_hint, jitter_prob=0.1):
        row = [0]
        x = 0
        internal_diffs = set()
        while len(row) < size:
            x += 1
            if x > max_val_hint + 20:
                # If we go too far, break or return partial? 
                # For validity, we must complete the row.
                # If we can't find a valid one, we might return a row with conflicts.
                # But we want to generate *valid* internal rows.
                pass
            new_diffs = {x - a for a in row}
            # Check internal conflict
            if new_diffs & internal_diffs:
                continue
            # Check conflict with other rows
            if new_diffs & used_diffs_set:
                # If we hit a conflict with other rows, we can still take it if we are "jittering"
                # or if we are desperate. But for now, let's skip.
                # Actually, to allow exploration, we might accept it if we want to fix conflicts later.
                # But a "valid" candidate row should ideally not conflict with others.
                # If we skip, we might loop forever.
                # Let's add a counter.
                if random.random() < jitter_prob:
                    # Accept even if it conflicts with others? No, that increases total_conflicts.
                    # Better to find a non-conflicting one.
                    continue
                # If we have tried many times and failed, we might just accept a conflict to move the search.
                # But for now, let's just keep searching.
                # To avoid infinite loop, we rely on x increasing.
                continue
            
            # If we get here, it's a good candidate
            # Apply jitter: occasionally skip a good candidate to find a better one
            if random.random() < jitter_prob and x < max_val_hint:
                continue
                
            row.append(x)
            internal_diffs |= new_diffs
        return row

    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a row to replace
        # Bias towards rows with larger scope
        # Simple random selection for now, or weighted
        r_idx = random.randint(0, n_rows - 1)
        
        # Calculate used diffs from OTHER rows
        used_diffs_set = set()
        for i in range(n_rows):
            if i != r_idx:
                used_diffs_set.update(get_diffs(current_rows[i]))
        
        # Determine max_val_hint
        # We want to try to keep the scope low.
        # Use the current scope of the row being replaced, or the global scope.
        max_val_hint = max(current_rows[r_idx])
        
        # Generate candidate
        new_row = generate_candidate_row(used_diffs_set, max_val_hint, jitter_prob=0.15)
        
        if len(new_row) < size:
            # Failed to generate a full row? Unlikely with large enough bound.
            continue
            
        # Calculate delta
        # Remove old row
        old_row = current_rows[r_idx]
        old_diffs = get_diffs(old_row)
        old_scope_row = max(old_row)
        
        new_diffs = get_diffs(new_row)
        new_scope_row = max(new_row)
        
        # We need to calculate the change in total_conflicts and total_scope.
        # This is tricky with the global diff_counts array if we don't update it.
        # Let's simulate the change.
        
        # 1. Remove old diffs
        old_conflict_delta = 0
        temp_counts = {} # Use a dict for temporary changes to avoid modifying global state if we reject
        for d in old_diffs:
            temp_counts[d] = temp_counts.get(d, 0) - 1
        for d in new_diffs:
            temp_counts[d] = temp_counts.get(d, 0) + 1
            
        # Calculate new total conflicts based on current global counts + temp changes
        # Current global counts are in diff_counts.
        # We need to check for each d involved in temp_counts if its count crosses 1.
        
        # Let's just re-calculate total_conflicts from scratch? 
        # No, that's O(N). 
        # But N is small (112). 
        # The overhead of Python loops might be higher than just re-calculating.
        # Let's try the O(N) re-calculation for simplicity and correctness first, 
        # then optimize if it's too slow.
        
        # Actually, let's stick to the delta method but carefully.
        # We maintain diff_counts globally.
        # To evaluate a move:
        # 1. Tentatively update diff_counts.
        # 2. Calculate new_score.
        # 3. If rejected, revert diff_counts.
        
        # Let's do that.
        
        # Apply move to global state
        for d in old_diffs:
            if diff_counts[d] == 1:
                total_conflicts -= 1
            diff_counts[d] -= 1
            
        for d in new_diffs:
            ensure_size(d)
            if diff_counts[d] == 1:
                total_conflicts += 1
            diff_counts[d] += 1
            
        current_rows[r_idx] = new_row
        new_scope = get_scope(current_rows)
        new_score = get_score(current_rows, total_conflicts)
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score < 0 or random.random() < pow(2.718281828, -delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows_il = [r[:] for r in current_rows]
        else:
            # Revert
            current_rows[r_idx] = old_row
            for d in old_diffs:
                ensure_size(d)
                if diff_counts[d] == 1:
                    total_conflicts += 1
                diff_counts[d] += 1
            for d in new_diffs:
                if diff_counts[d] == 1:
                    total_conflicts -= 1
                diff_counts[d] -= 1
        
        T *= cooling
        if T < min_T:
            # Restart with kick
            T = 10.0
            # Reset to best
            # First, revert current state to best?
            # We need to rebuild diff_counts for best_rows_il.
            # This is expensive. 
            # Alternative: Keep current state, just increase T and continue?
            # ILS usually kicks the current state.
            
            # Let's kick the current state.
            # Pick 2 random rows and replace them with random valid rows (greedy)
            for _ in range(2):
                r_idx = random.randint(0, n_rows - 1)
                used_diffs_set = set()
                for i in range(n_rows):
                    if i != r_idx:
                        used_diffs_set.update(get_diffs(current_rows[i]))
                        
                # Generate a random-ish row
                # Use greedy but with high jitter
                new_row = generate_candidate_row(used_diffs_set, max(current_rows[r_idx]) + 20, jitter_prob=0.5)
                
                # Update state
                old_row = current_rows[r_idx]
                old_diffs = get_diffs(old_row)
                new_diffs = get_diffs(new_row)
                
                for d in old_diffs:
                    if diff_counts[d] == 1:
                        total_conflicts -= 1
                    diff_counts[d] -= 1
                for d in new_diffs:
                    ensure_size(d)
                    if diff_counts[d] == 1:
                        total_conflicts += 1
                    diff_counts[d] += 1
                    
                current_rows[r_idx] = new_row
                current_score = get_score(current_rows, total_conflicts)
                
            # Continue SA loop with higher T
            
        # Early exit if we hit lower bound
        if best_score == 112:
            break
            
    # Return best valid solution
    # Ensure best_rows_il is valid
    if is_valid(best_rows_il):
        return best_rows_il
    else:
        # Fallback
        return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    for r in rows:
        print(r)
    diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                diffs.append(r[j]-r[i])
    print("Valid:", len(diffs) == len(set(diffs)))
    print("Scope:", max(max(r) for r in rows))
