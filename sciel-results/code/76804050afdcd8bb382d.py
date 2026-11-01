import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # We need to fit total_diffs distinct differences into [1, Scope].
    # Counting bound: Scope >= total_diffs.
    # Start with a scope that is likely to yield a valid solution.
    # 2 * total_diffs is a safe bet.
    initial_scope = max(total_diffs * 2, 200)
    
    def generate_random_row(scope):
        # Generate k+1 increasing integers starting at 0, ending <= scope
        if k - 1 > 0:
            if scope - 1 < k - 1:
                return list(range(k + 1))
            points = random.sample(range(1, scope), k - 1)
            points.sort()
            return [0] + points
        else:
            return [0]

    # Initialize all rows
    current_rows = [generate_random_row(initial_scope) for _ in range(n_rows)]
    
    MAX_DIFF = 1000  # Generous upper bound for differences
    diff_counts = [0] * (MAX_DIFF + 1)
    
    def evaluate_state(rows):
        for i in range(MAX_DIFF + 1):
            diff_counts[i] = 0
        
        conflicts = 0
        scope = 0
        
        for r in rows:
            max_val = 0
            for i in range(len(r)):
                if r[i] > max_val:
                    max_val = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= MAX_DIFF:
                        diff_counts[d] += 1
            if max_val > scope:
                scope = max_val
        
        for d in range(1, MAX_DIFF + 1):
            if diff_counts[d] > 1:
                conflicts += diff_counts[d] - 1
                
        return conflicts, scope

    init_conflicts, init_scope = evaluate_state(current_rows)
    
    # Track best valid solution
    best_valid_rows = None
    best_valid_scope = float('inf')
    
    # Track best overall (for fallback, though we want valid)
    best_rows = [r[:] for r in current_rows]
    best_conflicts = init_conflicts
    best_scope = init_scope
    
    if init_conflicts == 0:
        best_valid_rows = [r[:] for r in current_rows]
        best_valid_scope = init_scope

    W = 100000
    
    def get_cost(conflicts, scope):
        return conflicts * W + scope

    current_conflicts = init_conflicts
    current_scope = init_scope
    current_cost = get_cost(current_conflicts, current_scope)
    
    T = 100.0
    alpha = 0.995
    no_improve_count = 0
    
    while time.time() < deadline:
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, k)
        
        row = current_rows[ri]
        old_val = row[ei]
        
        lower = row[ei - 1] + 1
        upper = MAX_DIFF
        if ei < k:
            upper = min(upper, row[ei + 1] - 1)
            
        if lower > upper:
            continue
            
        new_val = random.randint(lower, upper)
        
        if new_val == old_val:
            continue
            
        # Apply move
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d <= MAX_DIFF:
                c = diff_counts[d]
                if c > 1:
                    current_conflicts -= 1
                diff_counts[d] -= 1
        
        row[ei] = new_val
        
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            if d <= MAX_DIFF:
                c = diff_counts[d]
                if c >= 1:
                    current_conflicts += 1
                diff_counts[d] += 1
        
        new_scope = max(max(r) for r in current_rows)
        new_cost = get_cost(current_conflicts, new_scope)
        
        delta_cost = new_cost - current_cost
        
        accepted = False
        if delta_cost < 0:
            accepted = True
        else:
            if T > 0:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    accepted = True
            
        if accepted:
            current_scope = new_scope
            current_cost = new_cost
            no_improve_count = 0
            
            # Update best valid
            if current_conflicts == 0:
                if current_scope < best_valid_scope:
                    best_valid_scope = current_scope
                    best_valid_rows = [r[:] for r in current_rows]
                    
            # Update best overall (not really needed if we find valid, but good for tracking)
            if new_cost < best_conflicts * W + best_scope:
                best_conflicts = current_conflicts
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                
        else:
            # Revert
            for j in range(len(row)):
                if j == ei:
                    continue
                d = abs(row[j] - new_val)
                if d <= MAX_DIFF:
                    c = diff_counts[d]
                    if c >= 2:
                        current_conflicts -= 1
                    diff_counts[d] -= 1
            
            row[ei] = old_val
            
            for j in range(len(row)):
                if j == ei:
                    continue
                d = abs(row[j] - old_val)
                if d <= MAX_DIFF:
                    c = diff_counts[d]
                    if c >= 1:
                        current_conflicts += 1
                    diff_counts[d] += 1
            
            no_improve_count += 1

        T *= alpha
        if T < 0.1:
            T = 0.1
            
        if no_improve_count > 5000:
            T = 100.0
            no_improve_count = 0
            # Kick
            ri_kick = random.randint(0, n_rows - 1)
            ei_kick = random.randint(1, k)
            row_kick = current_rows[ri_kick]
            old_val_kick = row_kick[ei_kick]
            lower_kick = row_kick[ei_kick - 1] + 1
            upper_kick = MAX_DIFF
            if ei_kick < k:
                upper_kick = min(upper_kick, row_kick[ei_kick + 1] - 1)
            
            if lower_kick <= upper_kick:
                new_val_kick = random.randint(lower_kick, upper_kick)
                
                for j in range(len(row_kick)):
                    if j == ei_kick:
                        continue
                    d = abs(row_kick[j] - old_val_kick)
                    if d <= MAX_DIFF:
                        c = diff_counts[d]
                        if c > 1:
                            current_conflicts -= 1
                        diff_counts[d] -= 1
                
                row_kick[ei_kick] = new_val_kick
                
                for j in range(len(row_kick)):
                    if j == ei_kick:
                        continue
                    d = abs(row_kick[j] - new_val_kick)
                    if d <= MAX_DIFF:
                        c = diff_counts[d]
                        if c >= 1:
                            current_conflicts += 1
                        diff_counts[d] += 1
                        
                new_scope_kick = max(max(r) for r in current_rows)
                current_scope = new_scope_kick
                current_cost = get_cost(current_conflicts, current_scope)
                
                if current_conflicts == 0 and current_scope < best_valid_scope:
                    best_valid_scope = current_scope
                    best_valid_rows = [r[:] for r in current_rows]

    if best_valid_rows is not None:
        return best_valid_rows
    else:
        # Fallback: return best overall, hoping it's valid or close
        return best_rows
