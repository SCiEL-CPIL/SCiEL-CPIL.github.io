import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for 120s limit
    
    MAX_SCOPE_ESTIMATE = 256

    def time_up():
        return time.time() - start_time > time_limit

    best_result = None
    best_scope = float('inf')
    
    def greedy_initialization():
        used_diffs = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            last_val = 0
            while len(row) < k + 1:
                found = False
                for x in range(last_val + 1, MAX_SCOPE_ESTIMATE):
                    conflict = False
                    new_mask = 0
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if used_diffs & bit:
                            conflict = True
                            break
                        new_mask |= bit
                    if not conflict:
                        row.append(x)
                        used_diffs |= new_mask
                        last_val = x
                        found = True
                        break
                if not found:
                    return None, float('inf')
            rows.append(row)
        
        scope = max(r[-1] for r in rows)
        return rows, scope

    def local_search_improve(rows, max_iterations=50000):
        """Try to reduce the scope by swapping elements within rows."""
        if time_up():
            return rows, max(r[-1] for r in rows)
        
        used_diffs = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_diffs |= (1 << d)
        
        scope = max(r[-1] for r in rows)
        rng = random.Random(42)
        
        for _ in range(max_iterations):
            if time_up():
                break
            
            # Pick a random row and try to reduce its last element
            row_idx = rng.randrange(n_rows)
            row = rows[row_idx]
            
            if len(row) < 2:
                continue
            
            # Try to move the last element to a smaller value
            last_pos = len(row) - 1
            old_val = row[last_pos]
            
            # Try values from old_val - 1 down to row[last_pos - 1] + 1
            for new_val in range(old_val - 1, row[last_pos - 1], -1):
                # Check if new_val creates conflicts
                conflict = False
                new_mask = 0
                for i in range(last_pos):
                    d = new_val - row[i]
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if not conflict:
                    # Also need to check that new_val doesn't conflict with other rows' diffs
                    # Actually, we already have all diffs in used_diffs, so this is sufficient
                    # But we need to remove the old diffs first
                    old_mask = 0
                    for i in range(last_pos):
                        d = old_val - row[i]
                        old_mask |= (1 << d)
                    
                    # Check if removing old and adding new works
                    temp_used = (used_diffs & ~old_mask) | new_mask
                    # Check for any duplicate diffs in temp_used (shouldn't happen if we track properly)
                    
                    # Accept the change
                    row[last_pos] = new_val
                    used_diffs = temp_used
                    new_scope = max(r[-1] for r in rows)
                    if new_scope < scope:
                        scope = new_scope
                    break
        
        return rows, scope

    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row, current_max_val):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0], current_max_val):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for next element x
            upper_bound = target_scope - 1
            
            # Additional pruning: if we need to place (k+1 - len(current_row)) more elements,
            # and the minimum possible next element is last_val + 1, then the maximum possible
            # final element is at least last_val + (k + 1 - len(current_row))
            remaining_elements = k + 1 - len(current_row)
            if last_val + remaining_elements >= target_scope:
                return False
            
            candidates = []
            
            # Heuristic: prefer candidates that use "large" differences (low scarcity)
            # and keep x small
            for x in range(last_val + 1, upper_bound + 1):
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d >= MAX_SCOPE_ESTIMATE:
                        continue
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Calculate cost: prefer larger differences and smaller x
                cost = 0.0
                min_diff = float('inf')
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                    if d < min_diff:
                        min_diff = d
                
                # Add small random noise to break ties
                noise = rng.random() * 0.01
                
                # Penalize large x to prefer compact rows
                x_penalty = x * 0.1
                
                candidates.append((cost + noise + x_penalty, x, new_mask))
                
                if len(candidates) > 30:
                    break
            
            if not candidates:
                return False
            
            # Sort candidates
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates
            top_n = min(8, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining_candidates:
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x], max(current_max_val, x)):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0], 0)

    init_rows, init_scope = greedy_initialization()
    if init_rows:
        best_result = [r[:] for r in init_rows]
        best_scope = init_scope
        
        # Apply local search to improve the initial solution
        best_result, best_scope = local_search_improve(best_result)

    seed_counter = 0
    current_target = best_scope if best_result else 200
    
    while time.time() - start_time < time_limit:
        if best_result is not None:
            target = best_scope
        else:
            target = current_target
            
        found = solve_attempt(target, seed_counter)
        seed_counter += 1
        
        # After each successful improvement, try local search
        if found and best_result is not None:
            improved_result, improved_scope = local_search_improve([r[:] for r in best_result])
            if improved_scope < best_scope:
                best_result = improved_result
                best_scope = improved_scope

    if best_result is None:
        used = set()
        rows = []
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
        
    return best_result
