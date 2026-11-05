"""Difference triangle set solver: minimize the largest entry over n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences within rows
    are distinct across all rows."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on scope
    lb = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate initial solutions with greedy
    seed = 0
    while time.time() < deadline - 60:  # Reserve 60s for SA + polish
        rows = _greedy_construct(n_rows, k, seed)
        if rows:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break
        seed += 1
    
    if best_rows is None:
        # Fallback if greedy failed (shouldn't happen for small k)
        return _fallback_construct(n_rows, k)
        
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 10:
        best_rows, best_scope = _simulated_annealing(best_rows, best_scope, deadline, k)
        
    # Phase 3: Polish (Local Search)
    if time.time() < deadline:
        best_rows, best_scope = _polish_max(best_rows, best_scope, deadline, k)
        
    return best_rows

def _scope(rows):
    if not rows:
        return float('inf')
    return max(max(r) for r in rows)

def _all_diffs(rows):
    diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
    return diffs

def _greedy_construct(n_rows, k, seed):
    """Deterministic greedy with randomization based on seed."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        max_x = 5000 # Safety cap
        while len(row) < k + 1 and x < max_x:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # Randomly skip to explore
                if seed > 0 and rng.random() < 0.05:
                    continue
                row.append(x)
                used |= new_diffs
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _fallback_construct(n_rows, k):
    # Simple greedy without randomization
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
                used |= new_diffs
        rows.append(row)
    return rows

def _cost(rows, k):
    """
    Cost function: 
    1. If invalid (duplicate diffs), return large penalty + scope.
    2. Otherwise, return scope.
    """
    scope = _scope(rows)
    all_diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                all_diffs.append(r[j] - r[i])
    
    if len(all_diffs) != len(set(all_diffs)):
        # Penalty for duplicates
        dup_count = len(all_diffs) - len(set(all_diffs))
        return 1000000 + scope + dup_count * 10000
        
    return scope

def _simulated_annealing(rows, best_scope, deadline, k):
    """
    Simulated Annealing to minimize scope.
    Moves:
    1. Change an element in a row (maintain increasing order).
    2. Swap elements between two rows (if valid).
    """
    current_rows = [list(r) for r in rows]
    current_cost = _cost(current_rows, k)
    best_rows = [list(r) for r in rows]
    best_cost = current_cost
    
    temperature = 100.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # Pick a random row and a random index to change
        row_idx = random.randint(0, len(current_rows) - 1)
        col_idx = random.randint(1, k) # Don't change 0
        
        old_val = current_rows[row_idx][col_idx]
        
        # Generate a new value
        # Try to move towards smaller values if possible, or random jump
        if random.random() < 0.7:
            # Local search
            new_val = old_val + random.choice([-2, -1, 1, 2])
        else:
            # Random jump
            new_val = random.randint(1, best_scope + 10)
            
        # Ensure increasing order
        if col_idx > 0 and new_val <= current_rows[row_idx][col_idx - 1]:
            new_val = current_rows[row_idx][col_idx - 1] + 1
        if col_idx < k and new_val >= current_rows[row_idx][col_idx + 1]:
            new_val = current_rows[row_idx][col_idx + 1] - 1
            
        if new_val <= current_rows[row_idx][col_idx - 1] or (col_idx < k and new_val >= current_rows[row_idx][col_idx + 1]):
            continue
            
        current_rows[row_idx][col_idx] = new_val
        new_cost = _cost(current_rows, k)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
                # If we found a valid solution with lower scope, cool down faster
                if best_cost < best_scope:
                    temperature *= 0.95
        else:
            # Revert
            current_rows[row_idx][col_idx] = old_val
            
        temperature *= cooling
        if temperature < 0.1:
            temperature = 10.0 # Reheat
            
    return best_rows, best_cost

def _polish_max(rows, best_scope, deadline, k):
    """
    Try to reduce the maximum element by checking if it can be lowered.
    """
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    while time.time() < deadline:
        # Find the row with the max element
        max_idx = max(range(len(current_rows)), key=lambda i: current_rows[i][-1])
        row = current_rows[max_idx]
        old_last = row[-1]
        
        # Try to reduce the last element
        found = False
        for new_last in range(old_last - 1, row[-2], -1):
            row[-1] = new_last
            if _cost(current_rows, k) == current_scope:
                # Valid and same scope? No, we want to check if it's valid.
                # _cost returns scope if valid. If it returns > 1000000, it's invalid.
                if _cost(current_rows, k) < 1000000:
                    # Check if scope actually reduced
                    new_scope = _scope(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        found = True
                        break
                    else:
                        # Valid but same scope? We want to minimize scope.
                        # If scope is same, we might have freed up space for other rows?
                        # For now, we only accept if scope reduces.
                        pass
            row[-1] = old_last
            
        if not found:
            # Try to rebuild the row with a lower cap
            other_diffs = set()
            for i in range(len(current_rows)):
                if i == max_idx:
                    continue
                r = current_rows[i]
                for a in range(len(r)):
                    for b in range(a + 1, len(r)):
                        other_diffs.add(r[b] - r[a])
                        
            # Greedy rebuild with cap current_scope - 1
            new_row = [0]
            x = 0
            while len(new_row) < k + 1 and x < current_scope:
                x += 1
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & other_diffs):
                    new_row.append(x)
                    other_diffs |= new_diffs # Tentative
            
            if len(new_row) == k + 1:
                # Verify full validity
                current_rows[max_idx] = new_row
                c = _cost(current_rows, k)
                if c < 1000000:
                    new_scope = _scope(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                    else:
                        # Revert if scope didn't improve (or if we just want to minimize max)
                        # Actually, if we rebuilt with a lower cap, scope should be lower or equal.
                        # If equal, we might have improved "tightness" for future moves?
                        # Let's accept if scope is not worse.
                        pass
                else:
                    # Revert
                    current_rows[max_idx] = row
                    
            if not found and _scope(current_rows) >= current_scope:
                # No improvement found in this iteration, break to avoid infinite loop
                # or try a different row?
                # For simplicity, break if no progress in a full pass.
                # We can add a counter.
                break
                
    return current_rows, current_scope
# EVOLVE-BLOCK-END
