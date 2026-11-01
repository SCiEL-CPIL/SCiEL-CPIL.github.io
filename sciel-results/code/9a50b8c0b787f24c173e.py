"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START

def _greedy_construct(n_rows, k):
    """Original greedy: each row takes smallest next mark with fresh differences."""
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

def _greedy_construct_ordered(n_rows, k, order):
    """Greedy where rows are built in a specified order (list of row indices)."""
    used, rows = [None] * n_rows, []
    for idx in order:
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        rows[idx] = row
    return rows

def _beam_search_rows(n_rows, k, beam_width, max_scope, deadline):
    """
    Beam search over rows. State: (used_set, list_of_completed_rows, current_row, current_max).
    We build rows one at a time. For each row, we try to extend it greedily but keep
    multiple candidate 'used' sets to allow backtracking at the row level.
    
    Actually, a simpler and more effective beam search:
    - We build rows sequentially.
    - For each row, we generate multiple possible completions (not just the greedy one)
      by exploring a small search tree for that row given the current 'used' set.
    - We keep the top `beam_width` states (by smallest max entry so far) after each row.
    """
    # State: (used, rows_so_far)
    # We'll represent states as (used_set, list_of_rows)
    # To make it efficient, we'll only store the last row's max and the used set.
    
    # Initial state
    states = [(set(), [])]
    
    for row_idx in range(n_rows):
        new_states = []
        for used, rows_so_far in states:
            # Generate candidate rows
            candidates = _generate_row_candidates(used, k, max_scope, deadline)
            for cand_row, new_used, new_max in candidates:
                new_states.append((new_used, rows_so_far + [cand_row], new_max))
        
        if not new_states:
            break
            
        # Sort by new_max (smallest first) and keep top beam_width
        new_states.sort(key=lambda x: x[2])
        states = new_states[:beam_width]
        
        if time.time() > deadline:
            break
    
    if not states:
        return None
        
    # Return the best state
    best_state = states[0]
    return best_state[1]

def _generate_row_candidates(used, k, max_scope, deadline):
    """
    Generate a few candidate rows for a given used set.
    Returns list of (row, new_used, new_max).
    We use a limited DFS to find rows that fit within max_scope.
    """
    candidates = []
    start_time = time.time()
    
    def dfs(row, current_max, used_so_far, depth):
        if time.time() > deadline:
            return
        if len(row) == k + 1:
            candidates.append((row[:], used_so_far, current_max))
            return
        if len(candidates) >= 5:  # Limit candidates per row
            return
            
        # Try next values
        next_val = row[-1] + 1 if row else 1
        # Upper bound: we can't exceed max_scope
        max_next = min(max_scope, current_max + 10)  # Heuristic: don't jump too far
        
        for x in range(next_val, max_next + 1):
            new = {x - a for a in row}
            if not new & used_so_far:
                row.append(x)
                new_used = used_so_far | new
                dfs(row, x, new_used, depth + 1)
                row.pop()
                if len(candidates) >= 5:
                    return
                if time.time() > deadline:
                    return
    
    dfs([], 0, used, 0)
    return candidates

def _local_search_improve(rows, n_rows, k, deadline):
    """
    Try to improve the rows by local search.
    For each row, try to replace it with a smaller one that fits the remaining used set.
    """
    # Compute used set for all rows
    all_used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                all_used.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_rows):
            # Remove row i from used
            row_used = set()
            row = rows[i]
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    row_used.add(row[b] - row[a])
            
            other_used = all_used - row_used
            old_max = row[-1]
            
            # Try to find a smaller row
            new_row, new_max = _find_smaller_row(other_used, k, old_max, deadline)
            if new_row is not None and new_max < old_max:
                # Update
                all_used = all_used - row_used
                new_row_used = set()
                for a in range(len(new_row)):
                    for b in range(a + 1, len(new_row)):
                        new_row_used.add(new_row[b] - new_row[a])
                all_used = all_used | new_row_used
                rows[i] = new_row
                improved = True
    
    return rows

def _find_smaller_row(used, k, max_allowed, deadline):
    """Find a row with max < max_allowed that doesn't conflict with used."""
    # Greedy with a tighter bound
    row, x = [0], 0
    while len(row) < k + 1:
        x += 1
        if x >= max_allowed:
            return None, None
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used = used | new
        if time.time() > deadline:
            return None, None
    return row, x

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Get a baseline greedy solution
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_scope = max(max(row) for row in baseline)
    
    # 2. Try different row orders
    import itertools
    # For small n_rows, try all permutations
    if n_rows <= 6:
        for perm in itertools.permutations(range(n_rows)):
            if time.time() > deadline:
                break
            rows = _greedy_construct_ordered(n_rows, k, list(perm))
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    else:
        # Try a few random permutations
        import random
        for _ in range(20):
            if time.time() > deadline:
                break
            perm = list(range(n_rows))
            random.shuffle(perm)
            rows = _greedy_construct_ordered(n_rows, k, perm)
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # 3. Try beam search with small beam width
    if time.time() < deadline - 30:
        beam_result = _beam_search_rows(n_rows, k, beam_width=3, max_scope=best_scope, deadline=deadline - 10)
        if beam_result is not None:
            scope = max(max(row) for row in beam_result)
            if scope < best_scope:
                best_scope = scope
                best_rows = beam_result
    
    # 4. Local search improvement
    if time.time() < deadline - 5:
        best_rows = _local_search_improve(best_rows, n_rows, k, deadline - 5)
        best_scope = max(max(row) for row in best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
