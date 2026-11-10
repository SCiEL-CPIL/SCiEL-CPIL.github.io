"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    # Total number of differences needed: n_rows * C(k+1, 2)
    num_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on scope: smallest m such that m*(m+1)/2 >= num_diffs
    # Actually we need differences from 1 to m, so m*(m+1)/2 >= num_diffs
    lb = 0
    while lb * (lb + 1) // 2 < num_diffs:
        lb += 1
    
    # Try scopes from lb upward
    best_result = None
    
    # First try a greedy construction to get a baseline
    greedy_result = _greedy_construct(n_rows, k, deadline)
    if greedy_result is not None:
        best_result = greedy_result
        greedy_scope = max(max(row) for row in greedy_result)
    else:
        greedy_scope = float('inf')
    
    # Try to improve with randomized local search / backtracking
    # Start from max(lb, greedy_scope - some_margin) up to greedy_scope
    search_start = max(lb, greedy_scope - 20) if greedy_scope < float('inf') else lb
    search_end = greedy_scope if greedy_scope < float('inf') else lb + 50
    
    # Try backtracking with randomization for smaller scopes
    for scope in range(search_start, search_end + 1):
        if time.time() > deadline:
            break
        
        # Try a few randomized attempts for this scope
        max_attempts = 50 if scope <= search_start + 5 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            random.seed(attempt * 1000 + scope)
            result = _backtracking_construct(n_rows, k, scope, deadline, attempt)
            if result is not None:
                best_result = result
                return best_result
            if time.time() > deadline:
                break
    
    # If we found something via greedy, return it
    if best_result is not None:
        return best_result
    
    # Fallback: pure greedy without scope constraint
    return _greedy_construct(n_rows, k, deadline)


def _greedy_construct(n_rows, k, deadline):
    """
    Greedy construction: each row takes the smallest next mark whose
    differences to earlier marks in the row are all unused.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows


def _backtracking_construct(n_rows, k, max_scope, deadline, seed=0):
    """
    Backtracking with randomization to construct rows within a given scope.
    Each row must have k+1 entries from {0, 1, ..., max_scope} including 0.
    """
    used = set()
    rows = []
    
    def _build_row(row_idx, remaining, current_row, start_from):
        """Recursively build one row."""
        if time.time() > deadline:
            return False
        
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            return True
        
        # Determine which positions are still needed
        # We need k+1 - len(current_row) more entries
        remaining_needed = k + 1 - len(current_row)
        
        # Available numbers: from start_from to max_scope
        # Try in a randomized order for diversity
        candidates = list(range(start_from, max_scope + 1))
        
        # Heuristic: try smaller values first but with some randomization
        if seed % 2 == 0:
            random.shuffle(candidates)
        else:
            # Sort by some heuristic: prefer values that create fewer conflicts
            def conflict_score(val):
                new_diffs = {val - a for a in current_row}
                return len(new_diffs & used)
            candidates.sort(key=conflict_score)
            # Add some randomness to top candidates
            top_n = min(10, len(candidates))
            random.shuffle(candidates[:top_n])
        
        for val in candidates:
            if len(current_row) > 0 and val <= current_row[-1]:
                continue
            if val > max_scope:
                break
            
            new_diffs = {val - a for a in current_row}
            if new_diffs & used:
                continue
            
            # Check if we have enough room to complete the row
            # We need remaining_needed - 1 more values after val
            # The minimum possible value for the next is val + 1, etc.
            # So we need at least (remaining_needed - 1) values in (val, max_scope]
            if val + remaining_needed - 1 > max_scope:
                continue
            
            current_row.append(val)
            used.update(new_diffs)
            
            if _build_row(row_idx, remaining - 1, current_row, val + 1):
                return True
            
            current_row.pop()
            used -= new_diffs
        
        return False
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        if not _build_row(row_idx, k, [0], 1):
            # Failed to build this row, backtrack
            if rows:
                rows.pop()
                # Remove the differences from the last row
                last = rows[-1] if rows else [0]
                # Actually we need to undo the differences properly
                # This is getting complex; let's just return None
            return None
    
    return rows


def _improved_backtracking(n_rows, k, max_scope, deadline, rng):
    """
    More careful backtracking with proper undo.
    """
    used = set()
    rows = []
    
    def build_row(row_idx):
        """Build one complete row."""
        if time.time() > deadline:
            return False
        
        row = [0]
        
        def backtrack(pos, last_val):
            if time.time() > deadline:
                return False
            
            if len(row) == k + 1:
                rows.append(list(row))
                return True
            
            remaining_needed = k + 1 - len(row)
            
            # Candidate values
            candidates = list(range(last_val + 1, max_scope + 1))
            
            # Prune: need enough room
            if candidates and candidates[0] + remaining_needed - 1 > max_scope:
                return False
            
            # Randomize order
            rng.shuffle(candidates)
            
            for val in candidates:
                new_diffs = {val - a for a in row}
                if new_diffs & used:
                    continue
                
                row.append(val)
                used.update(new_diffs)
                
                if backtrack(pos + 1, val):
                    return True
                
                row.pop()
                used -= new_diffs
            
            return False
        
        return backtrack(1, 0)
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        if not build_row(row_idx):
            # Undo this row
            if rows:
                last_row = rows.pop()
                for i in range(len(last_row)):
                    for j in range(i + 1, len(last_row)):
                        used.discard(last_row[j] - last_row[i])
            return None
    
    return rows


def _greedy_construct(n_rows, k, deadline):
    """
    Greedy construction: each row takes the smallest next mark whose
    differences to earlier marks in the row are all unused.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows


def _backtracking_construct(n_rows, k, max_scope, deadline, seed=0):
    """
    Backtracking with randomization to construct rows within a given scope.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    
    def build_row(row_idx):
        if time.time() > deadline:
            return False
        
        row = [0]
        
        def backtrack(last_val):
            if time.time() > deadline:
                return False
            
            if len(row) == k + 1:
                rows.append(list(row))
                return True
            
            remaining_needed = k + 1 - len(row)
            
            # Check feasibility
            if last_val + remaining_needed > max_scope:
                return False
            
            candidates = list(range(last_val + 1, max_scope + 1))
            
            # Heuristic ordering with randomness
            def score(val):
                new_diffs = {val - a for a in row}
                return len(new_diffs & used)
            
            # Sort by conflict score, then shuffle top portion
            candidates.sort(key=score)
            top_n = min(15, len(candidates))
            if top_n > 1:
                rng.shuffle(candidates[:top_n])
            
            for val in candidates:
                new_diffs = {val - a for a in row}
                if new_diffs & used:
                    continue
                
                row.append(val)
                used.update(new_diffs)
                
                if backtrack(val):
                    return True
                
                row.pop()
                used -= new_diffs
            
            return False
        
        return backtrack(0)
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        if not build_row(row_idx):
            # Undo
            if rows:
                last_row = rows.pop()
                for i in range(len(last_row)):
                    for j in range(i + 1, len(last_row)):
                        used.discard(last_row[j] - last_row[i])
            return None
    
    return rows

# EVOLVE-BLOCK-END
