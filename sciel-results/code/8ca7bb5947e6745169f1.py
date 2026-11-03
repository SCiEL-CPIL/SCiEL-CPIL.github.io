import time
import random
import math
from collections import Counter

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _get_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_init(n_rows, k, rng):
    """
    Generate an initial valid set of rows using a greedy approach.
    We process rows one by one. For each row, we pick the smallest possible
    next integer that doesn't create a difference conflict with any existing
    differences in the global set.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # We want to find k more numbers
        # To keep scope small, we try to fill in small gaps
        # But we must ensure no difference collision
        
        # Heuristic: Try to build the row with small numbers
        # We can use a backtracking search for the first few rows if they are hard,
        # but simple greedy usually works for initialization.
        
        # To improve initialization quality, we can try a few random restarts 
        # for the greedy process itself or just use a standard greedy.
        
        current_max = 0
        for step in range(k):
            # Try values from 1 upwards
            # We can limit the search space to be slightly larger than current_max
            # to avoid infinite loops if greedy gets stuck, but for init we assume it finds one
            found = False
            # We search for the next number. 
            # To encourage small scope, we search from 1.
            # However, we know the row is increasing.
            last_val = row[-1]
            
            # Optimization: We don't need to check 1..last_val because row is increasing.
            # We start from last_val + 1.
            # But we might want to skip some if they cause immediate conflicts.
            
            # To make initialization faster and tighter, we can try a few random 
            # "jumps" if the direct greedy gets stuck, but standard greedy is robust.
            
            # Let's try to find the smallest x > last_val such that 
            # {x - a for a in row} is disjoint from used_diffs AND 
            # {x - a for a in row} has no internal duplicates (which is impossible since row is distinct)
            
            # Actually, we just need to check against used_diffs.
            # Wait, the differences within the NEW row must also be distinct from each other?
            # Yes, "all differences ... inside the rows are distinct over all rows".
            # This implies:
            # 1. Differences in row R1 are distinct from differences in row R2.
            # 2. Differences within row R1 are distinct from each other.
            
            # So when adding x to row, we check:
            # new_diffs = {x - a for a in row}
            # If new_diffs intersects used_diffs, conflict.
            # If new_diffs has duplicates, conflict (impossible if row is distinct).
            # Also, we must ensure that new_diffs doesn't intersect with *future* differences?
            # No, we just add them to used_diffs.
            
            # So the check is:
            # candidate_diffs = {x - a for a in row}
            # if candidate_diffs & used_diffs: continue
            # else: accept x, update used_diffs |= candidate_diffs
            
            # We search for x.
            # To avoid very long searches in initialization, we can set a limit.
            # But for n_rows=7, k=6, the numbers are small (< 120).
            
            search_limit = 500 # Safe upper bound for initialization
            start_x = last_val + 1
            end_x = last_val + search_limit
            
            # To make it slightly randomized to avoid local traps in init:
            # We can shuffle the order of candidates? No, we want small scope.
            # Greedy smallest is best for init.
            
            for x in range(start_x, end_x):
                candidate_diffs = set(x - a for a in row)
                if not (candidate_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(candidate_diffs)
                    found = True
                    break
            
            if not found:
                # Fallback: try larger numbers
                x = last_val + search_limit
                while True:
                    candidate_diffs = set(x - a for a in row)
                    if not (candidate_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(candidate_diffs)
                        break
                    x += 1
        
        rows.append(row)
        
    return rows

def _rebuild_row(row_index, rows, k, rng, max_scope=None):
    """
    Try to rebuild rows[row_index] to have a smaller scope.
    We remove the differences of the current row from the global set,
    then try to construct a new valid row that fits in the available differences.
    We use a backtracking search with pruning.
    """
    n_rows = len(rows)
    
    # 1. Calculate global diffs excluding the target row
    all_diffs = set()
    for i in range(n_rows):
        if i == row_index:
            continue
        all_diffs.update(_get_diffs(rows[i]))
    
    # 2. We want to build a new row.
    # The new row must have differences disjoint from all_diffs.
    # We want to minimize the last element (scope of this row).
    
    # We can use a DFS/Backtracking to find the lexicographically smallest 
    # (or smallest max) valid row.
    
    # To make it fast, we use a greedy-with-backtracking approach.
    # We try to pick the smallest possible next element.
    
    def backtrack(current_row, used_diffs_local, depth):
        if depth == k + 1:
            return current_row[:]
        
        last_val = current_row[-1]
        
        # Determine search range
        # We want to keep the scope small.
        # If max_scope is provided, we don't exceed it.
        # Otherwise, we search up to some limit or current row's scope.
        
        # Heuristic: Start from last_val + 1.
        # We can also try to jump if we are stuck.
        
        # To improve performance, we limit the number of candidates we try per level
        # or use a heuristic ordering.
        
        # Let's try values from last_val + 1 upwards.
        # We stop if we exceed the current best known scope for this row (if we are improving)
        # or a general limit.
        
        current_row_scope = current_row[-1]
        # If we are rebuilding to improve, we likely want scope < current_row_scope
        # But we don't know the target exactly. Let's just find *any* valid row, 
        # then we can sort/compare.
        
        # Actually, finding the *smallest* valid row is ideal.
        
        # Limit search to prevent explosion
        # The max difference we can have is bounded by the global structure.
        # A safe limit for search in this context is maybe 200.
        
        limit = 200
        if max_scope is not None:
            limit = min(limit, max_scope)
        
        for x in range(last_val + 1, limit + 1):
            # Check if x is valid
            # New diffs: {x - a for a in current_row}
            new_diffs = set()
            conflict = False
            for a in current_row:
                d = x - a
                if d in used_diffs_local or d in all_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Check internal consistency of new_diffs (already ensured by set)
            # Recurse
            result = backtrack(current_row + [x], used_diffs_local | new_diffs, depth + 1)
            if result is not None:
                return result
        
        return None

    # Start backtracking
    # We start with [0]
    # used_diffs_local is empty initially
    new_row = backtrack([0], set(), 1)
    
    if new_row is not None:
        return new_row
    else:
        return None

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Leave 10s margin
    rng = random.Random(42)
    
    # 1. Initialization
    # Try a few different initialization strategies or seeds to get a good start
    best_rows = None
    best_scope = float('inf')
    
    # Standard Greedy
    init_rows = _greedy_init(n_rows, k, rng)
    if _valid(init_rows, k):
        best_rows = init_rows
        best_scope = _scope(init_rows)
    
    # Try a randomized greedy init to see if we get a better start
    # (The standard greedy is deterministic given the order, so randomizing the 
    # order of rows or the choice of next number might help)
    
    # For now, stick with one good init.
    
    if best_rows is None:
        # Fallback if init failed (unlikely)
        return [[0, 1, 2, 3, 4, 5, 6]] * n_rows # Invalid, but avoids crash

    # 2. Optimization Loop: Iterated Row Rebuild
    # We repeatedly pick a row (preferably the one with the largest scope)
    # and try to rebuild it to be smaller.
    
    # To avoid getting stuck, we can also perform "perturbations" if no improvement
    # is found after several attempts.
    
    stagnant_count = 0
    max_stagnant = 50 # Try 50 rebuilds before perturbing
    
    while time.time() < deadline:
        # Find the row with the largest scope
        scopes = [_scope([r]) for r in best_rows]
        max_idx = max(range(n_rows), key=lambda i: scopes[i])
        current_max_scope = scopes[max_idx]
        
        # Try to rebuild this row to have scope < current_max_scope
        # We pass max_scope = current_max_scope - 1 to force improvement
        # If that fails, we might try to find *any* valid row (which might be same or larger)
        # but we only accept if it's smaller.
        
        # Actually, _rebuild_row finds the *smallest* valid row (lexicographically/smallest max)
        # given the constraints of the OTHER rows.
        # So we just call it and see what we get.
        
        new_row = _rebuild_row(max_idx, best_rows, k, rng, max_scope=current_max_scope - 1)
        
        if new_row is not None:
            new_scope = new_row[-1]
            if new_scope < current_max_scope:
                # Accept improvement
                best_rows[max_idx] = new_row
                best_scope = _scope(best_rows)
                stagnant_count = 0
            else:
                # No improvement, but maybe a different configuration helps later?
                # For now, we ignore non-improving rebuilds to save time,
                # unless we are stuck.
                stagnant_count += 1
        else:
            # Could not find a row with scope < current_max_scope
            stagnant_count += 1
        
        # If stuck, perturb
        if stagnant_count > max_stagnant:
            # Perturbation strategy:
            # Pick a random row and change it to a random valid configuration 
            # that might be larger, to escape local minima.
            # Or simply swap two rows? Swapping rows doesn't change the set of differences,
            # so it doesn't help with scope.
            
            # Better perturbation: 
            # Pick a random row, and try to rebuild it *without* the constraint of 
            # being smaller than the current scope. Just find *any* valid row.
            # This might lead to a different "shape" that allows other rows to shrink.
            
            # Let's try rebuilding a random row (not necessarily the max scope one)
            # with a higher max_scope limit (e.g., current_max_scope + 10)
            # and accept it even if it's larger, to break symmetry.
            
            # However, this risks increasing the global scope.
            # We should only accept if the GLOBAL scope doesn't increase.
            
            target_idx = rng.randint(0, n_rows - 1)
            # Try to find a valid row for target_idx
            # We use a looser limit
            new_row = _rebuild_row(target_idx, best_rows, k, rng, max_scope=best_scope + 5)
            
            if new_row is not None:
                # Check if replacing this row keeps the global scope valid/small
                # Temporarily replace
                old_row = best_rows[target_idx]
                best_rows[target_idx] = new_row
                
                new_global_scope = _scope(best_rows)
                
                if new_global_scope <= best_scope:
                    # Accept, even if this specific row's scope increased,
                    # as long as global scope didn't. This allows exploration.
                    best_scope = new_global_scope
                    stagnant_count = 0
                else:
                    # Revert
                    best_rows[target_idx] = old_row
                    stagnant_count += 1 # Still stuck
            
            # If we are really stuck, we might need a stronger perturbation
            # like changing the order of rows or randomizing the init and restarting.
            # But given the time limit, simple perturbations are safer.
            
            if stagnant_count > 200:
                # Hard reset: Re-initialize with a new random seed and greedy
                # This is a drastic measure but can escape deep local minima.
                # We keep the best found so far.
                new_init = _greedy_init(n_rows, k, rng)
                if _valid(new_init, k):
                    new_scope = _scope(new_init)
                    if new_scope < best_scope:
                        best_rows = new_init
                        best_scope = new_scope
                    stagnant_count = 0

    return best_rows

