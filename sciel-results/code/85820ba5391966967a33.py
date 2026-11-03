"""Rewritten program using randomized greedy set-packing with pre-computed Sidon rows."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences within rows
    are distinct across all rows.
    
    Uses a randomized greedy approach with pre-computed candidate rows (Sidon sets)
    sorted by their maximum entry, with backtracking to escape local dead-ends.
    """
    # Counting bound: for k=6, n_rows=7, the counting bound is 105.
    # We'll search up to a reasonable upper bound. The best known is 112.
    # Let's set max_scope to 115 to be safe, but try to find smaller.
    
    # Pre-compute all valid Sidon rows (sets of k+1 integers starting at 0, strictly increasing)
    # where all positive differences within the row are unique within that row.
    # A Sidon set (B_2 sequence) has the property that all pairwise sums are unique,
    # which is equivalent to all positive differences being unique.
    
    # For efficiency, we generate candidate rows up to a maximum scope.
    # Let's start with a reasonable max_scope and increase if needed.
    
    def generate_sidon_rows(k, max_scope):
        """Generate all Sidon sets of size k+1 starting at 0, with max element <= max_scope."""
        candidates = []
        
        # Use backtracking to generate Sidon sets
        def backtrack(current, used_diffs, start_from):
            if len(current) == k + 1:
                candidates.append(current[:])
                return
            
            # Try each possible next value
            for x in range(start_from, max_scope + 1):
                # Check if x can be added
                new_diffs = set()
                valid = True
                for a in current:
                    d = x - a
                    if d <= 0:
                        valid = False
                        break
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current.append(x)
                backtrack(current, used_diffs | new_diffs, x + 1)
                current.pop()
        
        backtrack([0], set(), 1)
        return candidates
    
    # Generate candidate rows up to scope 115
    # This might be slow for large scopes, so let's be smart about it.
    # For k=6, we need 7 elements. The number of Sidon sets is manageable.
    
    # Let's try to find a solution with scope as small as possible.
    # Start with scope = 105 (counting bound) and increase if no solution found.
    
    best_scope = None
    best_rows = None
    
    # Try scopes from 105 upwards
    for scope in range(105, 120):
        # Generate all Sidon rows with max element <= scope
        candidates = generate_sidon_rows(k, scope)
        
        if not candidates:
            continue
        
        # Sort candidates by their maximum entry (scope)
        candidates.sort(key=lambda row: row[-1])
        
        # Now use greedy set-packing: select rows that don't conflict
        # Two rows conflict if they share any difference value
        
        # We'll use randomized greedy with backtracking
        found = _greedy_set_packing(candidates, n_rows, k)
        
        if found is not None:
            best_scope = scope
            best_rows = found
            break
    
    if best_rows is None:
        # Fallback: use the original greedy algorithm
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
    
    return best_rows


def _greedy_set_packing(candidates, n_rows, k):
    """
    Randomized greedy set-packing with backtracking.
    Selects n_rows non-conflicting candidates from the sorted list.
    Returns the selected rows or None if no solution found.
    """
    # Multiple random restarts to escape local dead-ends
    for attempt in range(50):
        result = _greedy_with_backtracking(candidates, n_rows, k, attempt)
        if result is not None:
            return result
    
    return None


def _greedy_with_backtracking(candidates, n_rows, k, seed_offset):
    """
    Greedy selection with limited backtracking.
    """
    random.seed(42 + seed_offset)
    
    # Shuffle candidates with same scope for random tie-breaking
    # But keep the overall order by scope
    
    # Group candidates by scope
    from collections import defaultdict
    by_scope = defaultdict(list)
    for row in candidates:
        by_scope[row[-1]].append(row)
    
    # Sort scopes
    scopes = sorted(by_scope.keys())
    
    # For each scope group, shuffle
    for s in scopes:
        random.shuffle(by_scope[s])
    
    # Greedy selection
    used_diffs = set()
    selected = []
    
    # We need to select n_rows rows. Try to pick the smallest scope rows first.
    # But if we get stuck, we might need to backtrack.
    
    # Simple greedy: iterate through candidates in order, pick if no conflict
    for scope in scopes:
        for row in by_scope[scope]:
            if len(selected) >= n_rows:
                break
            
            # Check if this row conflicts with already selected rows
            row_diffs = set()
            valid = True
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                selected.append(row)
                used_diffs |= row_diffs
        
        if len(selected) >= n_rows:
            break
    
    if len(selected) == n_rows:
        return selected
    
    return None
# EVOLVE-BLOCK-END
