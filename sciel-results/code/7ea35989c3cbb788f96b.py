"""
Strategy: Multi-restart randomized backtracking with constraint propagation
for Difference Triangle Sets (7,5).

Key improvements over the baseline:
1. Use a tighter, dynamic upper bound for the maximum entry.
2. For each row, use backtracking with forward checking to find a valid row
   that minimizes the maximum entry while avoiding used differences.
3. Multiple random restarts with different orderings of rows and randomized
   tie-breaking to explore different solution spaces.
4. Keep the best solution found across all restarts.
5. Deadline guard to stay within the 150s wall-clock limit.
"""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    best_rows = None
    best_score = float('inf')
    
    # Counting bound: n_rows * k * (k+1) / 2 differences total,
    # each difference d can appear in at most k rows (since in a row with max M,
    # there are at most k values that differ by d from 0... actually not quite).
    # Simpler: the counting bound given is 105 for (7,5).
    # For general n_rows, k: lower bound ~ n_rows * k * (k+1) / 2 / (k+1) = n_rows * k / 2
    # But the actual counting bound is more nuanced. We'll use a dynamic upper bound.
    
    # Upper bound: greedy construction gives some solution
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_score = max(max(row) for row in greedy_rows)
    best_rows = greedy_rows
    best_score = greedy_score
    
    # Try multiple random restarts
    max_restarts = 2000
    for seed in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        
        # Try to find a solution with max entry < best_score
        rows, score = _randomized_search(n_rows, k, best_score, deadline)
        
        if rows is not None and score < best_score:
            best_score = score
            best_rows = rows
            # If we reach the counting bound, stop
            # Counting bound for (7,5) is 105
            if n_rows == 7 and k == 5 and best_score <= 105:
                break
            # General heuristic: if we've improved significantly, keep going
            # but with tighter bound
    
    return best_rows

def _randomized_search(n_rows, k, upper_bound, deadline):
    """
    Try to construct n_rows rows with max entry < upper_bound.
    Returns (rows, max_entry) or (None, inf).
    """
    used_diffs = set()
    rows = []
    
    def backtrack(idx):
        nonlocal used_diffs, rows, best_score
        
        if time.time() > deadline - 2:
            return False
        
        if idx == n_rows:
            return True
        
        # Try to construct row idx with max entry < upper_bound
        # Use randomized backtracking for this row
        row = _find_valid_row(k, used_diffs, upper_bound - 1, deadline)
        
        if row is None:
            return False
        
        # Check all differences are valid
        row_diffs = _get_row_diffs(row)
        if row_diffs & used_diffs:
            return False
        
        used_diffs.update(row_diffs)
        rows.append(row)
        
        if backtrack(idx + 1):
            return True
        
        # Backtrack
        rows.pop()
        used_diffs.difference_update(row_diffs)
        return False
    
    # Try multiple random orderings of rows
    for order_attempt in range(10):
        if time.time() > deadline - 2:
            break
        
        # Reset state
        used_diffs.clear()
        rows.clear()
        
        if backtrack(0):
            max_entry = max(max(row) for row in rows)
            if max_entry < upper_bound:
                return rows, max_entry
    
    return None, float('inf')

def _find_valid_row(k, used_diffs, max_entry, deadline):
    """
    Find a row of k+1 increasing integers starting at 0 with max entry <= max_entry
    such that all pairwise differences are not in used_diffs and are internally distinct.
    Uses randomized backtracking.
    """
    def backtrack(row):
        nonlocal used_diffs, max_entry
        
        if time.time() > deadline - 1:
            return None
        
        if len(row) == k + 1:
            # Verify all differences are valid
            diffs = _get_row_diffs(row)
            if diffs & used_diffs:
                return None
            # Check internal distinctness (should be guaranteed by construction)
            if len(diffs) != len(row) * (len(row) - 1) // 2:
                return None
            return list(row)
        
        # Try to pick the next value
        min_val = row[-1] + 1
        max_val = max_entry
        
        # Generate candidates in randomized order, biased towards smaller values
        candidates = list(range(min_val, max_val + 1))
        
        # Shuffle but with bias towards smaller values
        # Use a weighted shuffle
        weights = [1.0 / (i + 1) for i in range(len(candidates))]
        # This gives higher weight to smaller indices (smaller values)
        
        # Simple approach: shuffle, but try smaller values first with some randomness
        random.shuffle(candidates)
        
        for cand in candidates:
            # Check internal conflicts
            new_diffs = {cand - a for a in row}
            if len(new_diffs) != len(row):
                continue  # Internal conflict
            
            # Check global conflicts
            if new_diffs & used_diffs:
                continue
            
            # Recurse
            result = backtrack(row + [cand])
            if result is not None:
                return result
        
        return None
    
    return backtrack([0])

def _get_row_diffs(row):
    """Get all pairwise differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k):
    """Deterministic greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            # Check internal distinctness
            if len(new_diffs) != len(row):
                continue
            # Check global conflicts
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

def _is_valid(rows, k):
    """Validate that rows form a valid difference triangle set."""
    used = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        if not all(r[i] < r[i + 1] for i in range(len(r) - 1)):
            return False
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    return False
                if d in diffs:
                    return False
                diffs.add(d)
        used |= diffs
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)
