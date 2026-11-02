"""
Difference triangle set (4,7) solver.

Goal: choose n_rows rows of k+1 increasing integers starting at 0 so that all
positive differences a_j - a_i (i < j) inside all rows are globally distinct.
Minimize the largest entry (the scope).

Strategy:
- Use a randomized greedy construction with backtracking on the last few
  positions to escape local minima.
- Maintain a global set of used differences.
- For each row, build it incrementally. At each step, consider candidate next
  values in increasing order. If a candidate conflicts, skip it.
- To improve beyond simple greedy, use a beam search or limited backtracking
  when stuck.
- Since the known best for (4,7) is 134, and the counting bound is 112, we
  aim to find a solution with scope close to 112-134.

Given the 120s time limit, we use a heuristic search:
1. Try greedy first.
2. If greedy fails or gives a high scope, use randomized restarts with
   randomized candidate ordering (but still preferring smaller values).
3. Use a deadline guard to return the best solution found.
"""

import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within each row must be globally distinct.
    Return the list of rows.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total number of differences needed: n_rows * (k+1 choose 2)
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on scope: need at least total_diffs distinct differences,
    # each at most scope. So scope >= total_diffs is a loose bound.
    # Actually, differences range from 1 to scope, so we need scope >= total_diffs.
    # But also, each row has k+1 elements, so the largest element in a row
    # is at least k (if consecutive). The counting bound mentioned is 112.
    
    best_solution = None
    best_scope = float('inf')
    
    # Try greedy first
    sol = _greedy_construct(n_rows, k)
    if sol is not None:
        scope = max(max(row) for row in sol)
        if scope < best_scope:
            best_scope = scope
            best_solution = sol
    
    if time.time() > deadline:
        return best_solution if best_solution is not None else _fallback(n_rows, k)
    
    # Try randomized restarts
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        sol = _randomized_construct(n_rows, k, random)
        if sol is not None:
            scope = max(max(row) for row in sol)
            if scope < best_scope:
                best_scope = scope
                best_solution = sol
                # If we hit the counting bound, we're done
                if scope <= 112:
                    break
    
    if best_solution is None:
        # Fallback: just return a valid solution even if not optimal
        best_solution = _fallback(n_rows, k)
    
    return best_solution


def _greedy_construct(n_rows, k):
    """Simple greedy: each row takes smallest possible next value."""
    used = set()
    rows = []
    for _ in range(n_rows):
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


def _randomized_construct(n_rows, k, rng):
    """
    Randomized greedy with occasional backtracking.
    For each row, build incrementally. At each step, consider a random
    subset of candidate values, preferring smaller ones.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        current_max = 0
        success = False
        
        # Try multiple times to build this row
        for attempt in range(20):
            row = [0]
            current_max = 0
            valid = True
            
            for pos in range(1, k + 1):
                # Generate candidate values: prefer small, but add randomness
                # Try values from current_max+1 upward, but skip some randomly
                candidates = []
                x = current_max + 1
                while len(candidates) < 30:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append((x, new_diffs))
                    x += 1
                    if x > 200:  # Safety bound
                        break
                
                if not candidates:
                    valid = False
                    break
                
                # Choose a candidate: prefer smaller, but with some randomness
                # Use a weighted selection: lower x gets higher weight
                if rng.random() < 0.7:
                    # Pick from the first few candidates
                    idx = min(len(candidates) - 1, rng.randint(0, 2))
                    chosen_x, chosen_diffs = candidates[idx]
                else:
                    # Pick a random candidate
                    idx = rng.randint(0, len(candidates) - 1)
                    chosen_x, chosen_diffs = candidates[idx]
                
                row.append(chosen_x)
                current_max = chosen_x
                used |= chosen_diffs
            
            if valid:
                success = True
                break
        
        if not success:
            return None
        
        rows.append(row)
    
    return rows


def _fallback(n_rows, k):
    """Fallback: just use greedy, which always works."""
    used = set()
    rows = []
    for _ in range(n_rows):
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
