"""Difference triangle set solver using backtracking with difference-conflict pruning."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Each row has k+1 entries starting at 0, strictly increasing.
    # We need n_rows such rows where all positive differences a_j - a_i (i<j) are globally distinct.
    # Score: minimize the largest entry (scope).

    # Greedy baseline first
    def greedy_solve(n_rows, k):
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

    best_rows = greedy_solve(n_rows, k)
    best_scope = max(max(r) for r in best_rows)

    # Now try to improve via backtracking search with limited scope
    # We'll try scopes from a lower bound up, but time-box it.
    # Lower bound: counting bound. Total distinct differences needed = n_rows * k*(k+1)/2
    # Max differences available with scope M: M*(M+1)/2
    # So M*(M+1)/2 >= n_rows * k*(k+1)/2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1

    # Try to find a solution with scope < best_scope using randomized backtracking
    # For small cases, we can enumerate row configurations.
    
    # Precompute all valid rows (subsets of {0..scope} of size k+1 starting with 0)
    # This is only feasible for small scope and k.
    
    def try_backtrack(n_rows, k, max_scope, deadline, seed=42):
        """Try to find n_rows difference-disjoint rows with max entry <= max_scope."""
        rng = random.Random(seed)
        
        # Generate candidate rows: all (k+1)-subsets of {0..max_scope} containing 0
        # For each, compute the set of differences
        candidates = []
        for combo in combinations(range(1, max_scope + 1), k):
            row = [0] + list(combo)
            diffs = frozenset(row[j] - row[i] for i in range(k + 1) for j in range(i + 1, k + 1))
            candidates.append((row, diffs))
        
        if not candidates:
            return None
        
        # Randomize candidate order
        rng.shuffle(candidates)
        
        # Backtracking search
        used_diffs = set()
        selected_rows = []
        
        def backtrack(idx, depth):
            nonlocal used_diffs, selected_rows
            if time.time() > deadline:
                return False
            if depth == n_rows:
                return True
            
            # Try candidates from idx onward
            for i in range(idx, len(candidates)):
                row, diffs = candidates[i]
                if diffs & used_diffs:
                    continue
                # Pruning: if remaining candidates can't provide enough
                remaining = len(candidates) - i - 1
                if remaining < n_rows - depth - 1:
                    break
                
                used_diffs |= diffs
                selected_rows.append(row)
                
                if backtrack(i + 1, depth + 1):
                    return True
                
                selected_rows.pop()
                used_diffs -= diffs
            
            return False
        
        # To make this more efficient, try different orderings
        for attempt in range(20):
            if time.time() > deadline:
                break
            rng.shuffle(candidates)
            used_diffs = set()
            selected_rows = []
            if backtrack(0, 0):
                return selected_rows
            used_diffs = set()
            selected_rows = []
        
        return None
    
    # Try scopes below current best
    for scope in range(max(lb, best_scope - 30), best_scope):
        if time.time() > deadline:
            break
        result = try_backtrack(n_rows, k, scope, deadline, seed=scope * 137 + 7)
        if result is not None:
            best_rows = result
            best_scope = max(max(r) for r in result)
            break

    # Also try a few more randomized restarts with the current best scope - 1
    if time.time() < deadline - 10:
        for scope in range(best_scope - 1, max(lb, best_scope - 15), -1):
            if time.time() > deadline - 5:
                break
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                result = try_backtrack(n_rows, k, scope, deadline, seed=seed * 997 + 3)
                if result is not None:
                    best_rows = result
                    best_scope = max(max(r) for r in result)
                    break
            if best_scope <= scope:
                break

    return best_rows
