import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..m, so m >= num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Iterated local search to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    # 3. Backtracking search for smaller scopes
    while time.time() < deadline - 3.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        found = False
        attempts = 0
        max_attempts = 30
        while not found and attempts < max_attempts and time.time() < deadline - 3.0:
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempts)
            attempts += 1
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Polish
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
        
        if not found:
            # If we can't find anything at this target, try a slightly larger one
            # but only if we're close to the lower bound
            if target_scope <= lower_bound + 5:
                break
            # Otherwise, try a different approach: randomized local search
            if time.time() < deadline - 5.0:
                best_solution, best_scope = _randomized_improve(
                    best_solution, n_rows, k, deadline, best_scope
                )
    
    return best_solution

def greedy_construct(n_rows, k):
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

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with bitset-based difference tracking for speed.
    """
    rng = random.Random(seed_offset * 7919 + 12345)
    
    # Use a bitset for difference tracking (differences up to max_scope)
    # diff_bits[d] = 1 if difference d is used
    diff_bits = bytearray(max_scope + 1)
    rows = []
    
    def build_row(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return list(row_so_far)
        
        # Find valid candidates using bitset for fast checking
        valid_candidates = []
        for x in range(current_mark + 1, max_scope + 1):
            # Check if all new differences are available
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope or diff_bits[d]:
                    conflict = True
                    break
            if not conflict:
                # Also check feasibility: need room for remaining elements
                remaining = k + 1 - pos - 1
                if x + remaining > max_scope:
                    continue
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Heuristic: try smaller values first, with some randomization
        n_smallest = min(8, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra_pool = valid_candidates[n_smallest:]
            extra = rng.sample(extra_pool, min(4, len(extra_pool)))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Record which differences we're using
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
            
            # Mark differences as used
            for d in new_diffs:
                diff_bits[d] = 1
            
            res = build_row(row_idx, pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            # Unmark
            for d in new_diffs:
                diff_bits[d] = 0
        
        return None
    
    for i in range(n_rows):
        row = build_row(i, 1, 0, [0])
        if row is None:
            return None
        rows.append(row)
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Reduce scope by shifting elements down. Uses efficient difference tracking.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    
    # Build difference set
    used_diffs = set()
    for row in best:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            # Process from end to start
            for pos in range(len(row) - 1, 0, -1):
                original = row[pos]
                if original <= row[pos - 1] + 1:
                    continue
                
                # Try shifting down
                for new_val in range(row[pos - 1] + 1, original):
                    # Compute new differences
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    
                    # Old differences
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, len(row)):
                        old_diffs.add(row[j] - original)
                    
                    # Check conflicts
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    
                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _randomized_improve(solution, n_rows, k, deadline, current_scope):
    """
    Randomized local search: try to reduce scope by random perturbations.
    """
    best = [list(row) for row in solution]
    best_scope = current_scope
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        row = best[row_idx]
        pos = rng.randint(1, len(row) - 1)
        
        # Try a random new value
        original = row[pos]
        min_val = row[pos - 1] + 1
        max_val = min(original, best_scope)
        
        if max_val <= min_val:
            continue
        
        new_val = rng.randint(min_val, max_val)
        if new_val == original:
            continue
        
        # Compute differences
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)
        
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(original - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - original)
        
        # Build global used set
        used_diffs = set()
        for r in best:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        if new_diffs & (used_diffs - old_diffs):
            continue
        
        # Apply
        row[pos] = new_val
        new_scope = max(max(r) for r in best)
        
        if new_scope < best_scope:
            best_scope = new_scope
            # Rebuild used_diffs
            used_diffs = set()
            for r in best:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used_diffs.add(r[j] - r[i])
        else:
            # Revert
            row[pos] = original
    
    return best, best_scope
