"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    # Greedy baseline
    best_rows = _greedy(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    # Try to improve via backtracking with time limit
    improved = _improve(n_rows, k, best_max, deadline, start_time)
    if improved is not None:
        best_rows = improved
        best_max = max(max(r) for r in best_rows)
    
    return best_rows


def _greedy(n_rows, k):
    """Greedy: smallest next mark with all new differences unused."""
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


def _improve(n_rows, k, current_max, deadline, start_time):
    """
    Try to find a better solution than current_max using randomized restarts
    and backtracking with pruning.
    """
    best_rows = None
    best_max = current_max
    
    rng = random.Random(12345)
    
    # Try to find solutions with max < best_max
    # Use iterative approach: try target_max from some lower bound up to best_max - 1
    # But that's expensive. Instead, do randomized search.
    
    for attempt in range(1000):
        if time.time() > deadline - 3:
            break
        
        # Randomized greedy with perturbation
        rows = _randomized_greedy(n_rows, k, best_max - 1, rng)
        if rows is not None:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = rows
                # Continue trying to go lower
                if best_max <= 112:  # counting bound, can't do better
                    break
    
    return best_rows


def _randomized_greedy(n_rows, k, max_limit, rng):
    """
    Randomized greedy: at each step, pick from a set of candidate next values.
    Prefer smaller values but with some randomness.
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            # Generate candidates: values that don't conflict
            candidates = []
            start = row[-1] + 1
            # Look ahead a bit
            for x in range(start, min(start + 50, max_limit + 1)):
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    candidates.append(x)
                if len(candidates) >= 10:
                    break
            
            if not candidates:
                # Try wider search
                for x in range(start, max_limit + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    if len(candidates) >= 10:
                        break
                
                if not candidates:
                    return None
            
            # Pick a candidate: prefer smaller but with randomness
            # Weight by 1/(x+1) to prefer smaller
            if len(candidates) == 1:
                x = candidates[0]
            else:
                # Random pick from top few smallest
                candidates.sort()
                top_n = min(5, len(candidates))
                x = rng.choice(candidates[:top_n])
            
            new_diffs = {x - a for a in row}
            row.append(x)
            used |= new_diffs
        
        if row[-1] > max_limit:
            return None
        rows.append(row)
    
    return rows


# EVOLVE-BLOCK-END
