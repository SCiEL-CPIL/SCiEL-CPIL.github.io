"""Improved program: multi-strategy search for strictly increasing sequences
in [1,256] with distinct consecutive-block sums."""
import time
import random

def _get_all_block_sums(values):
    """Get all consecutive block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _can_append(values, x, sums):
    """Check if x can be appended to values without creating duplicate block sums.
    Returns the set of new sums if valid, else None."""
    new_sums = set()
    total = x
    new_sums.add(total)
    for a in reversed(values):
        total += a
        new_sums.add(total)
    if new_sums & sums:
        return None
    return new_sums

def _greedy_strategy(values, sums, start_val, max_val, deadline, strategy='smallest'):
    """Greedy with different strategies for selecting next element."""
    while time.time() < deadline:
        added = False
        if strategy == 'smallest':
            candidates = range(start_val, max_val + 1)
        elif strategy == 'largest':
            candidates = range(max_val, start_val - 1, -1)
        elif strategy == 'random':
            candidates = list(range(start_val, max_val + 1))
            random.shuffle(candidates)
        else:
            candidates = range(start_val, max_val + 1)
        
        for x in candidates:
            if x < start_val:
                continue
            new_sums = _can_append(values, x, sums)
            if new_sums is not None:
                values.append(x)
                sums.update(new_sums)
                start_val = x + 1
                added = True
                break
        if not added:
            break
    return values, sums

def _greedy_with_lookahead(values, sums, start_val, max_val, deadline, lookahead=2):
    """Greedy that considers future flexibility when choosing next element."""
    while time.time() < deadline:
        best_x = None
        best_score = -1
        best_new_sums = None
        
        for x in range(start_val, max_val + 1):
            if time.time() > deadline:
                break
            new_sums = _can_append(values, x, sums)
            if new_sums is None:
                continue
            
            # Score: prefer elements that leave more options
            # Count how many elements > x can still be appended after this choice
            score = 0
            for y in range(x + 1, min(x + 1 + lookahead * 10, max_val + 1)):
                if _can_append(values + [x], y, new_sums) is not None:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
                best_new_sums = new_sums
        
        if best_x is None:
            break
        
        values.append(best_x)
        sums.update(best_new_sums)
        start_val = best_x + 1
    
    return values, sums

def _backtrack_extend(values, sums, min_val, max_val, deadline, max_depth=10, branch_factor=3):
    """Limited backtracking search to extend a sequence."""
    best = list(values)
    
    if time.time() > deadline or len(values) >= max_depth:
        return best
    
    # Collect valid candidates, limit to branch_factor
    candidates = []
    for x in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        new_sums = _can_append(values, x, sums)
        if new_sums is not None:
            candidates.append((x, new_sums))
            if len(candidates) >= branch_factor:
                break
    
    for x, new_sums in candidates:
        if time.time() > deadline:
            break
        
        values.append(x)
        sums.update(new_sums)
        
        if len(values) > len(best):
            best = list(values)
        
        result = _backtrack_extend(values, sums, x + 1, max_val, deadline, 
                                   max_depth, branch_factor)
        if len(result) > len(best):
            best = result
        
        values.pop()
        for s in new_sums:
            sums.discard(s)
    
    return best

def _multi_start_search(deadline):
    """Try multiple different starting strategies and return best sequence."""
    best = []
    best_len = 0
    
    strategies = [
        ('smallest', 1),
        ('largest', 1),
        ('random', 1),
        ('smallest', 2),
        ('smallest', 5),
        ('smallest', 10),
    ]
    
    for strat, start in strategies:
        if time.time() > deadline - 2:
            break
        values = []
        sums = set()
        values, sums = _greedy_strategy(values, sums, start, 256, deadline, strat)
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    return best

def _try_replacement(values, sums, deadline):
    """Try replacing elements to find longer sequences."""
    best = list(values)
    best_len = len(best)
    n = len(values)
    
    # Try removing each element (except last few) and re-extending
    for i in range(n - 1, max(0, n - 6), -1):
        if time.time() > deadline - 1:
            break
        
        trial = list(values[:i])
        trial_sums = _get_all_block_sums(trial)
        min_val = trial[-1] + 1 if trial else 1
        
        result = _backtrack_extend(trial, trial_sums, min_val, 256, deadline, 
                                   max_depth=8, branch_factor=2)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Phase 1: Multi-start greedy
    best = _multi_start_search(deadline)
    best_len = len(best)
    
    # Phase 2: Greedy with lookahead
    if time.time() < deadline - 5:
        values = []
        sums = set()
        values, sums = _greedy_with_lookahead(values, sums, 1, 256, deadline, lookahead=2)
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    # Phase 3: Randomized greedy with diverse seeds
    for seed in range(20):
        if time.time() > deadline - 3:
            break
        random.seed(seed * 7919 + 13)
        values = []
        sums = set()
        values, sums = _greedy_strategy(values, sums, 1, 256, deadline, 'random')
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    # Phase 4: Backtracking extension
    if time.time() < deadline - 2:
        best_sums = _get_all_block_sums(best)
        min_val = best[-1] + 1 if best else 1
        result = _backtrack_extend(list(best), set(best_sums), min_val, 256, deadline,
                                   max_depth=8, branch_factor=3)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 5: Replacement search
    if time.time() < deadline - 1:
        best_sums = _get_all_block_sums(best)
        result = _try_replacement(list(best), set(best_sums), deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best
