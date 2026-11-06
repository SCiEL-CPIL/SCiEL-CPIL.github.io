"""Extended Sidon set finder with backtracking search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_sums(a):
    """Compute all pairwise sums a+b for a<=b in set a."""
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            sums.add(a[i] + a[j])
    return sums

def _can_add(x, a, sums):
    """Check if x can be added to Sidon set a without creating duplicate sums."""
    n = len(a)
    for i in range(n):
        s = x + a[i]
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _add_element(x, a, sums):
    """Add element x to Sidon set, updating sums."""
    n = len(a)
    for i in range(n):
        sums.add(x + a[i])
    sums.add(2 * x)
    a.append(x)
    a.sort()

def _remove_element(x, a, sums):
    """Remove element x from Sidon set, updating sums."""
    n = len(a)
    idx = a.index(x)
    for i in range(n):
        if i != idx:
            sums.discard(x + a[i])
    sums.discard(2 * x)
    a.pop(idx)
    a.sort()

def _backtrack_search(a, sums, candidates, depth, max_depth, deadline, best):
    """Backtracking search for maximum Sidon set."""
    if time.time() > deadline:
        return
    
    if len(a) > len(best):
        best[:] = a[:]
    
    if depth >= max_depth:
        return
    
    # Try adding each candidate
    for i, x in enumerate(candidates):
        if time.time() > deadline:
            return
        
        if _can_add(x, a, sums):
            # Add x
            _add_element(x, a, sums)
            
            # Recurse with remaining candidates
            remaining = candidates[i+1:]
            _backtrack_search(a, sums, remaining, depth + 1, max_depth, deadline, best)
            
            # Remove x
            _remove_element(x, a, sums)
    
    # Also try not adding any more (already handled by base case)

def _greedy_with_backtrack(a_init, deadline, target_size):
    """Greedy fill followed by backtracking to find larger sets."""
    a = list(a_init)
    sums = _compute_sums(a)
    
    # Greedy fill first
    for x in range(12, 285):
        if time.time() > deadline:
            break
        if _can_add(x, a, sums):
            _add_element(x, a, sums)
    
    best = list(a)
    
    # Now try backtracking from different starting points
    # Strategy: try to improve by removing some elements and adding others
    
    # Generate candidates: all numbers not in current set
    current_set = set(a)
    candidates = [x for x in range(1, 285) if x not in current_set]
    
    # Try different removal strategies
    for remove_count in range(0, min(4, len(a))):
        if time.time() > deadline:
            break
        
        if remove_count == 0:
            # Try to add more from current state
            _backtrack_search(a[:], sums.copy(), candidates, 0, 5, deadline, best)
        else:
            # Try removing 'remove_count' elements and backtracking
            # Focus on removing larger elements that might block better solutions
            indices_to_remove = list(range(len(a)))
            random.shuffle(indices_to_remove)
            
            for combo in combinations(indices_to_remove, remove_count):
                if time.time() > deadline:
                    break
                
                a_test = list(a)
                sums_test = sums.copy()
                
                # Remove elements (from largest index to smallest to maintain order)
                for idx in sorted(combo, reverse=True):
                    x = a_test[idx]
                    _remove_element(x, a_test, sums_test)
                
                # Now backtrack to find better solution
                current_set_test = set(a_test)
                candidates_test = [x for x in range(1, 285) if x not in current_set_test]
                
                best_test = list(a_test)
                _backtrack_search(a_test, sums_test, candidates_test, 0, 8, deadline, best_test)
                
                if len(best_test) > len(best):
                    best = list(best_test)
    
    return best

def _multiple_restart_search(deadline):
    """Multiple restarts with different initial strategies."""
    best = [1, 2, 5, 11]
    
    # Strategy 1: Simple greedy
    a1 = [1, 2, 5, 11]
    sums1 = _compute_sums(a1)
    for x in range(12, 285):
        if time.time() > deadline:
            break
        if _can_add(x, a1, sums1):
            _add_element(x, a1, sums1)
    if len(a1) > len(best):
        best = list(a1)
    
    # Strategy 2: Greedy with backtracking
    a2 = _greedy_with_backtrack([1, 2, 5, 11], deadline, 20)
    if len(a2) > len(best):
        best = list(a2)
    
    # Strategy 3: Randomized restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start with prescribed elements plus random additions
        a3 = [1, 2, 5, 11]
        sums3 = _compute_sums(a3)
        
        # Add some random elements first
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        for x in candidates[:10]:
            if _can_add(x, a3, sums3):
                _add_element(x, a3, sums3)
        
        # Then greedy fill
        for x in range(12, 285):
            if time.time() > deadline:
                break
            if x in a3:
                continue
            if _can_add(x, a3, sums3):
                _add_element(x, a3, sums3)
        
        if len(a3) > len(best):
            best = list(a3)
        
        # Try backtracking from this state
        if len(a3) >= len(best):
            current_set = set(a3)
            candidates_bt = [x for x in range(1, 285) if x not in current_set]
            best_bt = list(a3)
            _backtrack_search(a3, sums3, candidates_bt, 0, 4, deadline, best_bt)
            if len(best_bt) > len(best):
                best = list(best_bt)
    
    return best

def construct():
    """Find maximum Sidon set in [1, 284] containing {1, 2, 5, 11}."""
    start_time = time.time()
    # Use 100s deadline for safety (120s limit, 20s margin)
    deadline = start_time + 100
    
    # First, try a quick greedy to get a baseline
    a_greedy = [1, 2, 5, 11]
    sums_greedy = _compute_sums(a_greedy)
    for x in range(12, 285):
        if _can_add(x, a_greedy, sums_greedy):
            _add_element(x, a_greedy, sums_greedy)
    
    best = list(a_greedy)
    
    # Now do more sophisticated search
    if time.time() < deadline:
        result = _multiple_restart_search(deadline)
        if len(result) > len(best):
            best = result
    
    # Final greedy fill on best to see if we can add more
    sums_best = _compute_sums(best)
    for x in range(1, 285):
        if x in best:
            continue
        if _can_add(x, best, sums_best):
            _add_element(x, best, sums_best)
    
    return best

# EVOLVE-BLOCK-END
