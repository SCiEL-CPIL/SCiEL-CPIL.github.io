"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction
import random

# Precompute lcm compatibility
MAX_VAL = 256

def _build_compatibility():
    """Build a boolean matrix comp[i][j] = True if lcm(i,j) > 256."""
    n = MAX_VAL - 1  # indices 2..256 -> 0..254
    # We'll use 0-based indexing where index i corresponds to value i+2
    comp = [[False] * n for _ in range(n)]
    for i in range(n):
        a = i + 2
        for j in range(i + 1, n):
            b = j + 2
            # lcm(a,b) = a * b // gcd(a,b)
            # We need lcm > 256, i.e., a*b > 256 * gcd(a,b)
            g = gcd(a, b)
            if a * b > 256 * g:
                comp[i][j] = comp[j][i] = True
    return comp

def _reciprocal_sum(selected_indices):
    """Compute exact reciprocal sum as Fraction."""
    s = Fraction(0, 1)
    for i in selected_indices:
        s += Fraction(1, i + 2)
    return s

def _is_valid(selected_indices, comp):
    """Check if all pairs in selected_indices are compatible."""
    n = len(selected_indices)
    for i in range(n):
        for j in range(i + 1, n):
            if not comp[selected_indices[i]][selected_indices[j]]:
                return False
    return True

def _greedy_by_weight(comp, deadline):
    """Greedy: repeatedly pick the element with highest 1/x that's compatible with all selected."""
    n = MAX_VAL - 1
    # Weights: 1/(i+2), so smaller i has higher weight
    # Process in order of decreasing weight (i.e., increasing i+2, so i=0 first)
    selected = []
    remaining = set(range(n))
    
    while remaining:
        # Find best candidate: highest weight (smallest value) that's compatible
        best = None
        best_val = None
        for i in sorted(remaining, key=lambda x: x):  # x=0 is value 2, highest weight
            if all(comp[i][s] for s in selected):
                best = i
                best_val = i
                break
        
        if best is None:
            break
        
        selected.append(best)
        remaining.remove(best)
        
        if time.time() > deadline:
            break
    
    return selected

def _local_search(selected, comp, deadline):
    """Try to improve by removing one element and adding a better compatible one, or swapping."""
    n = MAX_VAL - 1
    current_score = _reciprocal_sum(selected)
    best_score = current_score
    best_set = list(selected)
    
    selected_set = set(selected)
    
    # Try swaps: remove one, add another
    for i in range(len(selected)):
        for j in range(n):
            if j in selected_set:
                continue
            # Check if j is compatible with all except selected[i]
            new_set = [selected[k] for k in range(len(selected)) if k != i]
            if all(comp[j][s] for s in new_set):
                candidate = new_set + [j]
                score = _reciprocal_sum(candidate)
                if score > best_score:
                    best_score = score
                    best_set = candidate
        if time.time() > deadline:
            break
    
    # Try adding one more element
    for j in range(n):
        if j in selected_set:
            continue
        if all(comp[j][s] for s in best_set):
            candidate = best_set + [j]
            score = _reciprocal_sum(candidate)
            if score > best_score:
                best_score = score
                best_set = candidate
    
    # Try removing one element (if it doesn't help, we keep it, but maybe removing allows adding two?)
    # Actually, removing an element reduces sum, so only do this if it enables adding a higher-weight element
    # This is covered by swaps above
    
    return best_set, best_score

def _anchor_greedy(comp, anchor, deadline):
    """Greedy starting from a fixed anchor element."""
    n = MAX_VAL - 1
    selected = [anchor]
    remaining = set(range(n)) - {anchor}
    
    while remaining:
        best = None
        for i in sorted(remaining, key=lambda x: x):
            if all(comp[i][s] for s in selected):
                best = i
                break
        
        if best is None:
            break
        
        selected.append(best)
        remaining.remove(best)
        
        if time.time() > deadline:
            break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, leaving 10s safety margin
    
    comp = _build_compatibility()
    n = MAX_VAL - 1
    
    # Strategy 1: Standard greedy (highest weight first)
    best_set = _greedy_by_weight(comp, deadline)
    best_score = _reciprocal_sum(best_set)
    
    # Strategy 2: Try different anchors and do greedy from there
    # For each possible anchor, try greedy completion
    for anchor in range(n):
        if time.time() > deadline:
            break
        candidate = _anchor_greedy(comp, anchor, deadline)
        score = _reciprocal_sum(candidate)
        if score > best_score:
            best_score = score
            best_set = candidate
    
    # Strategy 3: Local search on the best set found
    if time.time() < deadline:
        improved_set, improved_score = _local_search(best_set, comp, deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Strategy 4: Multi-start local search
    if time.time() < deadline:
        # Try a few random starts
        random.seed(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            # Random subset of size 2-5
            size = random.randint(2, 5)
            random_set = random.sample(range(n), size)
            # Check validity
            if _is_valid(random_set, comp):
                # Greedy completion
                candidate = _greedy_completion(random_set, comp, deadline)
                score = _reciprocal_sum(candidate)
                if score > best_score:
                    best_score = score
                    best_set = candidate
                # Local search
                if time.time() < deadline:
                    improved_set, improved_score = _local_search(candidate, comp, deadline)
                    if improved_score > best_score:
                        best_score = improved_score
                        best_set = improved_set
    
    # Convert indices back to values
    result = [i + 2 for i in best_set]
    
    # Verify validity
    assert _is_valid(best_set, comp), "Invalid result!"
    assert len(result) > 0, "Empty result!"
    assert all(2 <= x <= 256 for x in result), "Value out of range!"
    
    return result

def _greedy_completion(initial, comp, deadline):
    """Complete a valid initial set using greedy."""
    selected = list(initial)
    selected_set = set(selected)
    n = MAX_VAL - 1
    remaining = set(range(n)) - selected_set
    
    while remaining:
        best = None
        for i in sorted(remaining, key=lambda x: x):
            if all(comp[i][s] for s in selected):
                best = i
                break
        
        if best is None:
            break
        
        selected.append(best)
        remaining.remove(best)
        
        if time.time() > deadline:
            break
    
    return selected
# EVOLVE-BLOCK-END
