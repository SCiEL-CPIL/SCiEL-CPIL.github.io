"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
import random

# Precompute lcm compatibility
MAX_VAL = 256

def _build_compatibility():
    """Build a boolean matrix comp[i][j] = True if lcm(i,j) > 256."""
    n = MAX_VAL - 1  # indices 2..256 -> 0..254
    comp = [[False] * n for _ in range(n)]
    for i in range(n):
        a = i + 2
        for j in range(i + 1, n):
            b = j + 2
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

def _greedy_by_weight(comp, deadline, start_set=None):
    """Greedy: repeatedly pick the element with highest 1/x that's compatible with all selected."""
    n = MAX_VAL - 1
    selected = list(start_set) if start_set else []
    selected_set = set(selected)
    remaining = set(range(n)) - selected_set
    
    while remaining:
        best = None
        for i in sorted(remaining, key=lambda x: x):  # x=0 is value 2, highest weight
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

def _local_search_2swap(selected, comp, deadline):
    """Try to improve by 1-swap and 2-swap moves."""
    n = MAX_VAL - 1
    current_score = _reciprocal_sum(selected)
    best_score = current_score
    best_set = list(selected)
    
    selected_set = set(selected)
    
    # 1-swap: remove one, add another
    for i in range(len(selected)):
        if time.time() > deadline:
            return best_set, best_score
        new_set = [selected[k] for k in range(len(selected)) if k != i]
        for j in range(n):
            if j in selected_set:
                continue
            if all(comp[j][s] for s in new_set):
                candidate = new_set + [j]
                score = _reciprocal_sum(candidate)
                if score > best_score:
                    best_score = score
                    best_set = candidate
                    selected_set = set(candidate)
    
    # 2-swap: remove two, add two (only if it improves)
    for i in range(len(selected)):
        if time.time() > deadline:
            return best_set, best_score
        for k in range(i + 1, len(selected)):
            new_set = [selected[m] for m in range(len(selected)) if m != i and m != k]
            # Try adding pairs from remaining
            candidates_to_add = [j for j in range(n) if j not in selected_set]
            for a in range(len(candidates_to_add)):
                if time.time() > deadline:
                    return best_set, best_score
                ca = candidates_to_add[a]
                if not all(comp[ca][s] for s in new_set):
                    continue
                for b in range(a + 1, len(candidates_to_add)):
                    cb = candidates_to_add[b]
                    if not all(comp[cb][s] for s in new_set):
                        continue
                    if not comp[ca][cb]:
                        continue
                    candidate = new_set + [ca, cb]
                    score = _reciprocal_sum(candidate)
                    if score > best_score:
                        best_score = score
                        best_set = candidate
                        selected_set = set(candidate)
    
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
                selected_set = set(candidate)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, leaving 15s safety margin
    
    comp = _build_compatibility()
    n = MAX_VAL - 1
    
    # Strategy 1: Standard greedy (highest weight first)
    best_set = _greedy_by_weight(comp, deadline)
    best_score = _reciprocal_sum(best_set)
    
    # Strategy 2: Try different anchors and do greedy from there
    for anchor in range(n):
        if time.time() > deadline:
            break
        candidate = _greedy_by_weight(comp, deadline, start_set=[anchor])
        score = _reciprocal_sum(candidate)
        if score > best_score:
            best_score = score
            best_set = candidate
    
    # Strategy 3: Local search with 2-swap on the best set found
    if time.time() < deadline:
        improved_set, improved_score = _local_search_2swap(best_set, comp, deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Strategy 4: Multi-start local search
    if time.time() < deadline:
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            # Random subset of size 2-5
            size = random.randint(2, 5)
            random_set = random.sample(range(n), size)
            if _is_valid(random_set, comp):
                candidate = _greedy_by_weight(comp, deadline, start_set=random_set)
                score = _reciprocal_sum(candidate)
                if score > best_score:
                    best_score = score
                    best_set = candidate
                if time.time() < deadline:
                    improved_set, improved_score = _local_search_2swap(candidate, comp, deadline)
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
# EVOLVE-BLOCK-END
