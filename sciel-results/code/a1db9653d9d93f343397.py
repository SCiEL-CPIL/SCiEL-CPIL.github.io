"""Rewritten program: greedy saturation with set-cover prioritization for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256
DEADLINE_FRACTION = 0.85  # use 85% of the 60s budget for safety

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all sums a+b for a<=b distinct)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon property."""
    sel_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would create a collision
        new_sums = set()
        for a in selected:
            s = x + a
            if s in sums or s in new_sums:
                return False  # x can be added without collision, so not maximal
            new_sums.add(s)
        if 2 * x in sums or 2 * x in new_sums:
            return False
    return True

def _greedy_forward(n, seed=0):
    """Standard forward greedy: iterate 1..n, add if no collision."""
    selected = []
    sums = set()
    for x in range(1, n + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_block_priority(n, seed=0):
    """
    Set-cover-inspired greedy: at each step, among all valid candidates,
    pick the one that 'blocks' the most other candidates (i.e., creates the most
    sum collisions with remaining elements). This tends to produce smaller maximal sets.
    """
    rng = random.Random(seed)
    selected = []
    sums = set()
    remaining = set(range(1, n + 1))
    
    while remaining:
        # Find all candidates that can be added without collision
        valid_candidates = []
        for x in remaining:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # Score each candidate by how many other remaining elements it blocks
        best_candidate = None
        best_score = -1
        for x in valid_candidates:
            added_sums = set([x + a for a in selected] + [2 * x])
            # Count how many other remaining elements would be blocked
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y would collide with the new sums
                y_sums = [y + a for a in selected] + [2 * y]
                # More efficient: check if any y-sum is in added_sums or if y-sums collide with existing sums
                # Actually, we need to check if adding x first would make y invalid
                # y is blocked if any of y's sums (with selected + x) collide
                for s in y_sums:
                    if s in added_sums:
                        block_count += 1
                        break
                # Also check if 2*y or y+a collides with x's new sums
                # The above loop covers y+a for a in selected, and 2*y
                # But we also need to check y+x against existing sums
                if (x + y) in sums:
                    block_count += 1
                    break
            
            if block_count > best_score:
                best_score = block_count
                best_candidate = x
        
        if best_candidate is None:
            best_candidate = valid_candidates[0]
        
        # Add the best candidate
        x = best_candidate
        added = [x + a for a in selected] + [2 * x]
        selected.append(x)
        sums.update(added)
        remaining.discard(x)
        # Remove any elements that are now blocked
        new_remaining = set()
        for y in remaining:
            y_added = [y + a for a in selected] + [2 * y]
            if len(set(y_added)) == len(y_added) and not sums.intersection(y_added):
                new_remaining.add(y)
        remaining = new_remaining
    
    return selected

def _greedy_randomized_block(n, seed=0, top_k=5):
    """
    Randomized version: among top-k blocking candidates, pick one randomly.
    Allows exploration of different local optima.
    """
    rng = random.Random(seed)
    selected = []
    sums = set()
    remaining = set(range(1, n + 1))
    
    while remaining:
        valid_candidates = []
        for x in remaining:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # Score and pick from top-k
        scored = []
        for x in valid_candidates:
            added_sums = set([x + a for a in selected] + [2 * x])
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                y_sums = [y + a for a in selected] + [2 * y]
                for s in y_sums:
                    if s in added_sums:
                        block_count += 1
                        break
                if (x + y) in sums:
                    block_count += 1
                    break
            scored.append((block_count, x))
        
        scored.sort(reverse=True)
        # Pick from top-k
        k = min(top_k, len(scored))
        _, best_candidate = scored[rng.randrange(k)]
        
        x = best_candidate
        added = [x + a for a in selected] + [2 * x]
        selected.append(x)
        sums.update(added)
        remaining.discard(x)
        new_remaining = set()
        for y in remaining:
            y_added = [y + a for a in selected] + [2 * y]
            if len(set(y_added)) == len(y_added) and not sums.intersection(y_added):
                new_remaining.add(y)
        remaining = new_remaining
    
    return selected

def _local_search_improve(selected, n, max_iter=1000):
    """
    Try to reduce the size of a maximal Sidon set by swapping elements.
    This is a heuristic local search: try replacing one element with a smaller/different one.
    """
    selected = list(selected)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(len(selected)):
            # Try removing selected[i] and see if the set is still maximal
            # If removing it makes it non-maximal, we can't remove it directly.
            # Instead, try swapping with another element.
            for j in range(n + 1):
                if j in selected:
                    continue
                new_selected = selected[:i] + [j] + selected[i+1:]
                new_selected.sort()
                if _is_valid_sidon(new_selected) and _is_maximal(new_selected, n):
                    if len(new_selected) < len(selected):
                        selected = new_selected
                        improved = True
                        break
            if improved:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_FRACTION * 60.0
    
    best_result = None
    best_size = N + 1
    
    # First, get a baseline from forward greedy
    baseline = _greedy_forward(N)
    if _is_valid_sidon(baseline) and _is_maximal(baseline, N):
        best_result = baseline
        best_size = len(baseline)
    
    # Try block-priority greedy with different seeds
    for seed in range(20):
        if time.time() > deadline - 2.0:
            break
        result = _greedy_block_priority(N, seed=seed)
        if _is_valid_sidon(result) and _is_maximal(result, N):
            if len(result) < best_size:
                best_size = len(result)
                best_result = result
    
    # Try randomized block-priority with different seeds and top-k
    for seed in range(30):
        if time.time() > deadline - 2.0:
            break
        for top_k in [3, 5, 10]:
            result = _greedy_randomized_block(N, seed=seed + top_k * 1000, top_k=top_k)
            if _is_valid_sidon(result) and _is_maximal(result, N):
                if len(result) < best_size:
                    best_size = len(result)
                    best_result = result
    
    # Try local search improvement on the best result
    if best_result is not None:
        improved = _local_search_improve(best_result, N, max_iter=500)
        if _is_valid_sidon(improved) and _is_maximal(improved, N):
            if len(improved) < best_size:
                best_size = len(improved)
                best_result = improved
    
    # Final safety check
    if best_result is None:
        # Fallback to forward greedy
        best_result = _greedy_forward(N)
    
    return best_result

# EVOLVE-BLOCK-END
