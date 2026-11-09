"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time

# EVOLVE-BLOCK-START

def _check_valid(elements):
    """Check if all 4096 subset sums are distinct. Returns True if valid."""
    subset_sums = {0}
    for elem in elements:
        new_sums = set()
        for s in subset_sums:
            ns = s + elem
            if ns in subset_sums:
                return False
            new_sums.add(ns)
        subset_sums = subset_sums | new_sums
    return len(subset_sums) == 4096

def _greedy_construct():
    """
    Greedy minimal extension: iteratively pick the smallest positive integer
    not yet used that does not cause any subset-sum collision.
    This directly minimizes the largest element by always taking the globally
    smallest valid next number.
    """
    elements = [1]
    subset_sums = {0, 1}
    
    for _ in range(11):  # need 12 total, already have 1
        candidate = 2  # smallest unused positive integer (1 is taken)
        while True:
            # Check if adding 'candidate' creates any collision
            has_collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    has_collision = True
                    break
            if not has_collision:
                break
            candidate += 1
        
        # Add candidate
        elements.append(candidate)
        # Update subset sums
        new_sums = set()
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums = subset_sums | new_sums
    
    return elements

def _try_improve_greedy(greedy, deadline):
    """
    Try to improve the greedy solution by attempting to reduce individual elements.
    Uses a targeted local search: for each element, try replacing it with a smaller
    value and check if the result is still valid.
    """
    best = greedy[:]
    best_max = max(best)
    
    # Try reducing the largest element first, then work down
    for i in range(len(best) - 1, -1, -1):
        if time.time() > deadline:
            break
        old_val = best[i]
        # Try values from 1 to old_val - 1
        for new_val in range(1, old_val):
            if new_val in best:
                continue
            candidate = best[:]
            candidate[i] = new_val
            candidate.sort()
            if len(set(candidate)) != len(candidate):
                continue
            if _check_valid(candidate):
                cand_max = max(candidate)
                if cand_max < best_max:
                    best = candidate
                    best_max = cand_max
                break  # Found improvement, move to next element
    
    return best

def construct():
    """
    Construct 12 distinct positive integers with all 4096 subset sums distinct,
    minimizing the largest element.
    
    Strategy:
    1. Greedy minimal extension (fast, gives near-optimal solution)
    2. Targeted local search to try to reduce individual elements
    """
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Phase 1: Greedy construction
    best = _greedy_construct()
    best_max = max(best)
    
    # Phase 2: Try to improve via targeted local search
    if time.time() < deadline - 5:
        improved = _try_improve_greedy(best, deadline - 5)
        if max(improved) < best_max:
            best = improved
            best_max = max(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
