"""Rewritten program: fast iterative random greedy restarts for dissociated reciprocal sets."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_dissociated_fast(elements):
    """
    Check if all subset sums of 1/x for x in elements are distinct.
    Uses incremental Fraction set construction with early collision detection.
    """
    if not elements:
        return True
    sums = {Fraction(0, 1)}
    for x in elements:
        inv = Fraction(1, x)
        new_sums = {s + inv for s in sums}
        # Collision if any new sum was already present
        if sums & new_sums:
            return False
        sums.update(new_sums)
    return True

def _try_add(elements_set, candidate, all_recips):
    """
    Try to add candidate to elements_set, returning (new_set, success).
    Uses incremental check from the existing subset sums.
    """
    if candidate in elements_set:
        return elements_set, False
    inv = all_recips[candidate]
    # We need to check if adding inv creates a collision.
    # We don't have the full sums set stored, so we rebuild quickly.
    # For efficiency, let's just use _is_dissociated_fast on the new list.
    # But that's O(2^k) which is fine for k up to ~12.
    new_list = list(elements_set | {candidate})
    if _is_dissociated_fast(new_list):
        return elements_set | {candidate}, True
    return elements_set, False

def construct():
    start_time = time.time()
    # Use 85s out of 90s for safety
    deadline = start_time + 85
    
    all_elements = list(range(1, 23))
    all_recips = {x: Fraction(1, x) for x in all_elements}
    
    best_set = set()
    best_size = 0
    
    # Precompute a few known good starting points to seed the search
    # Known dissociated sets from literature or quick computation
    initial_candidates = [
        [1, 2, 3, 5, 7, 11, 13, 17, 19],
        [1, 2, 4, 8, 16],
        [1, 2, 3, 5, 7, 11, 13, 17],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 21],
    ]
    
    # Validate and set best initial
    for cand in initial_candidates:
        if _is_dissociated_fast(cand):
            if len(cand) > best_size:
                best_size = len(cand)
                best_set = set(cand)
    
    if best_size == 0:
        # Fallback to trivial
        best_set = {1}
        best_size = 1
    
    # Random greedy restarts
    # Strategy:
    # 1. Pick a random permutation of [1..22]
    # 2. Greedily add elements if they maintain dissociation
    # 3. Record if size > best_size
    # 4. Also try local search from the greedy result: try swaps to improve
    
    num_restarts = 0
    while time.time() < deadline:
        # Random restart
        perm = all_elements[:]
        random.shuffle(perm)
        
        current_set = set()
        for x in perm:
            if time.time() > deadline:
                break
            # Try to add x
            new_set = current_set | {x}
            if _is_dissociated_fast(list(new_set)):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        num_restarts += 1
        
        # If we found a very large set, we might stop early
        if best_size >= 12:
            # Try a few more to see if we can get 13
            if num_restarts > 50:
                break
        
        # Local search from current_set: try to add more or swap
        # This is optional but can help
        if len(current_set) >= best_size and time.time() < deadline - 5:
            # Try to extend current_set by adding one more element
            improved = False
            candidates = [x for x in all_elements if x not in current_set]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                new_set = current_set | {x}
                if _is_dissociated_fast(list(new_set)):
                    current_set = new_set
                    improved = True
                    break
            
            if improved and len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
            # Try swaps: remove one, add one
            for x in list(current_set):
                if time.time() > deadline - 2:
                    break
                temp_set = current_set - {x}
                candidates = [y for y in all_elements if y not in temp_set]
                random.shuffle(candidates)
                for y in candidates[:10]:  # Limit swap attempts
                    if time.time() > deadline - 2:
                        break
                    new_set = temp_set | {y}
                    if _is_dissociated_fast(list(new_set)) and len(new_set) > len(current_set):
                        current_set = new_set
                        break

    return sorted(best_set)

# EVOLVE-BLOCK-END
