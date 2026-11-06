# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def _validate(selected):
    """Check if all unordered triples (with repetition) have unique sums."""
    if len(selected) < 3:
        return True
    sums = set()
    for a in selected:
        for b in selected:
            for c in selected:
                s = a + b + c
                if s in sums:
                    return False
                sums.add(s)
    return True

def _greedy_build(candidates, existing_sums=None, existing_set=None):
    """
    Greedily add elements from candidates to maintain unique triple sums.
    candidates: list of candidate values to try
    existing_sums: set of already-used triple sums (or None to start fresh)
    existing_set: set of already-selected values (or None)
    Returns: (selected_list, updated_sums_set, updated_selected_set)
    """
    if existing_sums is None:
        existing_sums = set()
    if existing_set is None:
        existing_set = set()
    
    selected = list(existing_set)
    
    for x in candidates:
        if x in existing_set:
            continue
        # Check if adding x creates any duplicate triple sum
        # New triples involving x: (x,x,x), (x,x,y), (x,y,y), (x,y,z) for y,z in selected
        # We need to check all triples that include at least one x against existing sums
        # and against each other
        
        # Efficient check: for each pair (y,z) from selected (with replacement),
        # the new sums are x+y+z. Also triples with multiple x's.
        
        # Triples with exactly one x: x + y + z where y,z in selected (with replacement, y<=z)
        # Triples with exactly two x's: 2*x + y where y in selected
        # Triples with three x's: 3*x
        
        new_sums = set()
        
        # 3*x
        s3x = 3 * x
        if s3x in existing_sums:
            continue
        new_sums.add(s3x)
        
        # 2*x + y for y in selected
        for y in selected:
            s2xy = 2 * x + y
            if s2xy in existing_sums:
                continue
            new_sums.add(s2xy)
        
        # x + y + z for y,z in selected with y <= z
        valid = True
        for i, y in enumerate(selected):
            for z in selected[i:]:
                sxyz = x + y + z
                if sxyz in existing_sums:
                    valid = False
                    break
                new_sums.add(sxyz)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check for duplicates among new_sums themselves
        if len(new_sums) != len(new_sums):  # always true for a set
            continue
        
        # All good, add x
        existing_sums.update(new_sums)
        existing_set.add(x)
        selected.append(x)
    
    return selected, existing_sums, existing_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    best_set = []
    best_size = 0
    
    # Strategy: Try multiple random orderings with greedy construction
    # Also try different starting ranges
    
    rng = random.Random(42)
    
    # Pre-generate a pool of candidate values
    # We'll try different strategies:
    # 1. Random permutation of [1, n]
    # 2. Random subset of larger numbers first
    # 3. Mix of approaches
    
    num_attempts = 50  # We'll adjust based on time
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 5:
            break
        
        # Different strategies per attempt
        if attempt % 3 == 0:
            # Random permutation of full range
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
        elif attempt % 3 == 1:
            # Bias toward larger numbers
            candidates = list(range(1, n + 1))
            # Weight toward larger values
            weighted = []
            for x in candidates:
                weight = x / n
                for _ in range(int(weight * 10) + 1):
                    weighted.append(x)
            rng.shuffle(weighted)
            # Remove duplicates while preserving order
            seen = set()
            candidates = []
            for x in weighted:
                if x not in seen:
                    seen.add(x)
                    candidates.append(x)
        else:
            # Start with some random small numbers, then random larger
            small_pool = list(range(1, 200))
            large_pool = list(range(200, n + 1))
            rng.shuffle(small_pool)
            rng.shuffle(large_pool)
            # Take a few small, then most from large
            num_small = min(5, len(small_pool))
            candidates = small_pool[:num_small] + large_pool
            rng.shuffle(candidates[num_small:])  # Shuffle only the large part
    
        selected, sums_set, sel_set = _greedy_build(candidates)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        # Early termination if we hit the cap
        if best_size >= 80:
            break
    
    # Final validation
    if best_set and not _validate(best_set):
        # Shouldn't happen, but just in case
        # Try to repair by removing conflicting elements
        for x in best_set[:]:
            test = [y for y in best_set if y != x]
            if _validate(test):
                best_set = test
                break
    
    return best_set
# EVOLVE-BLOCK-END
