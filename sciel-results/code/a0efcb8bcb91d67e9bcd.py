import time
import random
from itertools import combinations

# Precompute all 4-subsets as bitmasks
# Elements are 0..15
# Mask for element i is 1 << i

def generate_sets():
    sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        sets.append(mask)
    return sets

ALL_SETS = generate_sets()

def can_add(candidate, family):
    """
    Check if candidate can be added to family without creating a sunflower.
    family is a list of masks.
    """
    # Group existing sets in family by their intersection with candidate
    groups = {}
    for a in family:
        inter = a & candidate
        if inter not in groups:
            groups[inter] = []
        groups[inter].append(a)
    
    # For each group with intersection I, check if any pair A, B in the group
    # has A & B == I
    for I, group in groups.items():
        # If group size is 1, no pair exists
        if len(group) < 2:
            continue
        # Check all pairs in this group
        # Optimization: if len(group) is small, just loop
        # Worst case group size is large, but unlikely for random families
        for i in range(len(group)):
            a = group[i]
            for j in range(i+1, len(group)):
                b = group[j]
                if (a & b) == I:
                    return False
    return True

def construct():
    start_time = time.time()
    max_time = 60.0 # Leave margin for 90s limit
    best_family_size = 0
    best_family = []
    
    # Number of restarts depends on speed
    # Estimate time per restart
    # Let's do at least 1 restart, then as many as fit in time.
    
    n_sets = len(ALL_SETS)
    
    while True:
        current_time = time.time()
        if current_time - start_time > max_time - 5.0: # 5s safety margin
            break
        
        # Shuffle a copy of the sets
        shuffled_sets = ALL_SETS[:]
        random.shuffle(shuffled_sets)
        
        family = []
        
        for candidate in shuffled_sets:
            # Check if we should stop this restart if time is tight?
            # The check is O(|F|^2) worst case, but usually faster.
            # If family is large, checks are slower.
            if can_add(candidate, family):
                family.append(candidate)
        
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
        # If we found a very large family, we might want to stop or continue?
        # Theoretical max is likely not 1820. 
        # Let's just keep searching until time runs out.
        
    # Convert masks back to sorted lists of elements
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result

# To ensure we don't hang if random is slow or something, 
# we rely on the time check inside the loop.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found family of size {len(res)}")
