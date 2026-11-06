import time
import random
from itertools import combinations

def _submasks(mask):
    """Yield all submasks of mask."""
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask
    yield 0

def _can_add(family, x):
    """
    Check if adding x to family creates a sunflower triple.
    A sunflower triple A, B, x exists if A ∩ B = A ∩ x = B ∩ x.
    Let K = A ∩ x. Then we need B such that B ∩ x = K and A ∩ B = K.
    We group family members by their intersection with x.
    """
    m = len(family)
    if m < 2:
        return True
    
    # Group family members by K = A & x
    # K is a submask of x.
    groups = {}
    for i in range(m):
        a = family[i]
        k = a & x
        if k not in groups:
            groups[k] = []
        groups[k].append(a)
        
    # For each group K, check if there are two sets A, B in group K 
    # such that A & B == K.
    for k, group in groups.items():
        g_len = len(group)
        if g_len < 2:
            continue
        # Check pairs in group
        for i in range(g_len):
            a = group[i]
            for j in range(i + 1, g_len):
                b = group[j]
                if (a & b) == k:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)
    all_set_list = list(all_sets)
    
    best_family = []
    best_size = 0
    rng = random.Random(42)
    
    # Greedy Initialization
    # Try a few different greedy orderings
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        order = list(range(n_total))
        rng.shuffle(order)
        
        family = []
        for idx in order:
            if time.time() > deadline - 10:
                break
            x = all_set_list[idx]
            if _can_add(family, x):
                family.append(x)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # Local Search on this family
        # We will try to improve by adding/removing
        current_family = family[:]
        current_size = len(current_family)
        
        # Set of indices in all_set_list that are in current_family
        # Using a set of masks for O(1) lookup
        fam_set = set(current_family)
        
        improvement_found = True
        while improvement_found and time.time() < deadline - 5:
            improvement_found = False
            
            # Try to add a random set
            # We need a set of candidates not in fam_set
            # To be efficient, we just pick random indices and check
            # If we pick one already in fam_set, we skip
            # This is O(1) expected time per successful pick if fam is small
            # But fam is ~200/1820, so 10% success rate.
            
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                
                idx = rng.randint(0, n_total - 1)
                x = all_set_list[idx]
                if x in fam_set:
                    continue
                
                if _can_add(current_family, x):
                    current_family.append(x)
                    fam_set.add(x)
                    current_size += 1
                    improvement_found = True
                    if current_size > best_size:
                        best_size = current_size
                        best_family = current_family[:]
                    break
            
            # If no improvement, try to remove a set to unblock
            if not improvement_found:
                if current_size == 0:
                    break
                # Remove a random set
                rem_idx = rng.randint(0, current_size - 1)
                rem_set = current_family[rem_idx]
                current_family.pop(rem_idx)
                fam_set.remove(rem_set)
                current_size -= 1
                
                # Try to add a different set immediately
                # Try a few random candidates
                for _ in range(50):
                    idx = rng.randint(0, n_total - 1)
                    x = all_set_list[idx]
                    if x in fam_set:
                        continue
                    if _can_add(current_family, x):
                        current_family.append(x)
                        fam_set.add(x)
                        current_size += 1
                        improvement_found = True
                        if current_size > best_size:
                            best_size = current_size
                            best_family = current_family[:]
                        break

    # Convert best_family to list of lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
