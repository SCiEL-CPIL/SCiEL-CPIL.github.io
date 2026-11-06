<<<<<<< SEARCH
import random
import time
from itertools import combinations

# Precompute all 4-subsets of 16
N = 16
K = 4
ALL_COMBOS = list(combinations(range(1, N + 1), K))
ALL_SETS = [frozenset(c) for c in ALL_COMBOS]

def can_add(new_set, family):
    # Check if new_set forms a sunflower with any pair in family
    # Sunflower condition: A ∩ B == B ∩ C == A ∩ C
    # Here C = new_set.
    # So we need to check if there exist A, B in family such that:
    # A ∩ B == A ∩ new_set == B ∩ new_set
    
    # Optimization: 
    # Let S = new_set.
    # If A ∩ S == B ∩ S == A ∩ B.
    # Note that A ∩ S is a subset of S.
    # We can group A by (A ∩ S)? No, A ∩ B depends on B.
    
    # Simple O(M^2) check
    # M is at most ~250. M^2 is 62500.
    # 1820 candidates * 62500 = 113M ops. Too slow in Python if done naively in a loop without break?
    # But we break on first failure.
    # Most sets will fail early.
    
    for i in range(len(family)):
        a = family[i]
        inter_as = a & new_set
        for j in range(i + 1, len(family)):
            b = family[j]
            inter_bs = b & new_set
            if inter_as != inter_bs:
                continue
            if a & b == inter_as:
                return False
    return True

def greedy_randomized(seed=0):
    random.seed(seed)
    indices = list(range(len(ALL_SETS)))
    random.shuffle(indices)
    
    family = []
    for idx in indices:
        s = ALL_SETS[idx]
        if can_add(s, family):
            family.append(s)
    return family

def construct():
    start_time = time.time()
    timeout = 45.0 # Safety margin
    
    best_family = []
    
    # Initial greedy
    # Maybe start with a deterministic greedy to get a baseline quickly
    # The seed program does this.
    # Let's just run multiple random restarts.
    
    # We can also try to "complete" a partial family
    # For now, full greedy from scratch is simple.
    
    # Number of restarts
    # Let's estimate time for one run.
    # If it takes 0.5s, we can do 90 runs.
    
    for i in range(100):
        if time.time() - start_time > timeout:
            break
        fam = greedy_randomized(seed=i)
        if len(fam) > len(best_family):
            best_family = fam
            
    # Local search on best_family
    # Try to swap out a set for a better one?
    # Or just try to add more?
    # Since greedy fills up to a maximal set, we can't add more.
    # We can try to remove one and add two?
    # This is complex.
    
    # Let's convert frozensets to sorted lists for output
    return [sorted(s) for s in best_family]
=======
import random
import time
from itertools import combinations

N = 16
K = 4
TOTAL_SETS = len(list(combinations(range(N), K)))  # 1820

# Precompute all 4-subsets as bitmasks (16 bits, bit i represents element i+1)
ALL_MASKS = []
for combo in combinations(range(N), K):
    mask = 0
    for x in combo:
        mask |= (1 << x)
    ALL_MASKS.append(mask)

ALL_SETS = [frozenset(c) for c in combinations(range(1, N + 1), K)]
MASK_TO_SET = {mask: frozenset(c) for mask, c in zip(ALL_MASKS, combinations(range(1, N + 1), K))}

def can_add_mask(new_mask, family_masks):
    """Check if new_mask can be added without forming a sunflower with any pair in family_masks."""
    m = len(family_masks)
    for i in range(m):
        a = family_masks[i]
        inter_an = a & new_mask
        for j in range(i + 1, m):
            b = family_masks[j]
            inter_bn = b & new_mask
            if inter_an != inter_bn:
                continue
            if (a & b) == inter_an:
                return False
    return True

def greedy_min_degree(seed=0, deadline=None):
    """Greedy algorithm that picks the set with minimum conflicts (degree) at each step."""
    random.seed(seed)
    
    # Track which sets are still available
    available = set(range(TOTAL_SETS))
    family = []
    family_masks = []
    
    while available and (deadline is None or time.time() < deadline):
        best_idx = None
        best_conflicts = float('inf')
        
        # Sample a subset of available sets for efficiency
        if len(available) > 100:
            sample = random.sample(list(available), 100)
        else:
            sample = list(available)
        
        for idx in sample:
            mask = ALL_MASKS[idx]
            # Count how many pairs in family would form a sunflower with this set
            conflicts = 0
            m = len(family_masks)
            for i in range(m):
                a = family_masks[i]
                inter_an = a & mask
                for j in range(i + 1, m):
                    b = family_masks[j]
                    inter_bn = b & mask
                    if inter_an == inter_bn and (a & b) == inter_an:
                        conflicts += 1
                        if conflicts >= best_conflicts:
                            break
                if conflicts >= best_conflicts:
                    break
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_idx = idx
                if conflicts == 0:
                    break
        
        if best_idx is None:
            break
        
        # Verify we can actually add this set
        if can_add_mask(ALL_MASKS[best_idx], family_masks):
            family.append(ALL_MASKS[best_idx])
            family_masks.append(ALL_MASKS[best_idx])
            available.discard(best_idx)
        else:
            available.discard(best_idx)
    
    return family

def local_search(family, seed=0, deadline=None):
    """Try to improve family by swapping out one set for one that allows adding more."""
    random.seed(seed)
    family = list(family)
    best_size = len(family)
    
    for iteration in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try removing a random set and trying to add two new ones
        if len(family) < 2:
            break
        
        idx_to_remove = random.randint(0, len(family) - 1)
        removed_mask = family.pop(idx_to_remove)
        
        # Try to add up to 2 new sets
        added_count = 0
        candidates = random.sample(range(TOTAL_SETS), min(200, TOTAL_SETS))
        for cidx in candidates:
            if added_count >= 2:
                break
            cmask = ALL_MASKS[cidx]
            if cmask == removed_mask:
                continue
            if can_add_mask(cmask, family):
                family.append(cmask)
                added_count += 1
        
        if len(family) > best_size:
            best_size = len(family)
        
        # Restore if we didn't improve
        if len(family) < best_size:
            family.insert(idx_to_remove, removed_mask)
    
    return family

def construct():
    start_time = time.time()
    timeout = 80.0  # Safety margin before 90s limit
    
    best_family = []
    best_size = 0
    
    # Run multiple greedy min-degree restarts
    for seed in range(200):
        if time.time() - start_time > timeout:
            break
        
        deadline = start_time + timeout
        fam = greedy_min_degree(seed=seed, deadline=deadline)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam
            # Try local search on the best family found so far
            if time.time() - start_time < timeout - 5:
                improved = local_search(best_family, seed=seed + 1000, deadline=deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_family = improved
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(N):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
>>>>>>> REPLACE
