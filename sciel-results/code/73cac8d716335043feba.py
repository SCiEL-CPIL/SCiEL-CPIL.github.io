"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)
    
    # Map mask to index
    mask_to_idx = {mask: i for i, mask in enumerate(all_subsets)}
    
    # Precompute subsets of each mask (cores) and tails
    # For each mask, list of (I, T) where I is subset of mask, T = mask \ I
    # We only care about I with size <= 3
    mask_cores = [[] for _ in range(num_subsets)]
    for i, mask in enumerate(all_subsets):
        # Get bits in mask
        bits = [j for j in range(n) if mask & (1 << j)]
        # Generate all subsets of bits of size 0 to 3
        for size in range(4):
            for core_bits in combinations(bits, size):
                I = 0
                for b in core_bits:
                    I |= (1 << b)
                T = mask ^ I  # mask & ~I
                mask_cores[i].append((I, T))
                
    # State
    family = set() # Set of indices
    # collections[I] is a list of tails (masks)
    collections = defaultdict(list)
    
    def can_add(idx_new):
        mask_new = all_subsets[idx_new]
        for I, T in mask_cores[idx_new]:
            # Check if there are 2 disjoint tails in collections[I] that are also disjoint from T
            # And disjoint from each other
            cands = []
            for X in collections[I]:
                if (X & T) == 0:
                    cands.append(X)
            L = len(cands)
            for i in range(L):
                xi = cands[i]
                for j in range(i+1, L):
                    xj = cands[j]
                    if (xi & xj) == 0:
                        return False
        return True

    def add_set(idx):
        family.add(idx)
        mask = all_subsets[idx]
        for I, T in mask_cores[idx]:
            collections[I].append(T)
            
    def remove_set(idx):
        family.remove(idx)
        mask = all_subsets[idx]
        for I, T in mask_cores[idx]:
            # Remove T from collections[I]
            # Since tails might not be unique? 
            # If A1 \ I == A2 \ I, then A1 == A2. So tails are unique per I.
            try:
                collections[I].remove(T)
            except ValueError:
                pass # Should not happen

    # Strategy 1: Deterministic Greedy
    for idx in range(num_subsets):
        if time.time() > deadline:
            break
        if can_add(idx):
            add_set(idx)
            
    best_size = len(family)
    best_family = set(family)

    # Strategy 2: Local Search
    # We will try to improve the size.
    # Moves:
    # 1. Add a random unselected set.
    # 2. Swap: Remove one, add one.
    # 3. Remove 1, Add 2.
    
    # Precompute unselected list for faster random choice?
    # Just use random.randint and check
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 2.0:
        if no_improve_count > max_no_improve:
            break
            
        # Try Add
        moved = False
        for _ in range(10): # Try 10 random adds
            idx_new = random.randint(0, num_subsets - 1)
            if idx_new not in family:
                if can_add(idx_new):
                    add_set(idx_new)
                    best_size = len(family)
                    best_family = set(family)
                    no_improve_count = 0
                    moved = True
                    break
            else:
                continue
        
        if moved:
            continue
            
        # Try Swap
        for _ in range(10):
            idx_rem = random.choice(list(family))
            idx_new = random.randint(0, num_subsets - 1)
            if idx_new in family:
                continue
                
            # Check if we can swap
            # Remove idx_rem, check if idx_new can be added
            remove_set(idx_rem)
            if can_add(idx_new):
                add_set(idx_new)
                # Size unchanged, but structure changed. 
                # We accept it to escape local minima.
                no_improve_count = 0
                moved = True
                break
            else:
                # Rollback
                add_set(idx_rem)
                # Note: add_set(idx_rem) might fail if state is inconsistent? 
                # No, we just removed it, so it's safe to add back.
        
        if moved:
            continue
            
        # Try Remove 1, Add 2
        for _ in range(5):
            idx_rem = random.choice(list(family))
            idx_new1 = random.randint(0, num_subsets - 1)
            idx_new2 = random.randint(0, num_subsets - 1)
            
            if idx_new1 in family or idx_new2 in family or idx_new1 == idx_new2:
                continue
                
            # Simulate
            remove_set(idx_rem)
            
            if can_add(idx_new1):
                add_set(idx_new1)
                if can_add(idx_new2):
                    add_set(idx_new2)
                    # Success: Net +1
                    best_size = len(family)
                    best_family = set(family)
                    no_improve_count = 0
                    moved = True
                    break
                else:
                    # Rollback idx_new1
                    remove_set(idx_new1)
            # Rollback idx_rem
            add_set(idx_rem)
            
        if moved:
            continue
            
        no_improve_count += 1

    # Use best family found
    result = []
    for idx in best_family:
        mask = all_subsets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
