"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 16
    k = 4
    all_sets = [frozenset(c) for c in combinations(range(1, n + 1), k)]
    N = len(all_sets)  # 1820
    
    # Precompute all pairwise intersections as frozensets for speed
    # For each set i, store intersections with all j
    # is_sunflower(a, b, c): a&b == a&c == b&c
    # Precompute intersection matrix: inter[i][j] = set i & set j
    # This is N x N = 1820^2 ~ 3.3M entries, each a frozenset of size 0-4
    # That's a lot of memory but manageable. Let's use a list of dicts or just compute on fly.
    # Actually, let's precompute a bitset representation for faster intersection checks.
    # Each 4-set can be represented as a 16-bit integer.
    
    # Map each frozenset to a bitmask
    set_to_mask = {}
    mask_to_set = {}
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        set_to_mask[s] = mask
        mask_to_set[mask] = s
    
    masks = [set_to_mask[s] for s in all_sets]
    
    def masks_intersect_eq(m1, m2, m3):
        """Check if m1&m2 == m1&m3 == m2&m3"""
        i12 = m1 & m2
        i13 = m1 & m3
        return i12 == i13 and i12 == (m2 & m3)
    
    def can_add_mask(cand_mask, family_masks):
        """Check if adding cand_mask to family_masks creates any sunflower."""
        for i in range(len(family_masks)):
            mi = family_masks[i]
            i_ic = mi & cand_mask
            for j in range(i + 1, len(family_masks)):
                mj = family_masks[j]
                i_ij = mi & mj
                if i_ic == i_ij:
                    # Check if i_ij == mj & cand_mask
                    if i_ij == (mj & cand_mask):
                        return False
        return True
    
    def greedy_build(order, deadline):
        """Build family greedily following the given order."""
        family = []
        family_masks = []
        for idx in order:
            if time.time() > deadline:
                break
            m = masks[idx]
            if can_add_mask(m, family_masks):
                family.append(idx)
                family_masks.append(m)
        return family, family_masks
    
    def local_improve(family, family_masks, deadline, max_iters=100):
        """Try to improve family by swapping out sets for better ones."""
        family = list(family)
        family_masks = list(family_masks)
        n_fam = len(family)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each set and adding a better one
            for i in range(n_fam):
                removed_idx = family[i]
                removed_mask = family_masks[i]
                remaining_masks = family_masks[:i] + family_masks[i+1:]
                # Find a set that can be added to remaining
                for s_idx in range(N):
                    if s_idx == removed_idx:
                        continue
                    if s_idx in family:
                        continue
                    m = masks[s_idx]
                    if can_add_mask(m, remaining_masks):
                        # Accept this swap
                        family[i] = s_idx
                        family_masks[i] = m
                        improved = True
                        break
                if improved:
                    break
        return family, family_masks
    
    # Strategy 1: Greedy with natural order
    natural_order = list(range(N))
    family1, fam_masks1 = greedy_build(natural_order, deadline)
    best_family = family1
    best_family_masks = fam_masks1
    best_size = len(family1)
    
    # Strategy 2: Multiple random restarts with different seeds
    random.seed(42)
    for restart in range(80):
        if time.time() > deadline - 15:
            break
        
        order = natural_order[:]
        random.shuffle(order)
        family, fam_masks = greedy_build(order, deadline)
        
        if len(family) > best_size:
            best_family = family
            best_family_masks = fam_masks
            best_size = len(family)
        
        # Try local improvement on promising families
        if len(family) >= best_size - 2 and time.time() < deadline - 5:
            improved_fam, improved_masks = local_improve(family, fam_masks, deadline - 2, max_iters=30)
            if len(improved_fam) > best_size:
                best_family = improved_fam
                best_family_masks = improved_masks
                best_size = len(improved_fam)
    
    # Strategy 3: Greedy with heuristic ordering - prefer sets that conflict less
    if time.time() < deadline - 20:
        # Compute a score for each set: number of other sets it "conflicts with"
        # Two sets conflict if there exists a third set forming a sunflower
        # Approximate: sets with smaller average intersection with others are "less conflicting"
        # Use a quick heuristic: sort by mask value (arbitrary but different from natural)
        
        # Try ordering by mask value
        order3 = sorted(range(N), key=lambda i: masks[i])
        family3, fam_masks3 = greedy_build(order3, deadline)
        if len(family3) > best_size:
            best_family = family3
            best_family_masks = fam_masks3
            best_size = len(family3)
        
        # Try reverse order
        order4 = list(range(N - 1, -1, -1))
        family4, fam_masks4 = greedy_build(order4, deadline)
        if len(family4) > best_size:
            best_family = family4
            best_family_masks = fam_masks4
            best_size = len(family4)
    
    # Final local improvement on best family
    if time.time() < deadline - 3:
        improved_fam, improved_masks = local_improve(best_family, best_family_masks, deadline - 1, max_iters=200)
        if len(improved_fam) > best_size:
            best_family = improved_fam
            best_family_masks = improved_masks
            best_size = len(improved_fam)
    
    # Convert to sorted lists
    result = [sorted(all_sets[idx]) for idx in best_family]
    return result
# EVOLVE-BLOCK-END
