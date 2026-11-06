import time
import random
from itertools import combinations

DEADLINE = 85.0  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # 1. Generate all 4-element subsets as bitmasks
    # Elements are 0..15
    all_masks = []
    for comb in combinations(range(16), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
        
    num_sets = len(all_masks) # 1820
    
    # Helper to check sunflower condition with bitmasks
    def is_sunflower(A, B, C):
        return (A & B) == (A & C) == (B & C)

    # Helper to check if S can be added to a family represented by a list of masks
    # We maintain a list of pairs (A, B, A&B) to speed up checks?
    # Or just iterate. Given M is small, iterating is fine.
    # But let's try to optimize the check.
    
    def can_add(family, S):
        # family is a list of masks
        # Check if S forms a sunflower with any pair in family
        n = len(family)
        for i in range(n):
            A = family[i]
            # Optimization: if A & S is not a subset of A? No, A&S is always subset.
            # The condition is A&B == A&S == B&S
            # Let I_AS = A & S
            # We need B such that B & S == I_AS and A & B == I_AS
            # This implies B must contain I_AS and be disjoint from A \ I_AS?
            # And B & S == I_AS means B contains no elements from S \ I_AS.
            
            # Standard check:
            for j in range(i + 1, n):
                B = family[j]
                if (A & B) == (A & S) == (B & S):
                    return False
        return True

    def greedy_build(order):
        family = []
        for S in order:
            if can_add(family, S):
                family.append(S)
        return family

    def local_improve(family, deadline):
        """
        Try to improve the family size.
        1. Try to add any set.
        2. Try swaps: remove one, add one or more.
        """
        family = list(family)
        family_set = set(family)
        # Precompute list of all masks not in family for faster candidate selection?
        # Or just iterate all_masks.
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Phase 1: Greedy Add
            # Try to add any set that can be added
            # We iterate through all_masks. This is O(N * M^2).
            # If M is large, this is slow.
            # We can shuffle all_masks to avoid bias.
            candidates = list(all_masks)
            random.shuffle(candidates)
            
            added_any = False
            for S in candidates:
                if S in family_set:
                    continue
                if can_add(family, S):
                    family.append(S)
                    family_set.add(S)
                    improved = True
                    added_any = True
                    # Don't break, try to add more
                    # But if we add one, the state changes, so we should restart the scan?
                    # Or continue. Continuing is fine, but might miss some if order matters?
                    # Actually, if S1 and S2 are both addable, adding S1 might block S2.
                    # So we should be careful.
                    # Let's just do one pass. If we added any, we loop again.
            
            if not added_any:
                # Phase 2: Local Search (Swaps)
                # Try to remove one set and add another (or more)
                # We pick a random set to remove? Or iterate?
                # Iterating over all M sets and all N candidates is O(M * N * M^2) = O(N * M^3).
                # Too slow.
                
                # Strategy:
                # Pick a random index i.
                # Remove family[i].
                # Try to add as many sets as possible from the remaining candidates.
                # If we add > 1 set, we improve size.
                # If we add 1 set, size is same, but composition changes.
                # We only accept if size increases?
                # The goal is to MAXIMIZE size.
                # So we should only accept a move if len(new_family) > len(old_family).
                
                # Let's try a few random removals.
                m = len(family)
                if m == 0:
                    break
                    
                # Try up to 100 random removal attempts
                for _ in range(100):
                    if time.time() > deadline - 1.0:
                        break
                    
                    idx = random.randint(0, m - 1)
                    removed = family[idx]
                    rest = family[:idx] + family[idx+1:]
                    rest_set = set(rest)
                    
                    # Try to add sets to 'rest'
                    # We want to add at least 1 set. If we can add 2, great.
                    # Let's just try to add any valid set to 'rest'.
                    # If we can add 1, the size is m. No improvement in size.
                    # We need to add 2 to get m+1.
                    
                    # So, try to add 2 sets.
                    added = 0
                    new_family = list(rest)
                    new_family_set = set(rest)
                    
                    # Shuffle candidates
                    cands = list(all_masks)
                    random.shuffle(cands)
                    
                    for S in cands:
                        if S in new_family_set:
                            continue
                        if can_add(new_family, S):
                            new_family.append(S)
                            new_family_set.add(S)
                            added += 1
                            if added >= 2:
                                break
                            
                    if added >= 2:
                        family = new_family
                        family_set = new_family_set
                        m = len(family)
                        improved = True
                        break # Restart the while loop
                    
        return family

    best_family = []
    
    # 1. Deterministic Greedy
    family = greedy_build(all_masks)
    if len(family) > len(best_family):
        best_family = list(family)
        
    # 2. Randomized Greedy Restarts
    random.seed(42)
    num_restarts = 200 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        order = list(all_masks)
        random.shuffle(order)
        family = greedy_build(order)
        
        if len(family) > len(best_family):
            best_family = list(family)
            
        # Occasionally do local improvement on the current best
        if i % 10 == 0 and time.time() < deadline - 5.0:
            improved_family = local_improve(best_family, deadline)
            if len(improved_family) > len(best_family):
                best_family = improved_family

    # Final local improvement on the best family found
    if time.time() < deadline - 2.0:
        best_family = local_improve(best_family, deadline)
        
    # Convert back to sorted lists of integers 1..16
    # Bitmask to list:
    # bit i set means element i+1 is present.
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

# EVOLVE-BLOCK-END
