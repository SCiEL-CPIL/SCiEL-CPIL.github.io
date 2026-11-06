import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1, 16]
ALL_SETS = [set(c) for c in combinations(range(1, 17), 4)]
N_SETS = len(ALL_SETS)
# Map from set to index for fast lookup? Not strictly needed if we work with indices.

# Precompute conflict lists
# conflict[i][j] is a set of indices k such that {ALL_SETS[i], ALL_SETS[j], ALL_SETS[k]} is a sunflower
# We only need to store this for i < j to save memory, or use a dict.
# Given N=1820, N^2 is ~3.3M. Storing sets of ints might be heavy.
# Instead, we can compute on the fly or use a more compact representation.
# However, the number of conflicts per pair is very small.
# Let's estimate: 
# Case |A n B| = 0: C must be disjoint from A U B (size 8). Remaining 8 elements. C is 4-subset of 8. 70 sets.
# Case |A n B| = 1: C must contain the 1 common, and be disjoint from (A\B) U (B\A) (size 6). Remaining 9 elements (16-1-6=9? No. 16 total. A has 4, B has 4. Union has 7. Common 1. Disjoint part size 16-7=9. C must pick 3 from 9 and include the 1 common. 84 sets.
# Case |A n B| = 2: C must contain 2 common. Disjoint from (A\B) U (B\A) (size 4). Remaining 10 elements. C picks 2 from 10. 45 sets.
# Case |A n B| = 3: C must contain 3 common. Disjoint from (A\B) U (B\A) (size 2). Remaining 11 elements. C picks 1 from 11. 11 sets.
# Average conflicts per pair is roughly (70+84+45+11)/4 ~ 50.
# Total edges in hypergraph ~ 1.6M * 50 = 80M. This is too large to precompute fully in Python lists of sets within 60s if not careful.
# 
# Alternative: On-the-fly check.
# Check if S can be added to Family F:
# For all pairs (A, B) in F, check if S forms a sunflower with A and B.
# Cost: O(|F|^2 * 4) operations.
# If |F| is ~100, 10000 * 4 = 40k ops per check.
# If we try to add 1820 sets, 1820 * 40k = 72M ops. This is feasible in 60s in Python? Maybe tight.
# 
# Let's optimize the sunflower check.
def _is_sunflower(a, b, c):
    ab = a & b
    ac = a & c
    if ab != ac:
        return False
    return ab == (b & c)

def _can_add_fast(s, family_indices, all_sets):
    # family_indices is a list of indices into all_sets
    # s is a set
    # We need to check if there exists i, j in family_indices such that _is_sunflower(all_sets[i], all_sets[j], s)
    # To speed up, we can precompute for each pair in the family the "core" and "symmetric difference" constraints?
    # Or simply iterate.
    
    n_fam = len(family_indices)
    for i in range(n_fam):
        a = all_sets[family_indices[i]]
        ab = a & s
        for j in range(i + 1, n_fam):
            b = all_sets[family_indices[j]]
            if ab == (a & b) and ab == (b & s):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin
    
    best_family = []
    best_size = 0
    
    # We will run multiple iterations of randomized greedy + local search
    # Since the problem is symmetric, the order of selection matters for greedy.
    
    for iteration in range(50): # Limit iterations to ensure we finish
        if time.time() > deadline - 5.0:
            break
            
        # Randomize the order of all sets
        indices = list(range(N_SETS))
        random.shuffle(indices)
        
        family = []
        family_indices = []
        
        # Greedy Construction
        for idx in indices:
            if time.time() > deadline - 10.0:
                break
            s = ALL_SETS[idx]
            if _can_add_fast(s, family_indices, ALL_SETS):
                family.append(s)
                family_indices.append(idx)
        
        # Local Search: Try to improve by swapping
        # A simple improvement: try to add any set that was skipped.
        # If we can't add, try to remove one set and see if we can add two.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try 1-add: Can we add any set not in family?
            # This is O(N * |F|^2). 
            # To make it faster, we can just pick a random few.
            
            # Try 2-swap: Remove one, add two?
            # Or Remove one, Add one (if it allows future additions).
            
            # Let's try a simple random restart within the local search:
            # Pick a random set from family, remove it, then greedily fill with remaining best candidates.
            
            if len(family) > 0:
                # Randomly select a victim
                victim_idx_in_fam = random.randint(0, len(family) - 1)
                victim_set = family[victim_idx_in_fam]
                victim_global_idx = family_indices[victim_idx_in_fam]
                
                # Remove it
                family.pop(victim_idx_in_fam)
                family_indices.pop(victim_idx_in_fam)
                
                # Try to add as many as possible from the remaining candidates
                # We prioritize candidates that were previously skipped or just random
                # To keep it fast, we just run the greedy add again on a shuffled subset of non-family sets
                candidates = [i for i in range(N_SETS) if i not in set(family_indices) and i != victim_global_idx]
                random.shuffle(candidates)
                
                added_count = 0
                for idx in candidates:
                    if time.time() > deadline - 5.0:
                        break
                    s = ALL_SETS[idx]
                    if _can_add_fast(s, family_indices, ALL_SETS):
                        family.append(s)
                        family_indices.append(idx)
                        added_count += 1
                
                # If we added more than 1, we improved (net gain >= 1 if added_count > 1? No, we removed 1. So if added_count > 1, size increases. If added_count == 1, size same but different structure. If added_count == 0, size decreases).
                # We only accept if size increased or (size same and we want to explore).
                # Actually, if added_count >= 1, we might have lost the victim but gained others.
                # If added_count > 1, we definitely gained size.
                # If added_count == 1, size is same. We might accept to escape local optima.
                # If added_count == 0, we lost size. Reject.
                
                if added_count > 1:
                    improved = True
                elif added_count == 1:
                    # Keep it, it's a neutral move, might help later
                    # But to be safe and fast, let's only accept if it leads to a larger family eventually?
                    # For now, accept neutral moves to diversify.
                    improved = True # This allows random walk
                else:
                    # Restore
                    family.append(victim_set)
                    family_indices.append(victim_global_idx)
                    improved = False # No improvement, try another victim? 
                    # To avoid infinite loop on bad moves, we break or limit attempts.
                    # Let's just break the inner loop if we fail a few times?
                    # Simpler: just break the while loop if no improvement in this pass?
                    # The `improved` flag handles it. If we restore, improved stays False (unless we set it True above).
                    # Wait, if I restore, I should set improved to False to stop the while loop?
                    # No, I want to try other victims.
                    # Let's structure: 
                    #   Try victim.
                    #   If success (added > 1), improved = True.
                    #   If neutral (added == 1), improved = True (to allow further moves).
                    #   If failure (added == 0), restore and try next victim?
                    #   The while loop restarts if improved is True.
                    #   So if I restore, I should NOT set improved to True.
                    #   But I need to try the NEXT victim.
                    #   The current structure `while improved` restarts the whole loop.
                    #   This is inefficient.
                    
                    # Let's change structure:
                    #   for each victim in family:
                    #      try swap
                    #      if success: break (restart while)
                    #   if no success in this pass: break while
                    
                    # Since I'm inside a `while improved`, and I just failed one victim,
                    # I should continue to the next victim in the same pass?
                    # But my code structure is:
                    # while improved:
                    #   improved = False
                    #   ... pick one victim ...
                    #   if success: improved = True
                    #   else: pass
                    #   # loop ends, checks improved. If False, exits.
                    #   # This means if the first victim fails, it exits local search.
                    #   # This is bad.
                    
                    # Let's refactor the local search to try multiple victims in one pass.
                    # Actually, the code above is a bit messy. Let's simplify.
                    
                    # I will break out of the inner logic and just rely on the outer `iteration` loop to do multiple restarts.
                    # A single pass of "remove one, refill" is a decent move.
                    # If it fails (added 0), we restore and stop this local search phase.
                    # The next `iteration` will do a full random restart.
                    
                    pass # The restore is already done. improved remains False.

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    return [sorted(s) for s in best_family]
