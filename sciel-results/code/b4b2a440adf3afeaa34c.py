"""Rewritten program using QUBO formulation with simulated annealing."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Construct 40 distinct subsets of vertices 0..11 maximizing comparable pairs.
    Uses simulated annealing on the Boolean lattice B_12 (4096 possible subsets).
    """
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    n = 12
    m = 40
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: comp[i] is a bitset (as Python int) where bit j is set
    # if mask i is a strict subset of mask j
    # mask i is subset of mask j iff (i & j) == i
    # strict subset: (i & j) == i and i != j
    
    # For efficiency, we'll use numpy arrays
    masks = np.arange(num_masks, dtype=np.int64)
    
    # Build comparability matrix as a list of bitsets
    # comp[i] will be a Python int with bit j set if i is strict subset of j
    comp = [0] * num_masks
    for i in range(num_masks):
        # All j such that (i & j) == i and i != j
        # j must be a superset of i
        # j = i | k where k is any subset of ~i (within n bits)
        # But we need j != i, so k != 0
        # The supersets of i are i | (k & ~i) for k in 0..2^n-1
        # Actually, the supersets of i within n bits are: i | s where s is any subset of (full_mask ^ i)
        full_mask = (1 << n) - 1
        available = full_mask ^ i
        # Iterate over all submasks of available
        s = available
        bits = 0
        while s:
            j = i | s
            if j != i:  # strict
                bits |= (1 << j)
            s = (s - 1) & available
        # Also j = i itself is not strict, so skip
        comp[i] = bits
    
    # For delta computation: if we add mask i to the family, the change in comparable pairs
    # is the number of existing family members j such that i is strict subset of j OR j is strict subset of i
    # If we remove mask i, the change is negative of that.
    
    # We'll represent the family as a boolean array of length num_masks
    # And maintain a count of how many comparable pairs we have
    
    # Initial solution: greedy - start with empty, add the mask that gives most new comparable pairs
    # Actually, let's start with a reasonable heuristic: take masks sorted by popcount, 
    # and greedily add those that maximize comparable pairs
    
    # Better initial solution: use the structure. In B_12, the number of comparable pairs
    # among a set of masks depends on the inclusion relationships.
    
    # Let's use a greedy construction as initial solution:
    # Start with empty family, at each step add the mask that maximizes new comparable pairs
    
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy initial solution
    in_family = [False] * num_masks
    family = []
    current_score = 0
    
    # Precompute for each mask, the list of masks it's comparable with (for greedy)
    # For greedy, when considering adding mask i, the gain is:
    # (number of j in family such that i strict subset j) + (number of j in family such that j strict subset i)
    
    # For efficiency in greedy, we can compute this on the fly
    
    for step in range(m):
        if time.time() > deadline - 10:
            break
        best_gain = -1
        best_mask = -1
        for i in range(num_masks):
            if in_family[i]:
                continue
            # Compute gain
            gain = 0
            for j in family:
                # Check if i strict subset of j or j strict subset of i
                if (i & j) == i and i != j:
                    gain += 1
                elif (i & j) == j and i != j:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = i
        if best_mask >= 0:
            in_family[best_mask] = True
            family.append(best_mask)
            current_score += best_gain
    
    # Now do simulated annealing
    # State: which masks are in the family (exactly m of them)
    # Move: swap a mask in the family with one not in the family
    
    # Precompute comparability counts for faster delta
    # For a given family state, if we remove mask i and add mask j:
    # Delta = (comparability of j with remaining family) - (comparability of i with remaining family)
    # where comparability of x with family = number of y in family such that x comparable with y
    
    # Let's maintain for each mask i, the number of comparable family members
    comp_count = [0] * num_masks
    for i in family:
        # For each mask k, if i is comparable with k, increment comp_count[k]
        # i comparable with k means (i & k) == i or (i & k) == k
        # i strict subset k: (i & k) == i and i != k
        # k strict subset i: (i & k) == k and i != k
        # So comparable: ((i & k) == i or (i & k) == k) and i != k
        # Note: if i == k, (i & k) == i == k, but we exclude i == k
        for k in range(num_masks):
            if k == i:
                continue
            if ((i & k) == i) or ((i & k) == k):
                comp_count[k] += 1
    
    # Verify current_score
    # current_score should be sum of comp_count[i] for i in family, divided by 2
    # because each comparable pair is counted twice
    verify_score = sum(comp_count[i] for i in family) // 2
    # This should equal current_score
    
    best_family = list(family)
    best_score = current_score
    
    # Simulated annealing
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    # We need to be careful with time
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        
        # Pick a random mask in the family to remove
        idx = random.randint(0, m - 1)
        remove_mask = family[idx]
        
        # Pick a random mask not in the family to add
        # We need an efficient way to pick a random mask not in the family
        # Use rejection sampling or maintain a list
        # For simplicity, rejection sampling
        while True:
            add_mask = random.randint(0, num_masks - 1)
            if not in_family[add_mask]:
                break
        
        if add_mask == remove_mask:
            continue
        
        # Compute delta
        # Remove remove_mask: lose comp_count[remove_mask] comparable pairs (with other family members)
        # But comp_count[remove_mask] includes comparabilities with all family members including itself? No, we excluded k==i.
        # So comp_count[remove_mask] is the number of family members comparable with remove_mask.
        # When we remove remove_mask, we lose those comparabilities.
        # But we also need to update comp_count for all other family members.
        
        # Actually, the delta in score when swapping remove_mask with add_mask:
        # New score = old_score - (comparabilities of remove_mask with family\{remove_mask}) 
        #              + (comparabilities of add_mask with family\{remove_mask})
        
        # Let's compute:
        # comp_with_others_remove = comp_count[remove_mask]  # this is # of family members comparable with remove_mask
        # But this includes comparabilities with all family members. Since we're removing remove_mask,
        # the comparabilities lost are exactly comp_count[remove_mask].
        
        # For add_mask, we need the number of family members (excluding remove_mask) comparable with add_mask.
        # comp_count[add_mask] currently includes comparabilities with all family members including remove_mask.
        # If add_mask is comparable with remove_mask, then comp_count[add_mask] includes 1 for remove_mask.
        
        comparable_with_remove = 0
        if ((add_mask & remove_mask) == add_mask) or ((add_mask & remove_mask) == remove_mask):
            if add_mask != remove_mask:
                comparable_with_remove = 1
        
        comp_with_others_add = comp_count[add_mask] - comparable_with_remove
        
        delta = comp_with_others_add - comp_with_others_remove
        
        # Acceptance
        if delta >= 0 or random.random() < np.exp(delta / T):
            # Accept the move
            # Update family
            family[idx] = add_mask
            in_family[remove_mask] = False
            in_family[add_mask] = True
            
            # Update comp_count
            # First, remove the effect of remove_mask from all comp_count
            for k in range(num_masks):
                if k == remove_mask:
                    continue
                if ((remove_mask & k) == remove_mask) or ((remove_mask & k) == k):
                    comp_count[k] -= 1
            comp_count[remove_mask] = 0
            
            # Add the effect of add_mask to all comp_count
            for k in range(num_masks):
                if k == add_mask:
                    continue
                if ((add_mask & k) == add_mask) or ((add_mask & k) == add_mask):
                    comp_count[k] += 1
            comp_count[add_mask] = 0
            
            # Set comp_count[add_mask] to the number of family members comparable with add_mask
            # This is comp_with_others_add (since we just computed it)
            # But wait, we need to recompute it properly
            # Actually, comp_count[add_mask] should be the number of family members comparable with add_mask
            # After the swap, the family has add_mask instead of remove_mask
            # The comparable family members for add_mask are those in family\{remove_mask} that are comparable with add_mask
            # That's comp_with_others_add
            comp_count[add_mask] = comp_with_others_add
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return the best family found
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    # Ensure exactly m distinct subsets
    assert len(result) == m
    # Ensure all are distinct
    assert len(set(tuple(s) for s in result)) == m
    
    return result

# EVOLVE-BLOCK-END
