import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    def is_sunflower(A, B, C):
        """
        Check if A, B, C form a sunflower.
        They must be distinct.
        """
        if A == B or B == C or A == C:
            return False
        ab = A & B
        ac = A & C
        bc = B & C
        return ab == ac == bc

    def can_add(C, family):
        """
        Check if C can be added to family without forming a sunflower
        with any pair in family.
        """
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_sunflower(C, family[i], family[j]):
                    return False
        return True

    best_family_size = 0
    best_family = []
    
    # Try multiple random restarts
    # We don't know the exact max, but 250 is the cap.
    # Let's try 2000 restarts.
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle a copy to avoid modifying the original list order for subsequent restarts?
        # random.shuffle modifies in place. We should shuffle a copy or re-shuffle.
        # Since we shuffle all_masks in place, the next restart will start with the previously shuffled order.
        # This is fine, but let's ensure we are iterating over a random permutation.
        random.shuffle(all_masks)
        
        family = []
        
        for C in all_masks:
            if time.time() > deadline:
                break
                
            if can_add(C, family):
                family.append(C)
                
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
            # If we hit the theoretical max or a high enough number, we can stop early?
            # The task says max 250. If we reach 250, we stop.
            if best_family_size >= 250:
                break

    # If best_family is still empty (shouldn't happen), add one random valid set
    if not best_family:
        best_family = [all_masks[0]]

    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
