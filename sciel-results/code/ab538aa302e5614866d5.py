import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    candidates = []
    mask_to_idx = {}
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        idx = len(candidates)
        candidates.append(mask)
        mask_to_idx[mask] = idx
        
    N = len(candidates)
    
    # Precompute sunflower partners
    # partners[i][j] = list of k such that candidates[i], candidates[j], candidates[k] form a sunflower
    # Note: k can be i or j? No, distinct sets.
    # If i, j, k form a sunflower, they are distinct.
    # We only store k != i and k != j.
    
    partners = [[[] for _ in range(N)] for _ in range(N)]
    
    # Optimization: 
    # For each pair (i, j), find all k.
    # This is O(N^3) which is 1820^3 ~ 6e9. Too slow.
    
    # Alternative:
    # For each pair (i, j), compute I = A & B.
    # Find all k such that A & k == I and B & k == I.
    # This is equivalent to k being a subset of U = I | ([16] \ (A | B)) and containing I.
    # And |k| = 4.
    # We can iterate through all candidates k and check? No, still O(N^3).
    
    # Better: Iterate k, then pairs (i, j)?
    # Or just compute on the fly in the SA loop?
    # If we compute on the fly, adding C requires checking all pairs (A, B) in F.
    # That's O(|F|^2) checks. |F| ~ 200, so 40,000 checks.
    # Each check is bitwise AND and comparison. Very fast.
    # 40,000 * 1000 steps = 40 million ops. Python might be slow.
    
    # Let's stick to the conflict_count method but compute partners lazily or precompute smartly.
    # Precomputation of partners:
    # For each i, j:
    #   I = A[i] & A[j]
    #   U = I | (0xFFFF - (A[i] | A[j]))  # Elements not in A or B, plus I
    #   We need k such that (A[i] & k) == I and (A[j] & k) == I.
    #   This implies k & ~(A[i] | A[j]) is free, but k must contain I and be disjoint from (A[i]\I) and (A[j]\I).
    #   Let Free = [16] \ (A[i] | A[j])
    #   Let Req = I
    #   k must be Req U S where S is a subset of Free, |S| = 4 - |Req|.
    #   We can generate all such k and map them to indices.
    
    # Let's try to precompute `partners` efficiently.
    # Number of pairs is ~1.6 million.
    # For each pair, generating k's takes ~70 iterations max.
    # Total ~ 100 million iterations. Might be too slow in Python (10-20s).
    # Let's see if we can skip precomputation and use the O(|F|^2) check.
    
    # O(|F|^2) check for adding C:
    # for i in F:
    #   for j in F:
    #     if A[i] & A[j] == A[i] & C and A[i] & A[j] == A[j] & C: return False
    # This is fast enough for |F| < 300.
    
    def can_add(family, c_idx):
        """
        Check if c_idx can be added to family.
        family is a list of indices.
        """
        c_mask = candidates[c_idx]
        n_fam = len(family)
        for i in range(n_fam):
            a_mask = candidates[family[i]]
            a_c = a_mask & c_mask
            for j in range(i + 1, n_fam):
                b_mask = candidates[family[j]]
                a_b = a_mask & b_mask
                if a_b == a_c:
                    b_c = b_mask & c_mask
                    if a_b == b_c:
                        return False
        return True

    def remove_from_family(family, idx_to_remove):
        """
        Remove idx_to_remove from family (list).
        """
        family.remove(idx_to_remove)

    # Simulated Annealing
    best_family = []
    best_size = 0
    
    # Initialize with a greedy solution
    # Random order
    order = list(range(N))
    random.shuffle(order)
    current_family = []
    for idx in order:
        if can_add(current_family, idx):
            current_family.append(idx)
    
    best_family = current_family.copy()
    best_size = len(current_family)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    current_size = len(current_family)
    
    # Set to allow fast membership check
    fam_set = set(current_family)
    
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        # Choose a move
        # 1. Add
        # 2. Remove
        # 3. Swap
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random element
            if current_size < N:
                # Pick random candidate not in family
                # To avoid infinite loops if no valid adds, pick from a random subset
                # Or just pick random idx and check if in fam_set
                for _ in range(10): # Try 10 random picks
                    new_idx = random.randint(0, N - 1)
                    if new_idx not in fam_set:
                        break
                else:
                    continue # Couldn't find one? Unlikely unless full.
                
                if can_add(current_family, new_idx):
                    # Accept
                    current_family.append(new_idx)
                    fam_set.add(new_idx)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_family = current_family.copy()
                else:
                    # Reject or accept with prob?
                    # Adding invalid item is not a valid state.
                    # So we just reject.
                    pass
                    
        elif move_type < 0.7:
            # Try to remove a random element
            if current_size > 0:
                rem_idx = random.choice(current_family)
                # Always accept removal? No, it reduces size.
                # But it might allow adding more later.
                # Accept with probability 1.0 to allow exploration?
                # Or accept with probability based on T?
                # Let's always accept removal if it helps escape local optima.
                # Actually, removing reduces objective.
                # Let's accept with prob 0.1 + T?
                if random.random() < 0.5: 
                    current_family.remove(rem_idx)
                    fam_set.remove(rem_idx)
                    current_size -= 1
                    
        else:
            # Swap: Remove A, Add B
            if current_size > 0:
                rem_idx = random.choice(current_family)
                # Find a valid add candidate
                # Temporarily remove rem_idx
                current_family.remove(rem_idx)
                fam_set.remove(rem_idx)
                
                # Try to add a random B
                added = False
                for _ in range(10):
                    add_idx = random.randint(0, N - 1)
                    if add_idx not in fam_set:
                        if can_add(current_family, add_idx):
                            current_family.append(add_idx)
                            fam_set.add(add_idx)
                            added = True
                            break
                
                if not added:
                    # Revert
                    current_family.append(rem_idx)
                    fam_set.add(rem_idx)
                else:
                    # Size is same. Accept with prob?
                    # Let's accept to allow mixing.
                    pass
        
        # Update temperature
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: restart with random shuffle of best_family?
            # Let's just keep going.
            
    # Return best_family
    result = []
    for idx in best_family:
        m = candidates[idx]
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result
