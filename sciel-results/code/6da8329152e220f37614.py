import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n_max = 256
    odds = [i for i in range(1, n_max + 1) if i % 2 != 0]
    evens = [i for i in range(1, n_max + 1) if i % 2 == 0]
    
    num_odds = len(odds)
    num_evens = len(evens)
    
    # Map odd value to index in odds list for bitmasking
    odd_index = {v: i for i, v in enumerate(odds)}
    
    # Precompute conflicts
    # conflict_e_o[e_idx] is a bitmask of odds that conflict with evens[e_idx]
    # conflict_e_e[e_idx1] is a bitmask of evens that conflict with evens[e_idx1]
    
    conflict_e_o = [0] * num_evens
    conflict_e_e = [0] * num_evens
    
    for i, e in enumerate(evens):
        for j, o in enumerate(odds):
            if (e * o) % (e + o) == 0:
                conflict_e_o[i] |= (1 << j)
        
        for k in range(i + 1, num_evens):
            e2 = evens[k]
            if (e * e2) % (e + e2) == 0:
                conflict_e_e[i] |= (1 << k)
                conflict_e_e[k] |= (1 << i)
    
    def popcount(x):
        return x.bit_count()
    
    # Objective function for a set of selected evens (represented as a bitmask over evens)
    # Returns the total size of the solution
    def evaluate(even_mask):
        if even_mask == 0:
            return num_odds
        
        # Check if even_mask is a valid independent set in E
        # We assume the caller maintains validity, or we check here
        # For speed in SA, we might skip full check if moves are local, 
        # but for safety in this implementation, we'll trust the moves.
        
        # Calculate union of conflicts with odds
        removed_odds_mask = 0
        m = even_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            removed_odds_mask |= conflict_e_o[idx]
            m ^= lsb
            
        num_removed_odds = popcount(removed_odds_mask)
        num_selected_evens = popcount(even_mask)
        
        return num_odds - num_removed_odds + num_selected_evens

    # Initialize
    best_mask = 0
    best_score = num_odds
    
    # Greedy Initialization
    # Try adding evens that have low conflict cost
    # Cost of adding e is (num_new_odds_removed - 1)
    # We want to minimize cost.
    
    current_mask = 0
    current_removed = 0
    current_score = num_odds
    
    # Sort evens by number of conflicts with odds (ascending)
    # Fewer conflicts are "cheaper" to add
    evens_by_conflict = sorted(range(num_evens), key=lambda i: popcount(conflict_e_o[i]))
    
    for idx in evens_by_conflict:
        if time.time() > deadline - 20.0:
            break
        # Check if idx conflicts with current evens
        if conflict_e_e[idx] & current_mask:
            continue
        
        # Calculate new removed odds
        new_removed = current_removed | conflict_e_o[idx]
        new_num_removed = popcount(new_removed)
        old_num_removed = popcount(current_removed)
        
        # Delta score: +1 for even, -(new_removed - old_removed) for odds
        delta = 1 - (new_num_removed - old_num_removed)
        
        if delta > 0:
            current_mask |= (1 << idx)
            current_removed = new_removed
            current_score += delta
    
    if current_score > best_score:
        best_score = current_score
        best_mask = current_mask
    
    # Simulated Annealing on the even mask
    # State: even_mask
    # Moves: Add, Remove, Swap
    
    T = 5.0
    T_min = 0.001
    cooling = 0.995
    
    current_mask = best_mask
    current_score = evaluate(current_mask)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5.0 and T > T_min and no_improve < max_no_improve:
        r = random.random()
        new_mask = current_mask
        valid = False
        
        if r < 0.4:
            # Add a random even
            idx = random.randint(0, num_evens - 1)
            if not (current_mask & (1 << idx)):
                if not (conflict_e_e[idx] & current_mask):
                    new_mask = current_mask | (1 << idx)
                    valid = True
        elif r < 0.7:
            # Remove a random even
            if current_mask:
                # Pick a random set bit
                m = current_mask
                # Simple random selection among set bits
                bits = []
                mm = m
                while mm:
                    lsb = mm & -mm
                    bits.append(lsb.bit_length() - 1)
                    mm ^= lsb
                if bits:
                    idx = random.choice(bits)
                    new_mask = current_mask & ~(1 << idx)
                    valid = True
        else:
            # Swap: Remove one, Add one
            if current_mask:
                m = current_mask
                bits = []
                mm = m
                while mm:
                    lsb = mm & -mm
                    bits.append(lsb.bit_length() - 1)
                    mm ^= lsb
                if bits:
                    idx_rm = random.choice(bits)
                    idx_add = random.randint(0, num_evens - 1)
                    if idx_rm != idx_add:
                        if not (current_mask & (1 << idx_add)):
                            # Check if idx_add conflicts with (current_mask without idx_rm)
                            mask_without_rm = current_mask & ~(1 << idx_rm)
                            if not (conflict_e_e[idx_add] & mask_without_rm):
                                new_mask = mask_without_rm | (1 << idx_add)
                                valid = True
        
        if not valid:
            continue
            
        new_score = evaluate(new_mask)
        
        if new_score > current_score:
            current_mask = new_mask
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        elif new_score < current_score:
            if random.random() < math.exp((new_score - current_score) / T):
                current_mask = new_mask
                current_score = new_score
                no_improve += 1
        else:
            current_mask = new_mask
            no_improve += 1
            
        T *= cooling
        
        # Periodic Greedy Completion
        if no_improve % 50 == 0:
            # Try to add any even that improves score
            improved = True
            while improved:
                improved = False
                for idx in range(num_evens):
                    if time.time() > deadline - 5.0:
                        break
                    if current_mask & (1 << idx):
                        continue
                    if conflict_e_e[idx] & current_mask:
                        continue
                    # Check score improvement
                    old_score = current_score
                    new_mask = current_mask | (1 << idx)
                    new_score = evaluate(new_mask)
                    if new_score > old_score:
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask

    # Reconstruct the set
    result = set(odds)
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.add(evens[idx])
        m ^= lsb
        
    # Remove odds that conflict with selected evens
    removed_odds_mask = 0
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        removed_odds_mask |= conflict_e_o[idx]
        m ^= lsb
        
    for i in range(num_odds):
        if removed_odds_mask & (1 << i):
            result.remove(odds[i])
            
    return list(result)

# EVOLVE-BLOCK-END
