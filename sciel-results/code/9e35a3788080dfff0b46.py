import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    
    # Helper to check strict subset
    def is_subset(a, b):
        return (a & b) == a and a != b

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b

    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s

    def greedy_chain_init(seed_val):
        """Initialize with a chain structure, then greedily add sets that maximize comparability."""
        rng = random.Random(seed_val)
        # Start with a random chain: pick a random ordering of elements
        elem_order = list(range(n))
        rng.shuffle(elem_order)
        
        # Build chain: empty set, then add elements one by one
        chain = [0]
        cur = 0
        for e in elem_order:
            cur |= (1 << e)
            chain.append(cur)
        
        current_masks = list(chain)  # 13 sets
        used_masks = set(current_masks)
        
        # Greedily add sets that maximize immediate comparability
        for _ in range(m - len(current_masks)):
            best_mask = -1
            best_score = -1
            for mask in range(1 << n):
                if mask in used_masks:
                    continue
                # Count comparables with existing
                score = 0
                for cm in current_masks:
                    if is_comparable(mask, cm):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        
        return current_masks

    # Use greedy chain initialization, then local search to polish
    random.seed(42)
    
    # Precompute all masks
    all_masks = list(range(1 << n))
    
    # Precompute bit positions for each mask for faster moves
    # For each mask, store list of set bits and unset bits
    bit_sets = {}
    for mask in all_masks:
        ones = [i for i in range(n) if (mask >> i) & 1]
        zeros = [i for i in range(n) if not ((mask >> i) & 1)]
        bit_sets[mask] = (ones, zeros)
    
    best_global_score = -1
    best_global_solution = None
    
    # Try multiple initializations: greedy chain with different random seeds
    num_restarts = 8
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Greedy chain initialization with different random seeds
        current_masks = greedy_chain_init(42 + r)
        used_masks = set(current_masks)
        
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
        
        # Local Search: Hill Climbing with faster delta calculation
        stuck = 0
        max_steps_per_restart = 30000
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate a new mask using precomputed bit sets
            old_ones, old_zeros = bit_sets[old_mask]
            new_mask = old_mask
            
            op = random.random()
            if op < 0.35:
                # Add bit
                if old_zeros:
                    bit = random.choice(old_zeros)
                    new_mask = old_mask | (1 << bit)
            elif op < 0.70:
                # Remove bit
                if old_ones:
                    bit = random.choice(old_ones)
                    new_mask = old_mask ^ (1 << bit)
            elif op < 0.85:
                # Swap bit
                if old_ones and old_zeros:
                    bo = random.choice(old_ones)
                    bz = random.choice(old_zeros)
                    new_mask = (old_mask ^ (1 << bo)) | (1 << bz)
            else:
                # Random jump
                new_mask = random.randint(0, full_mask)
                
            if new_mask == old_mask:
                continue
                
            if new_mask in used_masks:
                continue
                
            # Calculate Delta efficiently
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mask_j = current_masks[j]
                old_comp = is_comparable(old_mask, mask_j)
                new_comp = is_comparable(new_mask, mask_j)
                delta += (1 if new_comp else 0) - (1 if old_comp else 0)
                
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                stuck = 0
                if cur_score > best_global_score:
                    best_global_score = cur_score
                    best_global_solution = list(current_masks)
            else:
                stuck += 1
                if stuck > 50:
                    # Random kick
                    idx = random.randint(0, m - 1)
                    old_mask = current_masks[idx]
                    while True:
                        new_mask = random.randint(0, full_mask)
                        if new_mask not in used_masks:
                            break
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[idx] = new_mask
                    cur_score = calc_score(current_masks)
                    stuck = 0
        
    # Convert best solution to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    # Ensure types are correct: integers must be Python int, sequences must be list
    # The above already produces Python ints and lists, so this should be fine
    return result
