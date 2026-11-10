import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n

    # Precompute comparability for all pairs?
    # 4096 * 4096 is ~16 million checks.
    # (a & b) == a or (a & b) == b is fast.
    # Let's precompute a bitset of comparable masks for each mask.
    # comp_bitset[a] is an integer where bit b is 1 if a and b are comparable.
    # This uses 4096 * 8 bytes (if using Python ints which are arbitrary precision, 
    # but 4096 bits is 512 bytes, so 4096 * 512 = 2MB).
    # Actually, Python ints are efficient for this.
    
    comp_bitsets = [0] * num_masks
    for a in range(num_masks):
        bs = 0
        for b in range(num_masks):
            if a == b:
                continue
            if (a & b) == a or (a & b) == b:
                bs |= (1 << b)
        comp_bitsets[a] = bs

    def is_comparable_fast(a, b):
        # Check if bit b is set in comp_bitsets[a]
        return (comp_bitsets[a] >> b) & 1

    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable_fast(masks[i], masks[j]):
                    s += 1
        return s

    def greedy_init(seed_val):
        rng = random.Random(seed_val)
        # Start with empty set
        current_masks = []
        used_masks = set()
        
        # To make it deterministic and strong, we can start with a chain
        # or just pick the mask with max potential.
        # Let's pick masks greedily based on degree in the comparability graph.
        # Or simply, start with a random chain to ensure some structure.
        
        # Better: Start with a random set of 10, then greedily add 30.
        # Or start with 0 and full_mask?
        
        # Let's use a greedy approach from scratch.
        # Candidate pool: all 4096 masks.
        # At each step, pick the mask that maximizes comparability with existing.
        
        # To avoid getting stuck in a single chain immediately, 
        # let's randomize the initial few picks.
        
        # Pick 5 random distinct masks
        candidates = list(range(num_masks))
        rng.shuffle(candidates)
        
        for k in range(5):
            mask = candidates[k]
            current_masks.append(mask)
            used_masks.add(mask)
            
        # Greedily add remaining
        for _ in range(m - 5):
            best_mask = -1
            best_score = -1
            # Optimization: We don't need to check all 4096 every time if we are smart,
            # but 4096 * 40 * 35 is ~5.7 million ops, which is fast in Python if optimized.
            # However, iterating 4096 masks and checking 5-39 comparables is 4096*40 = 163k ops per step.
            # 35 steps * 163k = 5.7M ops. This is fine.
            
            for mask in candidates:
                if mask in used_masks:
                    continue
                score = 0
                for cm in current_masks:
                    if is_comparable_fast(mask, cm):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            current_masks.append(best_mask)
            used_masks.add(best_mask)
            
        return current_masks

    best_global_score = -1
    best_global_solution = None
    
    # Run multiple restarts of Simulated Annealing
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        current_masks = greedy_init(42 + r)
        used_masks = set(current_masks)
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute list of all masks for random selection
        all_masks = list(range(num_masks))
        
        steps = 0
        max_steps = 200000
        
        while T > min_T and steps < max_steps:
            if time.time() > deadline - 2.0:
                break
                
            steps += 1
            
            # Pick a random index
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Pick a new mask
            # To improve mixing, we can bias towards masks similar to old_mask 
            # or just pick uniformly. Uniform is safer for exploration.
            new_mask = random.choice(all_masks)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue
                
            # Calculate Delta
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mask_j = current_masks[j]
                old_comp = is_comparable_fast(old_mask, mask_j)
                new_comp = is_comparable_fast(new_mask, mask_j)
                delta += (1 if new_comp else 0) - (1 if old_comp else 0)
                
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # Avoid log of 0 or negative issues, delta is integer
                if delta == 0:
                    accept = True # Neutral move, always accept to explore
                else:
                    # P = exp(delta / T)
                    # delta is negative
                    accept = random.random() < math.exp(delta / T)
            
            if accept:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                
                if cur_score > best_global_score:
                    best_global_score = cur_score
                    best_global_solution = list(current_masks)
            
            T *= cooling

    # Convert best solution to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
