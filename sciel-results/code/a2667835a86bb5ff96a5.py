import time
import random
import math

def construct():
    n = 12
    m = 40
    # Safety margin well under the 90s wall-clock limit
    deadline = time.time() + 82.0
    total_masks = 1 << n  # 4096
    
    # Precompute comparability matrix for O(1) lookup
    # comp[a][b] is True if a is a subset of b (a <= b in subset relation)
    # We need strict subset, so a < b means a is subset of b and a != b
    # But for pairs {a, b}, they are comparable if a <= b or b <= a
    # Since all masks are distinct, comparable means a < b or b < a (strict)
    
    # Precompute: subset[a][b] = 1 if a is subset of b, else 0
    # This is a 4096x4096 matrix, which is 16M entries.
    # Using a list of bytearrays or just a flat list of booleans.
    # Actually, we can compute it on the fly for the 40 selected masks,
    # but precomputing for all 4096 allows faster evaluation of any set.
    
    # Let's precompute a matrix where comp[i][j] is True if i is a strict subset of j
    # Or simply: i is subset of j (including equality)
    # Memory: 4096 * 4096 bytes ~ 16MB. This is fine.
    
    # Optimization: Instead of a full matrix, we can use bitset operations or just
    # precompute for the current candidate set.
    # Given the high cost of 4096x4096 initialization (16M ops), let's see if it's worth it.
    # 16M operations in Python might take ~1-2 seconds.
    # Let's try a hybrid: precompute subset relations for all pairs?
    # Actually, calculating (a & b) == a is very fast.
    # Let's stick to calculating on the fly but optimize the inner loop of SA.
    
    # A better optimization: Precompute the "subset count" or use a smarter scoring?
    # No, the score is pairwise.
    
    # Let's use a precomputed matrix for speed during SA.
    # 4096 * 4096 is 16,777,216 entries.
    # A list of lists of ints/bools in Python is heavy.
    # Let's use a list of bitsets? Or just a 2D list of small ints.
    # Or, we can just rely on the fast bitwise check.
    # (a & b) == a is extremely fast in Python.
    
    # The bottleneck in the previous code was the O(m) loop for score_delta.
    # m=40, so it's 40 comparisons per step.
    # If we do 100k steps, that's 4M comparisons. Very fast.
    # The previous code did 20000 steps * 100 restarts = 2M steps max.
    # With the deadline, it probably ran fewer.
    
    # Let's improve the search strategy:
    # 1. Better initialization: Start with a known good structure.
    #    A chain of length 12 gives 11 pairs.
    #    We can extend chains by duplicating elements? No, must be distinct.
    #    We can create multiple chains.
    #    E.g., Chain 1: {}, {0}, {0,1}, ...
    #    Chain 2: {2}, {2,3}, ...
    #    But we need to maximize cross-chain comparabilities too?
    #    If chains are disjoint, no cross comparabilities.
    #    If chains overlap, we get more.
    
    # 2. Use a "pairwise swap" or "insertion" heuristic?
    #    The current code swaps a mask with a new random/mutated mask.
    #    This is effectively a move in the space of 40-tuples.
    
    # 3. Improve the mutation:
    #    Instead of random bit flips, try to add/remove elements that maximize local gain.
    #    Or use a "greedy hill climb" phase before SA.
    
    # Let's implement a faster, more aggressive SA with:
    # - Precomputed subset check function (inline)
    # - Better initialization (greedy construction)
    # - More steps per restart
    # - Adaptive temperature
    
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def is_comparable(a, b):
        # a and b are distinct masks
        return (a & b) == a or (a & b) == b
    
    def calculate_score(masks):
        score = 0
        for i in range(m):
            a = masks[i]
            for j in range(i+1, m):
                b = masks[j]
                if (a & b) == a or (a & b) == b:
                    score += 1
        return score
    
    def score_delta(swap_idx, new_mask, masks):
        old_mask = masks[swap_idx]
        delta = 0
        for j in range(m):
            if j == swap_idx:
                continue
            other = masks[j]
            # Check new
            if (new_mask & other) == new_mask or (new_mask & other) == other:
                delta += 1
            # Check old
            if (old_mask & other) == old_mask or (old_mask & other) == other:
                delta -= 1
        return delta
    
    best_score = -1
    best_masks = None
    
    # Greedy Initialization:
    # Start with empty list.
    # Repeatedly add the mask that maximizes the score increase.
    # This is O(m * 4096) per step, so 40 * 4096 * 40 ~ 6.5M ops. Fast.
    
    def greedy_init():
        current_masks = []
        used = set()
        for _ in range(m):
            best_gain = -1
            best_mask = None
            # Sample a subset of masks to speed up? Or check all?
            # 4096 checks * 40 existing masks = 163k ops per addition.
            # 40 additions = 6.5M ops. Totally fine.
            for mask in range(total_masks):
                if mask in used:
                    continue
                # Calculate gain
                gain = 0
                for other in current_masks:
                    if (mask & other) == mask or (mask & other) == other:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask is not None:
                current_masks.append(best_mask)
                used.add(best_mask)
            else:
                # Should not happen
                break
        return current_masks
    
    # Run a few greedy inits with different random seeds?
    # Or just one greedy init, then SA.
    
    # Let's try multiple strategies:
    # 1. Pure Greedy
    # 2. Random + SA
    # 3. Greedy + SA
    
    num_restarts = 50
    max_steps_per_run = 50000 # Increased steps
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialization Strategy
        if restart % 3 == 0:
            # Greedy
            masks = greedy_init()
        elif restart % 3 == 1:
            # Random
            masks = list(random.sample(range(total_masks), m))
        else:
            # Greedy but with some random noise?
            # Or a "chain" based init
            masks = []
            used = set()
            # Create 4 chains of length 10?
            # Chain k: start with random bit set, then add bits
            for c in range(4):
                chain = []
                current = 0
                # Pick a random starting element
                if random.random() < 0.5:
                    current = 1 << random.randint(0, n-1)
                else:
                    current = 0
                chain.append(current)
                used.add(current)
                for _ in range(9):
                    # Add a random bit not in current
                    available = [i for i in range(n) if not (current >> i) & 1]
                    if not available:
                        break
                    bit = random.choice(available)
                    current = current | (1 << bit)
                    if current not in used:
                        chain.append(current)
                        used.add(current)
                masks.extend(chain)
            
            # Fill remaining with random
            while len(masks) < m:
                nm = random.randint(0, total_masks - 1)
                if nm not in used:
                    masks.append(nm)
                    used.add(nm)
            random.shuffle(masks)
            masks = masks[:m]
        
        current_score = calculate_score(masks)
        
        # Simulated Annealing
        T = 5.0 # Start lower, we have good inits
        min_T = 0.01
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 20000 # Allow longer plateaus
        
        local_best_score = current_score
        local_best_masks = masks[:]
        
        steps_in_run = 0
        
        while steps_in_run < max_steps_per_run and time.time() < deadline:
            steps_in_run += 1
            no_improve_count += 1
            
            i = random.randint(0, m - 1)
            old_mask = masks[i]
            
            # Mutation
            if random.random() < 0.85:
                new_mask = old_mask
                # Flip 1-2 bits
                num_flips = random.randint(1, 2)
                for _ in range(num_flips):
                    bit = random.randint(0, n - 1)
                    new_mask ^= (1 << bit)
            else:
                new_mask = random.randint(0, total_masks - 1)
            
            if new_mask == old_mask:
                continue
                
            # Check duplicate
            # This is an O(m) check.
            # We can maintain a set of current masks for O(1) check.
            # But updating the set on every accept is easy.
            # Let's use a set for duplicate checking.
            
            # For now, linear check. m=40 is small.
            duplicate = False
            for k in range(m):
                if k != i and masks[k] == new_mask:
                    duplicate = True
                    break
            
            if duplicate:
                continue
            
            delta = score_delta(i, new_mask, masks)
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                masks[i] = new_mask
                current_score += delta
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_masks = masks[:]
                    no_improve_count = 0
                    
            T *= cooling
            
            if no_improve_count > max_no_improve:
                break
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_masks = local_best_masks[:]
            
    if best_masks is None:
        # Fallback
        best_masks = list(range(m))
        
    return [mask_to_list(mask) for mask in best_masks]
