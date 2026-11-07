# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 12
    m = 40
    
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    max_time = 85.0
    
    # 1. Generate all 2^12 subsets as masks
    # We want to maximize comparable pairs. 
    # Theoretically, if we could pick 40 sets all from a single chain of length 13, 
    # we would have C(40,2) pairs if we could repeat, but we can't.
    # We must pick 40 distinct sets.
    
    # A good strategy is to pick sets that are "close" to each other in the subset lattice.
    # Specifically, a chain of sets S_0 < S_1 < ... < S_12 has 13 sets.
    # Any two sets in a chain are comparable.
    # If we pick sets from multiple chains that share elements, we might get more comparability.
    
    # Let's try a deterministic construction based on "layers" or specific structures.
    # However, the most robust approach for "maximize comparable pairs" with distinct sets 
    # is often to pick a large subset of a total order (chain) and then add sets 
    # that are comparable to as many existing sets as possible.
    
    # Since 40 > 13 (length of longest chain), we cannot just use one chain.
    # We need to mix chains or use a "grid" like structure.
    
    # Strategy:
    # 1. Start with a strong initial set of masks.
    # 2. Use a greedy hill-climbing approach to improve.
    #    For each set, check if replacing it with a better candidate increases the total count.
    
    # Initial Construction:
    # Let's generate a few long chains and merge them.
    # Or, simpler: Start with random distinct masks, then optimize.
    
    # To be faster and more deterministic, let's use a specific heuristic initialization:
    # Pick the 40 masks that are "central" in the lattice? No, comparability is about chains.
    # Let's pick masks from a few random permutations' chains.
    
    best_masks = []
    best_score = -1
    
    # Run a few deterministic-ish restarts with different random seeds for initialization
    # but use a very aggressive greedy optimizer.
    
    for restart in range(50):
        if time.time() - start_time > max_time - 5.0:
            break
            
        # Initialization:
        # Create a random permutation to define a chain
        perm = list(range(n))
        random.shuffle(perm)
        
        # The chain is: {perm[0]}, {perm[0], perm[1]}, ..., {perm[0]...perm[11]}
        # Plus empty set.
        chain_masks = [0]
        cur = 0
        for p in perm:
            cur |= (1 << p)
            chain_masks.append(cur)
            
        # We have 13 masks. We need 40.
        # Let's pick 3 different random chains and take the union of their masks, 
        # ensuring distinctness, until we have 40.
        init_masks = []
        seen = set()
        
        # Add first chain
        for mask in chain_masks:
            if mask not in seen:
                seen.add(mask)
                init_masks.append(mask)
                
        # Add more random chains until we have 40
        while len(init_masks) < m:
            # Generate a random chain
            perm2 = list(range(n))
            random.shuffle(perm2)
            cur2 = 0
            # Add empty set if not already there (it is, from first chain)
            for p in perm2:
                cur2 |= (1 << p)
                if cur2 not in seen:
                    seen.add(cur2)
                    init_masks.append(cur2)
                    if len(init_masks) == m:
                        break
            
            # If we still don't have enough, add random sets
            if len(init_masks) < m:
                while len(init_masks) < m:
                    size = random.randint(0, n)
                    elems = random.sample(range(n), size)
                    mask = 0
                    for e in elems:
                        mask |= (1 << e)
                    if mask not in seen:
                        seen.add(mask)
                        init_masks.append(mask)
        
        current_masks = init_masks[:m]
        
        # Calculate initial score
        def count_pairs(masks):
            cnt = 0
            m_len = len(masks)
            for i in range(m_len):
                mi = masks[i]
                for j in range(i+1, m_len):
                    mj = masks[j]
                    if (mi & mj) == mi or (mi & mj) == mj:
                        cnt += 1
            return cnt
        
        current_score = count_pairs(current_masks)
        
        # Greedy Optimization:
        # Iterate through all positions. For each position, try to replace the mask 
        # with a different mask that yields the highest gain.
        # To make this efficient, we don't check all 4096 masks for every position in every iteration.
        # We check a subset of "promising" masks or just do a few passes.
        
        improved = True
        passes = 0
        max_passes = 20
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            if time.time() - start_time > max_time - 2.0:
                break
                
            for i in range(m):
                old_mask = current_masks[i]
                old_score = current_score
                
                # Calculate gain/loss for removing old_mask
                # delta_remove = - (number of sets comparable to old_mask)
                # delta_add = + (number of sets comparable to new_mask)
                # We want to maximize sum.
                
                # Let's find the best replacement mask for position i.
                # We can't try all 4096 masks * 40 positions * 20 passes * 50 restarts in 90s easily if count_pairs is O(m^2).
                # But count_pairs is O(40^2) = 1600 ops.
                # 4096 candidates * 1600 ops = ~6.5M ops per position.
                # 40 positions * 6.5M = 260M ops per pass. Too slow in Python.
                
                # Optimization: 
                # Instead of recalculating full score, calculate delta.
                # delta = comparability(new, others) - comparability(old, others)
                # comparability(x, others) = sum(1 for y in others if x<=y or y<=x)
                
                # Pre-compute "comparability count" for each mask against the rest of the set?
                # That's still O(m) per mask.
                
                # Let's try a limited set of candidates for replacement:
                # 1. Masks that are subsets or supersets of the current mask (local moves).
                # 2. Masks from the "chains" we generated initially.
                # 3. Random samples.
                
                candidates = set()
                
                # Add subsets and supersets of old_mask
                # Subsets
                sub = old_mask
                while True:
                    candidates.add(sub)
                    if sub == 0: break
                    # Remove one bit
                    # To be efficient, just add a few random subsets
                    break 
                    
                # Let's just try random candidates and some structured ones.
                # Structured: 
                # Pick a random chain mask from our initial generation pool? 
                # We didn't store them. Let's just generate a few random ones.
                
                # Better: Use the fact that if we replace mask[i] with mask[j] (swapping), 
                # we can check if that helps. But we need distinct sets.
                
                # Let's try a simple local search:
                # Pick a random candidate mask (not in current_masks).
                # If it increases the score, accept.
                
                # To speed up, we will only try a few candidates per position per pass.
                found_better = False
                best_candidate = -1
                best_candidate_mask = -1
                
                # Try 10 random candidates
                for _ in range(10):
                    # Generate a random mask
                    size = random.randint(0, n)
                    elems = random.sample(range(n), size)
                    new_mask = 0
                    for e in elems:
                        new_mask |= (1 << e)
                    
                    if new_mask in current_masks:
                        continue
                    if new_mask == old_mask:
                        continue
                    
                    # Calculate delta
                    # Remove old_mask contribution
                    # Add new_mask contribution
                    
                    # To do this fast, we can precompute the "compatibility" of each mask in the set with every other mask?
                    # No, that's O(m^2) memory and time to build.
                    
                    # Let's just compute the delta directly.
                    # delta = 0
                    # for k in range(m):
                    #    if k == i: continue
                    #    mk = current_masks[k]
                    #    # old
                    #    if (old_mask & mk) == old_mask or (old_mask & mk) == mk:
                    #        delta -= 1
                    #    # new
                    #    if (new_mask & mk) == new_mask or (new_mask & mk) == new_mask:
                    #        delta += 1
                    
                    # This is O(m) = 40 ops per candidate.
                    # 10 candidates * 40 ops = 400 ops per position.
                    # 40 positions * 400 ops = 16,000 ops per pass.
                    # 20 passes * 16,000 ops = 320,000 ops per restart.
                    # 50 restarts * 320,000 = 16,000,000 ops.
                    # This is very fast in Python (< 1s).
                    
                    delta = 0
                    for k in range(m):
                        if k == i:
                            continue
                        mk = current_masks[k]
                        
                        # Check old
                        if (old_mask & mk) == old_mask or (old_mask & mk) == mk:
                            delta -= 1
                            
                        # Check new
                        if (new_mask & mk) == new_mask or (new_mask & mk) == new_mask:
                            delta += 1
                    
                    if delta > 0 and delta > best_candidate:
                        best_candidate = delta
                        best_candidate_mask = new_mask
                        found_better = True
                        
                if found_better:
                    current_masks[i] = best_candidate_mask
                    current_score += best_candidate
                    improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_masks = list(current_masks)
            
    # Convert best_masks to list of lists
    result = []
    for mask in best_masks:
        s = []
        for b in range(n):
            if mask & (1 << b):
                s.append(b)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
