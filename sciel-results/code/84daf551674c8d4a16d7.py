import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    n = 12
    m = 40
    
    # Precompute all subsets as bitmasks
    # Each subset is represented as a bitmask from 0 to 2^12 - 1
    # Subset A is a subset of B iff (A & B) == A
    
    def count_comparable(family_bitmasks):
        """Count comparable pairs in a family of bitmasks."""
        count = 0
        k = len(family_bitmasks)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = family_bitmasks[i], family_bitmasks[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
    
    def bitmask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]
    
    # Greedy construction
    def greedy_construct():
        family = []
        # Start with a chain: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
        # That's 13 sets in a chain
        chain = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]
        family.extend(chain)
        
        all_masks = set(range(1 << n))
        all_masks.difference_update(family)
        
        while len(family) < m and time.time() < deadline:
            best_mask = None
            best_gain = -1
            current_count = count_comparable(family)
            
            for mask in all_masks:
                # Count how many existing sets are comparable with mask
                gain = 0
                for existing in family:
                    if (mask & existing) == mask or (mask & existing) == existing:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            
            if best_mask is None:
                break
            
            family.append(best_mask)
            all_masks.discard(best_mask)
        
        return family
    
    # Local search: try swapping one set for another to improve
    def local_search(family, max_iterations=5000):
        family = list(family)
        all_masks = set(range(1 << n))
        in_family = set(family)
        all_masks.difference_update(in_family)
        
        current_score = count_comparable(family)
        best_family = list(family)
        best_score = current_score
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random index to replace
            idx = random.randint(0, len(family) - 1)
            old_mask = family[idx]
            
            # Find the best replacement
            best_new_mask = None
            best_new_score = current_score
            
            # Sample some candidates for speed
            candidates = list(all_masks)
            random.shuffle(candidates)
            candidates = candidates[:200]
            
            for new_mask in candidates:
                if new_mask == old_mask:
                    continue
                # Compute new score efficiently
                # Remove old_mask, add new_mask
                # Score change: lose comparabilities with old_mask, gain with new_mask
                gain = 0
                for j, other in enumerate(family):
                    if j == idx:
                        continue
                    # Was old_mask comparable with other?
                    was_comp = (old_mask & other) == old_mask or (old_mask & other) == other
                    # Is new_mask comparable with other?
                    is_comp = (new_mask & other) == new_mask or (new_mask & other) == other
                    if is_comp and not was_comp:
                        gain += 1
                    elif was_comp and not is_comp:
                        gain -= 1
                
                new_score = current_score + gain
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_mask = new_mask
            
            if best_new_mask is not None and best_new_score > current_score:
                family[idx] = best_new_mask
                all_masks.discard(old_mask)
                all_masks.add(best_new_mask)
                in_family.discard(old_mask)
                in_family.add(best_new_mask)
                current_score = best_new_score
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
            else:
                # Try a random move (accept with some probability for diversity)
                if candidates:
                    new_mask = random.choice(candidates)
                    if new_mask != old_mask:
                        # Just accept to explore
                        family[idx] = new_mask
                        all_masks.discard(old_mask)
                        all_masks.add(new_mask)
                        in_family.discard(old_mask)
                        in_family.add(new_mask)
                        current_score = count_comparable(family)
                        if current_score > best_score:
                            best_score = current_score
                            best_family = list(family)
        
        return best_family, best_score
    
    # Try multiple random restarts with different initial chains
    best_family = None
    best_score = -1
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Different initial chains
        if seed % 3 == 0:
            # Standard chain
            chain = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]
        elif seed % 3 == 1:
            # Chain with different element ordering
            perm = list(range(n))
            random.shuffle(perm)
            chain = []
            mask = 0
            chain.append(0)
            for i in range(n):
                mask |= (1 << perm[i])
                chain.append(mask)
        else:
            # Start with empty and full, then greedy
            chain = [0, (1 << n) - 1]
        
        family = list(chain)
        all_masks = set(range(1 << n))
        all_masks.difference_update(family)
        
        # Greedy fill
        while len(family) < m and time.time() < deadline:
            best_mask = None
            best_gain = -1
            
            # Sample candidates for speed
            candidates = list(all_masks)
            random.shuffle(candidates)
            candidates = candidates[:500]
            
            for mask in candidates:
                gain = 0
                for existing in family:
                    if (mask & existing) == mask or (mask & existing) == existing:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            
            if best_mask is None:
                break
            
            family.append(best_mask)
            all_masks.discard(best_mask)
        
        if len(family) < m:
            # Fill remaining with random
            remaining = list(all_masks)
            random.shuffle(remaining)
            family.extend(remaining[:m - len(family)])
        
        # Local search
        family, score = local_search(family, max_iterations=3000)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Convert to list of lists
    result = [bitmask_to_list(mask) for mask in best_family]
    
    # Ensure types are correct
    final_result = []
    for s in result:
        final_result.append([int(x) for x in s])
    
    return final_result
