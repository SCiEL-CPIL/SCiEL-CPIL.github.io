"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    n = 12
    m = 40
    
    all_masks = list(range(1 << n))
    
    def count_comparable(masks):
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                a = masks[i]
                b = masks[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count

    # Greedy Initialization using bitmasks
    current_masks = []
    used = set()
    
    for _ in range(m):
        best_score = -1
        best_mask = None
        for s in all_masks:
            if s in used:
                continue
            gain = 0
            for t in current_masks:
                if (s & t) == s and s != t:
                    gain += 1
                elif (s & t) == t and s != t:
                    gain += 1
            if gain > best_score:
                best_score = gain
                best_mask = s
        
        if best_mask is not None:
            current_masks.append(best_mask)
            used.add(best_mask)
            
    current_score = count_comparable(current_masks)
    best_masks = current_masks[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    max_iterations = 100000
    no_improve_count = 0
    max_no_improve = 5000
    
    for iter in range(max_iterations):
        if T < min_T or no_improve_count > max_no_improve:
            break
            
        i = random.randint(0, m - 1)
        old_mask = current_masks[i]
        
        # Pick a random new mask
        new_mask = random.choice(all_masks)
        if new_mask in used:
            continue
            
        delta = 0
        for j in range(m):
            if j == i:
                continue
            other = current_masks[j]
            
            # Old contribution
            if (old_mask & other) == old_mask and old_mask != other:
                delta -= 1
            elif (old_mask & other) == other and old_mask != other:
                delta -= 1
                
            # New contribution
            if (new_mask & other) == new_mask and new_mask != other:
                delta += 1
            elif (new_mask & other) == other and new_mask != other:
                delta += 1
        
        if delta > 0:
            used.remove(old_mask)
            used.add(new_mask)
            current_masks[i] = new_mask
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_masks = current_masks[:]
        else:
            if random.random() < math.exp(delta / T):
                used.remove(old_mask)
                used.add(new_mask)
                current_masks[i] = new_mask
                current_score += delta
            no_improve_count += 1
            
        T *= cooling

    result = []
    for mask in best_masks:
        s = [i for i in range(n) if mask & (1 << i)]
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
