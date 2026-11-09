import time
import random
import math

N = 12
M = 40
FULL_MASK = (1 << N) - 1

def _count_pairs(masks):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _incremental_score(masks, idx, new_mask):
    """Compute score change when replacing masks[idx] with new_mask."""
    old_mask = masks[idx]
    delta = 0
    for j in range(M):
        if j == idx:
            continue
        c = masks[j]
        old_comp = 1 if ((old_mask & c) == old_mask or (old_mask & c) == c) else 0
        new_comp = 1 if ((new_mask & c) == new_mask or (new_mask & c) == c) else 0
        delta += new_comp - old_comp
    return delta

def _greedy_construct_v2():
    """Improved greedy: prefer smaller sets, use randomization for diversity."""
    chosen = []
    chosen_set = set()
    
    for _ in range(M):
        best_mask = -1
        best_score = -999
        # Try multiple candidates and pick best
        candidates = list(range(1 << N))
        random.shuffle(candidates)
        
        for mask in candidates:
            if mask in chosen_set:
                continue
            # Score: number of comparable pairs with existing + bonus for small sets
            score = 0
            for c in chosen:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            # Bonus for smaller sets (they're more likely to be subsets)
            popcount = bin(mask).count('1')
            score += (12 - popcount) * 0.1
            if score > best_score:
                best_score = score
                best_mask = mask
        chosen.append(best_mask)
        chosen_set.add(best_mask)
        
    return chosen

def _full_scan_local_search(masks, deadline=None):
    """Full-scan local search with precomputed comparability table."""
    current_score = _count_pairs(masks)
    all_masks = list(range(1 << N))
    mask_set = set(masks)
    improved = True
    iterations = 0
    
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for idx in range(M):
            if deadline and time.time() > deadline:
                break
            old_mask = masks[idx]
            best_delta = 0
            best_new_mask = -1
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                if new_mask in mask_set:
                    continue
                delta = _incremental_score(masks, idx, new_mask)
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            if best_new_mask >= 0:
                mask_set.remove(old_mask)
                mask_set.add(best_new_mask)
                masks[idx] = best_new_mask
                current_score += best_delta
                improved = True
                iterations += 1
    return masks, current_score

def _simulated_annealing(masks, deadline=None):
    """Simulated annealing with structured neighborhood moves."""
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    all_masks = list(range(1 << N))
    mask_set = set(masks)
    
    T = 3.0
    T_min = 0.005
    cooling = 0.9992
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.5
            cooling = 0.9995
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        if random.random() < 0.3:
            flips = random.randint(1, 2)
            new_mask = old_mask
            for _ in range(flips):
                bit = random.randint(0, N - 1)
                new_mask ^= (1 << bit)
        else:
            new_mask = random.choice(all_masks)
        
        if new_mask == old_mask:
            continue
        if new_mask in mask_set:
            continue
        
        delta = _incremental_score(masks, idx, new_mask)
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            mask_set.remove(old_mask)
            mask_set.add(new_mask)
            masks[idx] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
        
        T *= cooling
    
    return best_masks, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    random.seed(42)
    
    # Multi-start approach
    best_masks = []
    best_score = -1
    
    for restart in range(3):
        if time.time() > deadline - 15:
            break
            
        if restart == 0:
            masks = _greedy_construct_v2()
        else:
            # Perturb the best solution found so far
            if best_masks:
                masks = list(best_masks)
                # Randomly swap 3-5 elements
                for _ in range(random.randint(3, 5)):
                    i, j = random.sample(range(M), 2)
                    masks[i], masks[j] = masks[j], masks[i]
            else:
                masks = _greedy_construct_v2()
        
        # Local search
        ls_deadline = min(deadline - 10, time.time() + 10)
        masks, score = _full_scan_local_search(masks, deadline=ls_deadline)
        
        if score > best_score:
            best_score = score
            best_masks = list(masks)
        
        # Simulated annealing
        sa_deadline = min(deadline - 8, time.time() + 12)
        if time.time() < sa_deadline:
            masks, sa_score = _simulated_annealing(masks, deadline=sa_deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_masks = list(masks)
    
    # Final polish
    if time.time() < deadline - 2 and best_masks:
        masks = list(best_masks)
        masks, final_score = _full_scan_local_search(masks, deadline=deadline - 1)
        if final_score > best_score:
            best_score = final_score
            best_masks = list(masks)
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_masks:
        result.append([b for b in range(N) if (mask >> b) & 1])
        
    return result
