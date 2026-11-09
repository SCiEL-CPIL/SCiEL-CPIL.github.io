"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 12
M = 40
FULL_MASK = (1 << N) - 1

def mask_to_list(mask):
    return [b for b in range(N) if (mask >> b) & 1]

def list_to_mask(lst):
    m = 0
    for v in lst:
        m |= (1 << v)
    return m

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

def _greedy_construct():
    """Greedy construction of 40 masks."""
    chosen = []
    all_masks = list(range(1 << N))
    
    for _ in range(M):
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in chosen:
                continue
            score = 0
            for c in chosen:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        chosen.append(best_mask)
        
    return chosen

def _local_search(masks, deadline):
    """Local search with incremental score updates and simulated annealing."""
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    
    # Maintain comp_count for incremental updates
    comp_count = [0] * M
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            mi = masks[i]
            mj = masks[j]
            if (mi & mj) == mi and mi != mj:
                comp_count[i] += 1
            elif (mi & mj) == mj and mi != mj:
                comp_count[i] += 1
    
    # Verify
    assert current_score == sum(comp_count) // 2, f"Score mismatch: {current_score} vs {sum(comp_count)//2}"
    
    T = 10.0  # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Pick a random index to modify
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        # Generate candidates
        candidates = set()
        
        # 1. Random mask
        candidates.add(random.randint(0, FULL_MASK))
        
        # 2. Neighbors (flip 1 bit)
        for b in range(N):
            candidates.add(old_mask ^ (1 << b))
        
        # 3. Random subsets/supersets
        for _ in range(3):
            subset = 0
            for b in range(N):
                if (old_mask >> b) & 1 and random.random() < 0.5:
                    subset |= (1 << b)
            candidates.add(subset)
            
            superset = old_mask
            for b in range(N):
                if not ((old_mask >> b) & 1) and random.random() < 0.5:
                    superset |= (1 << b)
            candidates.add(superset)
        
        # Remove old mask from consideration of duplicates
        existing = set(masks[j] for j in range(M) if j != idx)
        
        best_delta = -1000
        best_new_mask = None
        
        for cand in candidates:
            if cand in existing:
                continue
            
            # Calculate delta using incremental approach
            loss = comp_count[idx]
            
            gain = 0
            for j in range(M):
                if idx == j:
                    continue
                mj = masks[j]
                if (cand & mj) == cand and cand != mj:
                    gain += 1
                elif (cand & mj) == mj and cand != mj:
                    gain += 1
            
            delta = gain - loss
            
            if delta > best_delta:
                best_delta = delta
                best_new_mask = cand
        
        # Accept based on simulated annealing
        if best_new_mask is not None:
            if best_delta > 0 or random.random() < math.exp(best_delta / T):
                # Update
                # 1. Remove old_mask from comp_count
                for j in range(M):
                    if idx == j:
                        continue
                    mj = masks[j]
                    if (old_mask & mj) == old_mask and old_mask != mj:
                        comp_count[j] -= 1
                    elif (old_mask & mj) == mj and old_mask != mj:
                        comp_count[j] -= 1
                
                # 2. Replace
                masks[idx] = best_new_mask
                current_score += best_delta
                
                # 3. Add best_new_mask to comp_count
                for j in range(M):
                    if idx == j:
                        continue
                    mj = masks[j]
                    if (best_new_mask & mj) == best_new_mask and best_new_mask != mj:
                        comp_count[j] += 1
                    elif (best_new_mask & mj) == mj and best_new_mask != mj:
                        comp_count[j] += 1
                
                # 4. Update comp_count[idx]
                comp_count[idx] = gain
                
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks)
    
    return best_masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_overall = None
    best_overall_score = -1
    
    # Multiple restarts with different seeds
    for restart in range(5):
        if time.time() >= deadline:
            break
        
        random.seed(42 + restart * 137)
        
        # Greedy initialization
        masks = _greedy_construct()
        
        # Add some random perturbation for diversity
        for _ in range(restart * 5):
            idx = random.randint(0, M - 1)
            new_mask = random.randint(0, FULL_MASK)
            if new_mask not in masks:
                masks[idx] = new_mask
        
        # Local search
        masks = _local_search(masks, deadline)
        
        score = _count_pairs(masks)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = list(masks)
        
        if best_overall_score >= 200:  # Good enough, stop early
            break
    
    if best_overall is None:
        best_overall = _greedy_construct()
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_overall:
        result.append(mask_to_list(mask))
    
    return result

# EVOLVE-BLOCK-END
