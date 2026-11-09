import time
import random
import math

N = 12
M = 40
FULL_MASK = (1 << N) - 1
NUM_MASKS = 1 << N

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

def _is_comparable(a, b):
    """Check if two masks are comparable (one is subset of other)."""
    return (a & b) == a or (a & b) == b

def _fast_local_search(masks, deadline=None):
    """
    Local search using precomputed comparability counts.
    Maintains comp_count[m] = number of masks in current family comparable to m.
    """
    current_score = _count_pairs(masks)
    mask_set = set(masks)
    
    # Precompute comp_count for all 4096 masks
    comp_count = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        cnt = 0
        for c in masks:
            if _is_comparable(m, c):
                cnt += 1
        comp_count[m] = cnt
        
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Find the best swap: remove i, add j
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(M):
            a = masks[i]
            # Cost of removing a: (comp_count[a] - 1) is the number of pairs involving a
            # Actually, we want to maximize the gain.
            # Gain = (pairs gained by adding j) - (pairs lost by removing a)
            # Pairs lost by removing a = comp_count[a] - 1
            # Pairs gained by adding j = comp_count[j] - 1 - (1 if j comparable to a else 0)
            # Wait, comp_count[j] includes comparability with a if j is comparable to a.
            # If we remove a first, the family becomes F \ {a}.
            # Then we add j.
            # New pairs involving j = (number of x in F \ {a} such that x ~ j)
            # = comp_count[j] - (1 if j ~ a else 0)
            
            lost = comp_count[a] - 1
            
            for j in range(NUM_MASKS):
                if j in mask_set:
                    continue
                
                # Check if j is comparable to a
                is_comp_ja = 1 if _is_comparable(j, a) else 0
                gained = comp_count[j] - is_comp_ja
                
                delta = gained - lost
                
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i != -1:
            # Perform the swap
            a = masks[best_i]
            b = best_j
            
            # Update masks list
            masks[best_i] = b
            mask_set.remove(a)
            mask_set.add(b)
            
            # Update comp_count
            # For every mask m in 0..4095:
            # old contribution of a: 1 if m ~ a
            # new contribution of b: 1 if m ~ b
            for m in range(NUM_MASKS):
                old_comp = 1 if _is_comparable(m, a) else 0
                new_comp = 1 if _is_comparable(m, b) else 0
                comp_count[m] += new_comp - old_comp
                
            current_score += best_delta
            improved = True
            
    return masks, current_score

def _simulated_annealing_fast(masks, deadline=None):
    """
    Simulated annealing using the fast delta calculation.
    """
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    mask_set = set(masks)
    
    # Precompute comp_count
    comp_count = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        cnt = 0
        for c in masks:
            if _is_comparable(m, c):
                cnt += 1
        comp_count[m] = cnt
        
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
            
        idx = random.randint(0, M - 1)
        a = masks[idx]
        
        # Pick a random new mask
        b = random.randint(0, NUM_MASKS - 1)
        if b == a or b in mask_set:
            continue
            
        # Calculate delta
        is_comp_ab = 1 if _is_comparable(a, b) else 0
        lost = comp_count[a] - 1
        gained = comp_count[b] - is_comp_ab
        delta = gained - lost
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            # Accept
            masks[idx] = b
            mask_set.remove(a)
            mask_set.add(b)
            
            # Update comp_count
            for m in range(NUM_MASKS):
                old_comp = 1 if _is_comparable(m, a) else 0
                new_comp = 1 if _is_comparable(m, b) else 0
                comp_count[m] += new_comp - old_comp
                
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
                
        T *= cooling
        
    return best_masks, best_score

def _greedy_init():
    """
    Greedy initialization: start with empty set, then add sets that maximize comparability.
    """
    masks = []
    mask_set = set()
    
    # Precompute popcount
    popcount = [0] * NUM_MASKS
    for m in range(1, NUM_MASKS):
        popcount[m] = popcount[m >> 1] + (m & 1)
        
    # Start with empty set
    masks.append(0)
    mask_set.add(0)
    
    comp_count = [0] * NUM_MASKS
    comp_count[0] = 1 # 0 is comparable with 0 (itself)
    # Actually, for greedy, we just count comparability with existing.
    # Let's just do a simple greedy loop.
    
    for _ in range(M - 1):
        best_mask = -1
        best_score = -1
        
        for m in range(NUM_MASKS):
            if m in mask_set:
                continue
            
            score = 0
            for c in masks:
                if _is_comparable(m, c):
                    score += 1
            
            # Tie-breaker: prefer middle sizes
            pc = popcount[m]
            if 2 <= pc <= 9:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_mask = m
        
        masks.append(best_mask)
        mask_set.add(best_mask)
        
    return masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_overall = None
    best_overall_score = -1
    
    # Multiple restarts with different seeds/initializations
    strategies = [
        _greedy_init,
        lambda: [random.randint(0, FULL_MASK) for _ in range(M)],
        lambda: [i * (FULL_MASK // M) for i in range(M)], # Spread out
    ]
    
    for strategy_idx, strategy in enumerate(strategies):
        if time.time() >= deadline - 10:
            break
            
        random.seed(42 + strategy_idx * 137)
        
        # Ensure distinct masks for random init
        if strategy_idx > 0:
            masks = strategy()
            # Remove duplicates
            unique_masks = list(set(masks))
            while len(unique_masks) < M:
                new_m = random.randint(0, FULL_MASK)
                if new_m not in unique_masks:
                    unique_masks.append(new_m)
            masks = unique_masks[:M]
        else:
            masks = strategy()
        
        # 1. Fast Local Search
        ls_deadline = deadline - 20
        if time.time() < ls_deadline:
            masks, score = _fast_local_search(masks, deadline=ls_deadline)
            if score > best_overall_score:
                best_overall_score = score
                best_overall = list(masks)
                
        # 2. Simulated Annealing
        sa_deadline = deadline - 5
        if time.time() < sa_deadline:
            masks, score = _simulated_annealing_fast(masks, deadline=sa_deadline)
            if score > best_overall_score:
                best_overall_score = score
                best_overall = list(masks)
                
        # 3. Final Polish with Local Search
        if time.time() < deadline - 1:
            masks, score = _fast_local_search(masks, deadline=deadline - 0.5)
            if score > best_overall_score:
                best_overall_score = score
                best_overall = list(masks)
                
    if best_overall is None:
        best_overall = _greedy_init()
        
    # Convert masks to lists of vertices
    result = []
    for mask in best_overall:
        result.append([b for b in range(N) if (mask >> b) & 1])
        
    return result
