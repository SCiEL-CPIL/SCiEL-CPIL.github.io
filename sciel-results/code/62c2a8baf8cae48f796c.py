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

def _is_comparable(a, b):
    """Check if two masks are comparable (one is subset of other)."""
    return (a & b) == a or (a & b) == b

def _greedy_construct_chain_aware():
    """Greedy construction that favors chain extensions."""
    chosen = []
    chosen_set = set()
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << N)
    for mask in range(1, 1 << N):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    for step in range(M):
        best_mask = -1
        best_score = -1
        
        for mask in range(1 << N):
            if mask in chosen_set:
                continue
            
            # Score: count how many existing masks this is comparable with
            # Bonus for being in a "chain position" (subset of some, superset of others)
            score = 0
            is_subset_of = 0
            is_superset_of = 0
            
            for c in chosen:
                if (mask & c) == mask:
                    score += 1
                    is_subset_of += 1
                elif (mask & c) == c:
                    score += 1
                    is_superset_of += 1
            
            # Bonus: prefer masks that are in the "middle" of potential chains
            # (comparable with both smaller and larger existing masks)
            if is_subset_of > 0 and is_superset_of > 0:
                score += 3  # Strong bonus for chain extension
            elif is_subset_of > 0:
                score += 1
            elif is_superset_of > 0:
                score += 1
            
            # Slight preference for masks with "good" popcount distribution
            # (not too small, not too large) to allow future extensions
            pc = popcount[mask]
            if 2 <= pc <= 9:
                score += 0.5
            
            if score > best_score or (score == best_score and best_mask == -1):
                best_score = score
                best_mask = mask
        
        chosen.append(best_mask)
        chosen_set.add(best_mask)
    
    return chosen

def _full_scan_local_search(masks, deadline=None):
    """Full-scan local search: for each position, try all 4096 masks to find best replacement."""
    current_score = _count_pairs(masks)
    all_masks = list(range(1 << N))
    mask_set = set(masks)
    improved = True
    
    # Precompute comparability count for each mask against current set
    # comp_count[m] = number of masks in current set comparable with m
    comp_count = [0] * (1 << N)
    for m in all_masks:
        cnt = 0
        for c in masks:
            if _is_comparable(m, c):
                cnt += 1
        comp_count[m] = cnt
    
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for idx in range(M):
            if deadline and time.time() > deadline:
                break
            old_mask = masks[idx]
            old_comp = comp_count[old_mask]
            
            best_delta = 0
            best_new_mask = -1
            
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                if new_mask in mask_set:
                    continue
                
                # Delta = (new comparability count) - (old comparability count)
                # But we need to be careful: comp_count includes the mask itself if it were in the set
                # Since old_mask is in the set, comp_count[old_mask] includes comparability with all others
                # For new_mask, comp_count[new_mask] is computed against the current set (which includes old_mask)
                # So delta = comp_count[new_mask] - comp_count[old_mask] + adjustment
                # Actually, comp_count[new_mask] counts comparability with all M masks including old_mask
                # But we're replacing old_mask with new_mask, so we need comparability with the other M-1 masks
                # comp_count[new_mask] = comparability with all M masks (including old_mask)
                # We need comparability with the M-1 masks excluding old_mask
                # If new_mask is comparable with old_mask, then comp_count[new_mask] - 1 is what we want
                # If not, comp_count[new_mask] is what we want
                # Similarly for old_mask: comp_count[old_mask] includes comparability with itself (which is always true)
                # So comp_count[old_mask] - 1 is comparability with other M-1 masks
                
                new_comp_with_others = comp_count[new_mask]
                if _is_comparable(new_mask, old_mask):
                    new_comp_with_others -= 1  # Remove comparability with old_mask itself
                
                old_comp_with_others = comp_count[old_mask] - 1  # Remove self-comparability
                
                delta = new_comp_with_others - old_comp_with_others
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask >= 0:
                # Update comp_count for all masks
                for m in all_masks:
                    old_was_comp = 1 if _is_comparable(m, old_mask) else 0
                    new_is_comp = 1 if _is_comparable(m, best_new_mask) else 0
                    comp_count[m] += new_is_comp - old_was_comp
                
                mask_set.remove(old_mask)
                mask_set.add(best_new_mask)
                masks[idx] = best_new_mask
                current_score += best_delta
                improved = True
    
    return masks

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
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.5
            cooling = 0.9995
            no_improve_count = 0
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        # Generate candidate: 50% structured (based on existing masks), 50% random
        if random.random() < 0.5:
            # Structured: take a random existing mask and modify it
            base = random.choice(masks)
            if base == old_mask:
                base = random.choice(masks)
            flips = random.randint(1, 3)
            new_mask = base
            for _ in range(flips):
                bit = random.randint(0, N - 1)
                new_mask ^= (1 << bit)
        else:
            new_mask = random.choice(all_masks)
        
        if new_mask == old_mask:
            continue
        if new_mask in mask_set:
            continue
        
        # Compute delta
        delta = 0
        for j in range(M):
            if j == idx:
                continue
            c = masks[j]
            old_comp = 1 if _is_comparable(old_mask, c) else 0
            new_comp = 1 if _is_comparable(new_mask, c) else 0
            delta += new_comp - old_comp
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            mask_set.remove(old_mask)
            mask_set.add(new_mask)
            masks[idx] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling
        
        # If stuck for too long, restart with higher temperature
        if no_improve_count > max_no_improve:
            T = 2.0
            no_improve_count = 0
    
    return best_masks

def _greedy_construct_size_aware():
    """Alternative greedy: build by size layers to create good chain structure."""
    chosen = []
    chosen_set = set()
    
    # Group masks by popcount
    by_size = [[] for _ in range(N + 1)]
    for mask in range(1 << N):
        pc = bin(mask).count('1')
        by_size[pc].append(mask)
    
    # Build from small to large, preferring masks that extend chains
    for size in range(0, N + 1):
        if len(chosen) >= M:
            break
        
        # For each mask of this size, score it
        candidates = []
        for mask in by_size[size]:
            if mask in chosen_set:
                continue
            
            score = 0
            for c in chosen:
                if _is_comparable(mask, c):
                    score += 1
            
            candidates.append((score, mask))
        
        # Sort by score descending, take as many as needed
        candidates.sort(key=lambda x: -x[0])
        for score, mask in candidates:
            if len(chosen) >= M:
                break
            chosen.append(mask)
            chosen_set.add(mask)
    
    # If we don't have enough, fill with remaining
    if len(chosen) < M:
        for mask in range(1 << N):
            if len(chosen) >= M:
                break
            if mask not in chosen_set:
                chosen.append(mask)
                chosen_set.add(mask)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_overall = None
    best_overall_score = -1
    
    # Try multiple initialization strategies
    strategies = [
        _greedy_construct_chain_aware,
        _greedy_construct_size_aware,
    ]
    
    for strategy_idx, strategy in enumerate(strategies):
        if time.time() >= deadline - 5:
            break
        
        random.seed(42 + strategy_idx * 137)
        
        # Initialize
        masks = strategy()
        
        # Add random perturbation for diversity
        for _ in range(3):
            idx = random.randint(0, M - 1)
            new_mask = random.randint(0, FULL_MASK)
            if new_mask not in masks:
                masks[idx] = new_mask
        
        # Full-scan local search
        ls_deadline = deadline - 10
        if time.time() < ls_deadline:
            masks = _full_scan_local_search(masks, deadline=ls_deadline)
        
        # Simulated annealing
        sa_deadline = deadline - 5
        if time.time() < sa_deadline:
            masks = _simulated_annealing(masks, deadline=sa_deadline)
        
        # Final local search polish
        if time.time() < deadline - 2:
            masks = _full_scan_local_search(masks, deadline=deadline - 1)
        
        score = _count_pairs(masks)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = list(masks)
    
    # If no good result, fall back to simple greedy
    if best_overall is None:
        best_overall = _greedy_construct_chain_aware()
        best_overall_score = _count_pairs(best_overall)
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_overall:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
