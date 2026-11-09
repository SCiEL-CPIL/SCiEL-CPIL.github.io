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
    """Compute score change when replacing masks[idx] with new_mask.
    Returns the delta in comparable pair count."""
    old_mask = masks[idx]
    delta = 0
    for j in range(M):
        if j == idx:
            continue
        c = masks[j]
        # Old comparability
        old_comp = 1 if ((old_mask & c) == old_mask or (old_mask & c) == c) else 0
        # New comparability
        new_comp = 1 if ((new_mask & c) == new_mask or (new_mask & c) == c) else 0
        delta += new_comp - old_comp
    return delta

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
    """Local search with simulated annealing and incremental scoring."""
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    
    T = 10.0  # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Pick a random index to modify
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        # Pick a random new mask not in the set
        new_mask = random.randint(0, FULL_MASK)
        if new_mask in masks:
            continue
        
        delta = _incremental_score(masks, idx, new_mask)
        
        if delta > 0 or random.random() < math.exp(delta / T):
            masks[idx] = new_mask
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
        else:
            # Revert
            pass
        
        # Occasionally try swapping two elements
        if random.random() < 0.3:
            i = random.randint(0, M - 1)
            j = random.randint(0, M - 1)
            if i != j:
                old_i = masks[i]
                old_j = masks[j]
                masks[i], masks[j] = old_j, old_i
                # Compute delta for swap
                delta_swap = 0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    c = masks[k]
                    old_comp_i = 1 if ((old_i & c) == old_i or (old_i & c) == c) else 0
                    old_comp_j = 1 if ((old_j & c) == old_j or (old_j & c) == c) else 0
                    new_comp_i = 1 if ((old_j & c) == old_j or (old_j & c) == c) else 0
                    new_comp_j = 1 if ((old_i & c) == old_i or (old_i & c) == c) else 0
                    delta_swap += (new_comp_i + new_comp_j) - (old_comp_i + old_comp_j)
                
                # Also check i-j comparability change
                old_ij = 1 if ((old_i & old_j) == old_i or (old_i & old_j) == old_j) else 0
                new_ij = 1 if ((old_j & old_i) == old_j or (old_j & old_i) == old_i) else 0
                delta_swap += new_ij - old_ij
                
                if delta_swap > 0 or random.random() < math.exp(delta_swap / T):
                    current_score += delta_swap
                    if current_score > best_score:
                        best_score = current_score
                        best_masks = list(masks)
                else:
                    masks[i], masks[j] = old_i, old_j
    
    return best_masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
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
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
