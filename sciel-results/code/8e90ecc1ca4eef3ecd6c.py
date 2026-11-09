import time
import random
import math

N_ORIG = 12
M = 40
FULL_MASK = (1 << N_ORIG) - 1

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

def _greedy_init():
    """
    Greedily construct a set of M masks from B_12.
    At each step, add the mask that maximizes the number of comparable pairs 
    with the current set.
    """
    current_set = set()
    # We want to add M distinct masks
    for _ in range(M):
        best_mask = -1
        best_gain = -1
        # Sample a subset of all masks to speed up if needed, but 4096 is small
        # So we can check all
        for mask in range(1 << N_ORIG):
            if mask in current_set:
                continue
            
            gain = 0
            # Check comparability with all in current_set
            for s in current_set:
                if (mask & s) == mask or (mask & s) == s:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
            elif gain == best_gain:
                # Tie-breaking: random choice among best
                if random.random() < 0.5:
                    best_mask = mask
        
        current_set.add(best_mask)
        
    return list(current_set)

def _optimize_subset():
    """
    Use Simulated Annealing starting from a Greedy Initialization.
    Search space is B_12.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Initial solution from Greedy
    current_list = _greedy_init()
    current_set = set(current_list)
    
    # Calculate initial score
    current_score = _count_pairs(current_list)
    best_score = current_score
    best_mask_set = set(current_list)
    
    # SA Parameters
    T = 5.0
    min_T = 0.1
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 2000
    
    # Pre-compute all masks list for fast iteration if needed, 
    # but we use random sampling for candidates
    all_masks = list(range(1 << N_ORIG))
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with a perturbation
            # Perturb current best by swapping a few elements
            current_list = list(best_mask_set)
            current_set = set(current_list)
            for _ in range(5):
                i = random.randint(0, M - 1)
                old_mask = current_list[i]
                # Find a new mask
                while True:
                    new_mask = random.randint(0, FULL_MASK)
                    if new_mask not in current_set:
                        break
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_list[i] = new_mask
            current_score = _count_pairs(current_list)
            no_improve = 0
            T = 5.0
            continue
        
        # Pick index to replace
        i = random.randint(0, M - 1)
        old_mask = current_list[i]
        
        # Pick new mask not in current_set
        while True:
            new_mask = random.randint(0, FULL_MASK)
            if new_mask not in current_set:
                break
        
        # Calculate delta
        delta = 0
        for j in range(M):
            if j == i:
                continue
            m_j = current_list[j]
            
            old_comp = 1 if ((old_mask & m_j) == old_mask or (old_mask & m_j) == m_j) else 0
            new_comp = 1 if ((new_mask & m_j) == new_mask or (new_mask & m_j) == m_j) else 0
            
            delta += new_comp - old_comp
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / T):
            # Update state
            current_set.remove(old_mask)
            current_set.add(new_mask)
            current_list[i] = new_mask
            current_score += delta
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_mask_set = set(current_list)
        else:
            no_improve += 1
            
        T *= cooling
        if T < min_T:
            T = min_T

    return list(best_mask_set), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin
    
    # Run the optimizer
    masks, score = _optimize_subset()
        
    # Convert to list of lists of ints
    result = []
    for mask in masks:
        result.append([b for b in range(N_ORIG) if (mask >> b) & 1])
    
    return result
