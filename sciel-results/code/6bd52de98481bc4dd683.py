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
        old_comp = 1 if ((old_mask & c) == old_mask or (old_mask & c) == c) else 0
        new_comp = 1 if ((new_mask & c) == new_mask or (new_mask & c) == c) else 0
        delta += new_comp - old_comp
    return delta

def _construct_chain(start_mask, length):
    """Construct a chain of given length starting from start_mask.
    Assumes start_mask is 0. Adds bits one by one."""
    chain = []
    current = start_mask
    for i in range(length):
        chain.append(current)
        # Add the next available bit
        for b in range(N):
            if not (current & (1 << b)):
                current |= (1 << b)
                break
    return chain

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

def _initial_chain_structure():
    """Initialize with the 13, 11, 11, 5 chain structure."""
    masks = []
    
    # Chain 1: Length 13 (0..12)
    c1 = _construct_chain(0, 13)
    masks.extend(c1)
    
    # Chain 2: Length 11 (1..11)
    # Start with {1}, add bits 2..11
    start2 = 1 << 1
    c2 = _construct_chain(start2, 11)
    masks.extend(c2)
    
    # Chain 3: Length 11 (1..11)
    # Start with {2}, add bits 1, 3..11? 
    # To be disjoint from C2, we should use different starting sets.
    # C2 uses sets containing 1. 
    # Let's start C3 with {2}.
    start3 = 1 << 2
    c3 = _construct_chain(start3, 11)
    masks.extend(c3)
    
    # Check for duplicates
    seen = set()
    unique_masks = []
    for m in masks:
        if m not in seen:
            seen.add(m)
            unique_masks.append(m)
    masks = unique_masks
    
    # Chain 4: Length 5 (2..6)
    # We need 5 more sets.
    # Start with {3, 4}?
    # Let's just pick random sets that are not in masks and form a chain?
    # Or just extend the greedy approach for the remaining 5.
    # Actually, let's just add 5 sets from a new chain starting at {0, 3}?
    # Wait, 0 is used in C1.
    # Let's start at {3}.
    start4 = 1 << 3
    c4 = _construct_chain(start4, 5)
    masks.extend(c4)
    
    # Ensure exactly 40 distinct masks
    seen = set()
    final_masks = []
    for m in masks:
        if m not in seen:
            seen.add(m)
            final_masks.append(m)
    
    # If we have less than 40, fill with greedy
    if len(final_masks) < M:
        # Use greedy to fill the rest
        remaining = M - len(final_masks)
        all_masks = list(range(1 << N))
        for _ in range(remaining):
            best_mask = -1
            best_score = -1
            for mask in all_masks:
                if mask in final_masks:
                    continue
                score = 0
                for c in final_masks:
                    if (mask & c) == mask or (mask & c) == c:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            if best_mask >= 0:
                final_masks.append(best_mask)
    
    return final_masks[:M]

def _full_scan_local_search(masks, deadline=None):
    """Full-scan local search: for each position, try all 4096 masks to find best replacement."""
    current_score = _count_pairs(masks)
    all_masks = list(range(1 << N))
    mask_set = set(masks)
    improved = True
    
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
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.5
            cooling = 0.9995
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        # Generate candidate: 70% random, 30% neighbor (flip 1-2 bits)
        if random.random() < 0.3:
            # Neighbor: flip 1 or 2 random bits
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
    
    return best_masks

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    # 1. Chain-based initialization
    masks = _initial_chain_structure()
    
    # 2. Full-scan local search (exhaustive per position)
    masks = _full_scan_local_search(masks, deadline=deadline - 10)
    
    # 3. Simulated annealing to escape local optima
    sa_deadline = deadline - 8
    if time.time() < sa_deadline:
        masks = _simulated_annealing(masks, deadline=sa_deadline)
    
    # 4. Final full-scan polish
    if time.time() < deadline - 2:
        masks = _full_scan_local_search(masks, deadline=deadline - 1)
    
    # Convert masks to lists of vertices
    result = []
    for mask in masks:
        result.append([b for b in range(N) if (mask >> b) & 1])
        
    return result
