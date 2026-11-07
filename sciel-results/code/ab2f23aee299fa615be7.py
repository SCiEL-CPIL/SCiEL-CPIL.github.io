import time
import random

def _count_comparable(masks):
    """Count unordered pairs where one mask is a strict subset of another."""
    n = len(masks)
    count = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            if mi != mj and (mi & mj) == mi:
                count += 1
            elif mi != mj and (mi & mj) == mj:
                count += 1
    return count

def _delta_swap(masks, idx_out, mask_in):
    """Compute change in comparable count when replacing masks[idx_out] with mask_in."""
    old = masks[idx_out]
    if old == mask_in:
        return 0
    delta = 0
    for j in range(len(masks)):
        if j == idx_out:
            continue
        mj = masks[j]
        # Remove contribution of old with mj
        if old != mj:
            if (old & mj) == old:
                delta -= 1
            elif (old & mj) == mj:
                delta -= 1
        # Add contribution of mask_in with mj
        if mask_in != mj:
            if (mask_in & mj) == mask_in:
                delta += 1
            elif (mask_in & mj) == mj:
                delta += 1
    return delta

def _local_search(masks, all_masks, deadline):
    """Improve masks via single-swap local search."""
    n = len(masks)
    in_set = set(masks)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            best_gain = 0
            best_mask = -1
            for m in all_masks:
                if m in in_set:
                    continue
                gain = _delta_swap(masks, i, m)
                if gain > best_gain:
                    best_gain = gain
                    best_mask = m
            if best_mask >= 0 and best_gain > 0:
                in_set.discard(masks[i])
                masks[i] = best_mask
                in_set.add(best_mask)
                improved = True
        if time.time() > deadline:
            break
    return masks

def _initial_solution():
    """Create a strong initial solution: a chain plus additional sets."""
    # Full chain: {0}, {0,1}, {0,1,2}, ..., {0,...,11}
    # Represented as bitmasks: 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    # That's 12 sets (sizes 1-12). Add empty set for 13.
    chain = [0]
    for i in range(1, 13):
        chain.append((1 << i) - 1)
    # chain has 13 sets. Need 27 more.
    
    # Add sets that extend the chain structure
    # For each size, add sets that are "close" to the chain
    # Strategy: for each size k, the chain set is {0,...,k-1}. 
    # Add other k-sets that contain {0,...,k-2} (i.e., differ by one element)
    
    masks = list(chain)
    in_set = set(masks)
    
    # For sizes 1-11, add sets that are "neighbors" in the lattice
    # A k-set S is comparable to the chain set C_k={0,...,k-1} if S subset C_k or C_k subset S
    # To maximize comparability, add sets that are subsets of or supersets of chain sets
    
    # Add 1-element subsets of {0,...,10} that aren't {0}
    # Actually, let's be more systematic: add sets that maximize comparability with existing
    
    # For each size from 1 to 11, add sets of that size that contain the chain set of size-1
    # i.e., for size k, add sets containing {0,...,k-2}
    for k in range(1, 12):
        if len(masks) >= 40:
            break
        base = (1 << (k-1)) - 1 if k > 1 else 0  # {0,...,k-2}
        # All k-sets containing base: choose 1 more element from {k-1, ..., 11}
        for extra in range(k-1, 12):
            if len(masks) >= 40:
                break
            m = base | (1 << extra)
            if m not in in_set:
                masks.append(m)
                in_set.add(m)
    
    # If we still need more, add random sets
    while len(masks) < 40:
        m = random.randint(0, 4095)
        if m not in in_set:
            masks.append(m)
            in_set.add(m)
    
    return masks[:40]

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Precompute all masks
    all_masks = list(range(4096))
    
    best_masks = None
    best_score = -1
    
    # Try multiple initializations
    random.seed(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        masks = _initial_solution()
        score = _count_comparable(masks)
        
        if score > best_score:
            best_score = score
            best_masks = list(masks)
        
        # Local search
        masks = _local_search(masks, all_masks, deadline - 5.0)
        score = _count_comparable(masks)
        
        if score > best_score:
            best_score = score
            best_masks = list(masks)
    
    # Convert masks to lists of integers
    result = []
    for m in best_masks:
        result.append([b for b in range(12) if (m >> b) & 1])
    
    return result
