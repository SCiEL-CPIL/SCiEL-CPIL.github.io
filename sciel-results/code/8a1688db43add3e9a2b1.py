import random
import time

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute subset relationships? 
    # Or just use bitwise operations.
    # (a & b) == a means a is subset of b.
    
    def count_comparables(family_masks):
        count = 0
        for i in range(len(family_masks)):
            for j in range(i+1, len(family_masks)):
                a, b = family_masks[i], family_masks[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    def greedy_build(initial_mask):
        selected = [initial_mask]
        selected_set = {initial_mask}
        # scores[mask] = number of comparable sets in 'selected'
        scores = [0] * total_masks
        
        # Initialize scores for the first mask
        for u in range(total_masks):
            if u == initial_mask:
                continue
            if (u & initial_mask) == u or (u & initial_mask) == initial_mask:
                scores[u] = 1
        
        for _ in range(m - 1):
            # Find best candidate
            best_mask = -1
            best_score = -1
            # We can optimize this by only checking unselected masks
            # But total_masks is 4096, which is small.
            for u in range(total_masks):
                if u in selected_set:
                    continue
                if scores[u] > best_score:
                    best_score = scores[u]
                    best_mask = u
            
            if best_mask == -1:
                break
                
            selected.append(best_mask)
            selected_set.add(best_mask)
            
            # Update scores for all other unselected masks
            for u in range(total_masks):
                if u in selected_set:
                    continue
                if (u & best_mask) == u or (u & best_mask) == best_mask:
                    scores[u] += 1
        
        return selected

    best_family = []
    best_score = -1
    
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # Try several random initializations
    # Also try specific ones like 0, 4095, and some mid-sized sets
    initial_candidates = [0, (1<<n)-1]
    for _ in range(20):
        initial_candidates.append(random.randint(0, total_masks - 1))
        
    for init in initial_candidates:
        if time.time() > deadline:
            break
            
        family = greedy_build(init)
        if len(family) < m:
            continue
            
        # Calculate score
        score = count_comparables(family)
        
        if score > best_score:
            best_score = score
            best_family = family
        
        # Early stopping if we reach theoretical max?
        # Max pairs for 40 sets is C(40, 2) = 780.
        # But in a 12-cube, we can't have a clique of size 40.
        # The width of B12 is C(12, 6) = 924. The height is 13.
        # A chain has 13 elements.
        # We can't form a single chain of 40.
        # So 780 is not reachable.
        
    if not best_family:
        # Fallback: just return first 40 masks
        best_family = list(range(m))
        
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)
        
    return result
