import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute subset relationships: subset[a][b] = True if a is strict subset of b
    # We'll use a 2D array or compute on the fly with bit operations
    
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def count_comparable(family):
        """Count comparable pairs in family (list of masks)."""
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            for j in range(i + 1, k):
                b = family[j]
                if is_strict_subset(a, b) or is_strict_subset(b, a):
                    count += 1
        return count
    
    def greedy_construct(seed):
        """Greedy construction: add sets one by one, choosing the one that maximizes new comparable pairs."""
        random.seed(seed)
        family = []
        remaining = set(range(total_masks))
        
        # Start with a random small set
        first = random.choice([m for m in range(total_masks) if bin(m).count('1') <= 2])
        family.append(first)
        remaining.discard(first)
        
        for _ in range(m - 1):
            if time.time() > deadline:
                break
            best_mask = None
            best_gain = -1
            # Sample a subset of candidates to speed up
            candidates = list(remaining)
            if len(candidates) > 200:
                candidates = random.sample(candidates, 200)
            
            for cand in candidates:
                gain = 0
                for f in family:
                    if is_strict_subset(cand, f) or is_strict_subset(f, cand):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = cand
            
            if best_mask is not None:
                family.append(best_mask)
                remaining.discard(best_mask)
        
        return family
    
    def local_search(family, max_iters=1000):
        """Try swapping elements to improve the score."""
        if time.time() > deadline:
            return family, count_comparable(family)
        
        family = list(family)
        current_score = count_comparable(family)
        family_set = set(family)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            i = random.randint(0, len(family) - 1)
            old_mask = family[i]
            
            # Find a replacement that improves the score
            # Try a random mask not in family
            for attempt in range(50):
                if time.time() > deadline:
                    break
                new_mask = random.randint(0, total_masks - 1)
                if new_mask in family_set:
                    continue
                
                # Compute score change
                old_gain = 0
                new_gain = 0
                for j, f in enumerate(family):
                    if j == i:
                        continue
                    if is_strict_subset(old_mask, f) or is_strict_subset(f, old_mask):
                        old_gain += 1
                    if is_strict_subset(new_mask, f) or is_strict_subset(f, new_mask):
                        new_gain += 1
                
                delta = new_gain - old_gain
                if delta > 0:
                    family[i] = new_mask
                    family_set.discard(old_mask)
                    family_set.add(new_mask)
                    current_score += delta
                    break
        
        return family, current_score
    
    best_family = None
    best_score = -1
    
    # Try multiple random restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        family = greedy_construct(restart * 12345 + 678)
        if len(family) < m:
            # Fill remaining with random sets
            family_set = set(family)
            while len(family) < m:
                cand = random.randint(0, total_masks - 1)
                if cand not in family_set:
                    family.append(cand)
                    family_set.add(cand)
        
        family, score = local_search(family, max_iters=500)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    if best_family is None or len(best_family) < m:
        # Fallback: simple deterministic construction
        best_family = []
        # Add all singletons and pairs first (they're subsets of many things)
        for i in range(n):
            best_family.append(1 << i)
        for i in range(n):
            for j in range(i + 1, n):
                best_family.append((1 << i) | (1 << j))
        # Add larger sets
        for mask in range(1, total_masks):
            if len(best_family) >= m:
                break
            if mask not in set(best_family):
                best_family.append(mask)
        best_family = best_family[:m]
    
    # Ensure exactly m distinct sets
    best_family = list(dict.fromkeys(best_family))[:m]
    while len(best_family) < m:
        # Add a random set not already in the family
        for _ in range(1000):
            cand = random.randint(0, total_masks - 1)
            if cand not in best_family:
                best_family.append(cand)
                break
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
