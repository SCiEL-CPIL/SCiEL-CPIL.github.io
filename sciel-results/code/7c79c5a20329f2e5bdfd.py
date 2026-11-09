import time
import itertools

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 12
    M = 40
    TOTAL = 1 << N
    
    # Precompute subset relations? 
    # is_subset[a][b] is True if a is subset of b
    # This is a 4096x4096 boolean matrix. 16M entries. 
    # Might be too slow to build or memory heavy in Python lists.
    # Bit operations are fast enough.
    
    def is_subset(a, b):
        return (a & b) == a
        
    def is_comparable(a, b):
        if a == b: return False
        return (a & b) == a or (a & b) == b

    # 1. Greedy Initialization
    family = []
    used = [False] * TOTAL
    
    for _ in range(M):
        best_mask = -1
        best_score = -1
        for m in range(TOTAL):
            if used[m]:
                continue
            # Calculate score with existing family
            s = 0
            for f in family:
                if is_comparable(m, f):
                    s += 1
            if s > best_score:
                best_score = s
                best_mask = m
        family.append(best_mask)
        used[best_mask] = True
        
    if time.time() > deadline:
        return [[b for b in range(N) if (mask >> b) & 1] for mask in family]

    # 2. Local Search
    # We want to maximize sum of comparable pairs.
    # Change in score when replacing f[i] with m:
    # Delta = (comparables of m with others) - (comparables of f[i] with others)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(M):
            old_mask = family[i]
            others = [f for j, f in enumerate(family) if j != i]
            
            # Calculate base score of old_mask with others
            old_score = 0
            for f in others:
                if is_comparable(old_mask, f):
                    old_score += 1
            
            best_gain = -1
            best_new_mask = -1
            
            # Try all other masks
            for m in range(TOTAL):
                if used[m]:
                    continue
                # Calculate new score
                new_score = 0
                for f in others:
                    if is_comparable(m, f):
                        new_score += 1
                
                gain = new_score - old_score
                if gain > best_gain:
                    best_gain = gain
                    best_new_mask = m
            
            if best_gain > 0 and best_new_mask != -1:
                # Perform swap
                used[old_mask] = False
                used[best_new_mask] = True
                family[i] = best_new_mask
                improved = True
                # Continue to next i or restart? 
                # Standard hill climbing: continue to next i with updated family.
                # But 'others' changed. 
                # For simplicity and speed, we can just continue.
                # Or break and restart loop for safety.
                # Let's continue, it's usually fine.
                
    # Convert masks to lists of ints
    result = []
    for mask in family:
        result.append([b for b in range(N) if (mask >> b) & 1])
        
    return result
