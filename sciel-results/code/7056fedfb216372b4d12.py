import time
import itertools

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 20s margin
    
    # Primes p = 3 mod 4
    candidates = []
    for p in range(31, 84):
        if p % 4 == 3 and is_prime(p):
            candidates.append(p)
    
    # Sort descending to find valid ones quickly? 
    # Actually we want smallest n. So sort ascending.
    # But we know 67 works. If 31 works, it's better.
    # So check 31, 43, 47, 59, 67.
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Build Paley tournament
        squares = {x*x % p for x in range(1, p)}
        
        # D[v] is bitmask of vertices dominated by v
        D = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i != j:
                    if (j - i) % p in squares:
                        mask |= (1 << j)
            D[i] = mask
            
        full_mask = (1 << p) - 1
        
        # Check if any 4-set is a dominating set
        valid = True
        # Optimization: If a 4-set is a dominating set, 
        # we can stop checking this p.
        
        # Iterate 4-sets
        # For small p, itertools.combinations is fine.
        for S in itertools.combinations(range(p), 4):
            mask_S = 0
            for v in S:
                mask_S |= (1 << v)
                
            mask_union = 0
            for v in S:
                mask_union |= D[v]
                
            mask_not_S = full_mask ^ mask_S
            
            # Check if S dominates all v not in S
            if (mask_union & mask_not_S) == mask_not_S:
                valid = False
                break
        
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i+1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to seed if no smaller found or time out
    p = 83
    squares = {x*x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
