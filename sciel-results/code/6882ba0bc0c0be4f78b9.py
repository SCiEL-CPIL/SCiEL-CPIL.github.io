import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 257))
    max_k = 60
    
    # B is a list of ints representing bitsets for each cardinality
    # B[k] has bit s set if sum s is achievable with cardinality k
    
    def try_add(x, S, B):
        # S is current list of added elements
        # B is current bitsets
        # Returns (True, new_B) if x can be added, else (False, None)
        
        max_current_k = len(S)
        
        # We need to check if the new set of bitsets is disjoint
        # new_B[k] = B[k] | (B[k-1] << x)
        
        # We can compute the new bitsets and check intersections.
        # To save time, we can check intersections incrementally?
        # No, the new bitsets are global.
        
        # Let's compute the new bitsets for k=1..max_current_k+1
        # Note: B[0] is always 1.
        
        new_B = [0] * (max_k + 2)
        new_B[0] = 1
        
        for k in range(1, max_current_k + 2):
            part1 = B[k] if k <= max_current_k else 0
            part2 = (B[k-1] << x) if k-1 <= max_current_k else 0
            new_B[k] = part1 | part2
            
        # Check disjointness
        # Only check pairs where both are non-zero
        active = [k for k in range(1, max_current_k + 2) if new_B[k] != 0]
        
        for i_idx in range(len(active)):
            i = active[i_idx]
            bi = new_B[i]
            for j_idx in range(i_idx + 1, len(active)):
                j = active[j_idx]
                if bi & new_B[j]:
                    return False, None
            
        return True, new_B

    # Run multiple random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        S = []
        B = [0] * (max_k + 2)
        B[0] = 1
        
        for x in candidates:
            # Optional: if S is already large, skip small x? No, x value matters for sum.
            # Just try to add.
            valid, new_B = try_add(x, S, B)
            if valid:
                S.append(x)
                B = new_B
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    if best_size == 60:
                        return best_set

    return best_set
