import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for safety
    
    # Initial prescribed elements
    initial = [1, 2, 5, 11]
    
    # Use a bitset for sums for faster conflict checking
    # Max sum is 284 + 283 = 567. 2*284 = 568.
    sums_bitset = 0
    for i in range(len(initial)):
        for j in range(i, len(initial)):
            sums_bitset |= (1 << (initial[i] + initial[j]))
    
    best_set = initial[:]
    best_size = len(initial)
    
    def can_add(x, S, sums_mask):
        # Check if 2*x conflicts
        if (sums_mask >> (2 * x)) & 1:
            return False, 0
        mask = 1 << (2 * x)
        # Check if x + y conflicts for any y in S
        for y in S:
            s = x + y
            if (sums_mask >> s) & 1:
                return False, 0
            mask |= (1 << s)
        return True, mask

    def backtrack(S, sums_mask, candidates):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(S) > best_size:
            best_set = S[:]
            best_size = len(S)
            
        # Filter valid candidates from the provided order
        valid_candidates = []
        for x in candidates:
            ok, new_mask_contrib = can_add(x, S, sums_mask)
            if ok:
                valid_candidates.append((x, new_mask_contrib))
                
        if not valid_candidates:
            return
            
        for x, new_mask_contrib in valid_candidates:
            if time.time() > deadline:
                return
            
            new_S = S + [x]
            new_sums_mask = sums_mask | new_mask_contrib
            
            # Filter remaining candidates > x to maintain increasing order
            # This avoids permutations of the same set
            remaining = [c for c in candidates if c > x]
            
            backtrack(new_S, new_sums_mask, remaining)
            
            if time.time() > deadline:
                return

    # First pass: Deterministic increasing order
    candidates = list(range(12, 285))
    backtrack(initial, sums_bitset, candidates)
    
    # Random restarts to escape local optima
    # The bitset optimization allows for more restarts in the same time
    while time.time() < deadline - 2:
        random.shuffle(candidates)
        backtrack(initial, sums_bitset, candidates)
        
    return best_set
