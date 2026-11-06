import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_size = N + 1
    best_set = None
    
    # Bitset for sums 0..512
    # Sum of two numbers in [1, 256] is in [2, 512]
    
    def can_add(x, S, used_sums):
        # Check if x can be added to S
        # S is a list/set of ints
        # used_sums is an int bitset
        for s in S:
            if (used_sums >> (x + s)) & 1:
                return False
        if (used_sums >> (2 * x)) & 1:
            return False
        return True

    def update_sums(x, S, used_sums):
        # Add x to S, return new used_sums
        new_used = used_sums
        for s in S:
            new_used |= (1 << (x + s))
        new_used |= (1 << (2 * x))
        return new_used

    def is_maximal(S, used_sums):
        # Check if any x in [1, N] not in S can be added
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add(x, S, used_sums):
                return False
        return True

    def run_greedy():
        S = []
        used_sums = 0
        candidates = set(range(1, N + 1))
        
        while True:
            valid_candidates = []
            for x in candidates:
                if can_add(x, S, used_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Find x with max blocking power
            best_x = None
            max_blocks = -1
            best_ties = []
            
            for x in valid_candidates:
                # Calculate new sums if x is added
                new_sums = update_sums(x, S, used_sums)
                new_S = S + [x]
                
                # Count how many other valid candidates become invalid
                blocks = 0
                for y in valid_candidates:
                    if y == x:
                        continue
                    # Check if y can be added to new_S with new_sums
                    if not can_add(y, new_S, new_sums):
                        blocks += 1
                
                if blocks > max_blocks:
                    max_blocks = blocks
                    best_x = x
                    best_ties = [x]
                elif blocks == max_blocks:
                    best_ties.append(x)
            
            # Random tie-breaking to explore different structures
            chosen_x = random.choice(best_ties)
            
            S.append(chosen_x)
            used_sums = update_sums(chosen_x, S[:-1], used_sums)
            candidates.remove(chosen_x)
            
        # Ensure maximality
        # The greedy loop stops when no valid candidates exist.
        # This implies that for all x not in S, x cannot be added to S.
        # Thus S is maximal.
        
        return S

    while time.time() < deadline:
        S = run_greedy()
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
        
        # If we find a very small set, we can stop early
        if best_size <= 15:
            break

    if best_set is None:
        # Fallback: simple greedy
        S = []
        used_sums = 0
        for x in range(1, N + 1):
            if can_add(x, S, used_sums):
                S.append(x)
                used_sums = update_sums(x, S[:-1], used_sums)
        return S

    return best_set
