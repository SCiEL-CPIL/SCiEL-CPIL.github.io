import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Precompute APs
    # For each x, list of masks representing the other 3 elements in APs containing x
    other_masks = [[] for _ in range(N + 1)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            # Convert to 0-indexed bits for mask
            # x in 1..N -> bit x-1
            mask = 0
            for x in ap:
                mask |= (1 << (x - 1))
            
            # For each element in ap, the other 3 form a mask
            for x in ap:
                other_mask = mask ^ (1 << (x - 1))
                other_masks[x].append(other_mask)
                
    # Helper: Check if x can be added to S (S is int mask)
    def can_add(x, S):
        for om in other_masks[x]:
            if (S & om) == om:
                return False
        return True
        
    # Helper: Greedily add all possible elements from candidates
    # candidates: list of x not in S
    def greedy_pass(S, candidates):
        added_any = False
        # Shuffle to break symmetry
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S |= (1 << (x - 1))
                added_any = True
        return S, added_any
        
    # Helper: Calculate block counts
    # Returns list of (count, x) for x in S
    def get_block_counts(S):
        # block_count[y] = number of x not in S such that y blocks x
        # y blocks x if there is an AP containing x and y where other 2 are in S
        # Or simpler: x is blocked if exists om in other_masks[x] s.t. S & om == om.
        # If x is blocked by om, then all y in om block x.
        
        block_counts = [0] * (N + 1)
        
        # Iterate over x not in S
        # To be fast, we can just iterate 1..N and check if not in S
        for x in range(1, N + 1):
            if S & (1 << (x - 1)):
                continue # x is in S, skip
                
            # Check if x is blocked
            # We need to find which y in S are in the blocking masks
            # If x is blocked, there is at least one om in other_masks[x] such that S & om == om
            # We want to increment block_counts for all y in that om
            
            # Optimization: Only check if x is actually blocked?
            # If x is not blocked, it doesn't contribute to block counts.
            # But checking if blocked is the same cost as finding the blockers.
            
            for om in other_masks[x]:
                if (S & om) == om:
                    # This om is fully in S. All elements in om block x.
                    # Iterate bits in om
                    m = om
                    while m:
                        # Get lowest set bit
                        lsb = m & -m
                        y = lsb.bit_length()
                        block_counts[y] += 1
                        m ^= lsb
                    # Note: x might be blocked by multiple APs, 
                    # but for the purpose of "who blocks x", 
                    # incrementing for each blocking AP is a valid weight.
                    
        return block_counts

    # Initial Construction: Randomized Greedy
    # We run a few quick greedy passes to get a good starting point
    best_S = 0
    best_size = 0
    
    # Quick Greedy Init
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    S = 0
    for x in candidates:
        if can_add(x, S):
            S |= (1 << (x - 1))
            
    best_S = S
    best_size = S.bit_count()
    
    # Local Search
    current_S = best_S
    current_size = best_size
    
    while time.time() < deadline:
        # 1. Greedy Pass: Try to add any free elements
        # Candidates are those not in current_S
        # Creating the list of candidates is O(N)
        candidates = [x for x in range(1, N + 1) if not (current_S & (1 << (x - 1)))]
        
        new_S, added_any = greedy_pass(current_S, candidates)
        
        if added_any:
            current_S = new_S
            current_size = current_S.bit_count()
            if current_size > best_size:
                best_size = current_size
                best_S = current_S
        else:
            # Stuck. Need to kick.
            # Calculate block counts
            block_counts = get_block_counts(current_S)
            
            # Find y in S with max block count
            max_count = 0
            kick_y = -1
            for y in range(1, N + 1):
                if current_S & (1 << (y - 1)):
                    if block_counts[y] > max_count:
                        max_count = block_counts[y]
                        kick_y = y
                        
            if kick_y == -1:
                # No element in S has a block count?
                # This implies no x outside S is blocked?
                # If so, we should have added them.
                # This case shouldn't happen if greedy_pass was correct.
                # But if it does, pick a random element to remove to break symmetry
                S_list = [y for y in range(1, N + 1) if current_S & (1 << (y - 1))]
                if S_list:
                    kick_y = random.choice(S_list)
                else:
                    break # Empty set?
                    
            # Remove kick_y
            current_S &= ~(1 << (kick_y - 1))
            current_size -= 1
            
            # After removal, we might be able to add more than just 1.
            # The next iteration of the loop will handle the greedy pass.
            
    # Convert best_S to sorted list
    result = []
    for i in range(1, N + 1):
        if best_S & (1 << (i - 1)):
            result.append(i)
            
    return result
