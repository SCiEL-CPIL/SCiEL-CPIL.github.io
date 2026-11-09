import random
import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    N = 511
    # Weights: 1 / (512 - a)
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        
    # Precompute conflict masks
    # conflict_mask[a] is a bitmask of all x in 1..N such that gcd(a, x) > 1
    # We use Python integers as bitsets. Bit i is set if i is in the set.
    # Note: Bit 0 is unused or used for 1? Let's use bit a for number a.
    # So bit 1 for 1, bit 2 for 2, ..., bit 511 for 511.
    
    conflict_masks = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_masks[a] = mask
        
    # Greedy initialization
    # Sort by weight descending (which is a ascending? No, weight is 1/(512-a).
    # Larger a -> smaller denominator -> larger weight.
    # So sort a descending.
    candidates = list(range(N, 0, -1))
    
    def greedy_construct(order):
        S = []
        occupied_mask = 0
        current_score = 0.0
        for a in order:
            # Check if a is coprime to all in S
            # a is coprime to S if conflict_masks[a] & occupied_mask == 0
            if (conflict_masks[a] & occupied_mask) == 0:
                S.append(a)
                occupied_mask |= (1 << a)
                current_score += weights[a]
        return S, occupied_mask, current_score

    # Run greedy
    S, occupied_mask, current_score = greedy_construct(candidates)
    best_S = S[:]
    best_score = current_score
    best_mask = occupied_mask
    
    # Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 0.001
    cooling = 0.999
    T = T_start
    
    # We will iterate until time limit or T is very low
    # To make it fast, we pre-calculate valid candidates for swaps?
    # No, just random sampling.
    
    # Helper to check validity of adding v to S excluding u
    def is_valid_swap(u, v, S, occupied_mask):
        # v must not be in S
        if (occupied_mask >> v) & 1:
            return False
        # v must be coprime to all x in S except u
        # Mask of S without u
        mask_without_u = occupied_mask ^ (1 << u)
        # Check conflict
        if (conflict_masks[v] & mask_without_u) != 0:
            return False
        return True

    # Helper to check validity of adding v to S
    def is_valid_add(v, S, occupied_mask):
        if (occupied_mask >> v) & 1:
            return False
        if (conflict_masks[v] & occupied_mask) != 0:
            return False
        return True

    # Helper to check validity of removing u from S
    # Always valid to remove
    
    # Main SA Loop
    # We need to track S as a list for easy iteration and set operations for score?
    # Score update: delta = w_new - w_old.
    
    # To speed up, we can maintain a set of S for O(1) lookup?
    # But we have occupied_mask.
    
    # Let's run SA
    while T > T_end and time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap: remove u, add v
            if not S:
                continue
            u = random.choice(S)
            # Try to find a valid v
            # Random v from 1..N
            v = random.randint(1, N)
            if is_valid_swap(u, v, S, occupied_mask):
                delta = weights[v] - weights[u]
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Accept
                    S.remove(u)
                    S.append(v)
                    occupied_mask = (occupied_mask ^ (1 << u)) | (1 << v)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_mask = occupied_mask
                        
        elif move_type < 0.85:
            # Add: add v
            v = random.randint(1, N)
            if is_valid_add(v, S, occupied_mask):
                delta = weights[v]
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.append(v)
                    occupied_mask |= (1 << v)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_mask = occupied_mask
        else:
            # Remove: remove u
            if S:
                u = random.choice(S)
                delta = -weights[u]
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.remove(u)
                    occupied_mask ^= (1 << u)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_mask = occupied_mask
                        
        T *= cooling
        
    return best_S

# EVOLVE-BLOCK-START
def construct():
    import random
    import math
    import time

    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    N = 511
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        
    conflict_masks = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_masks[a] = mask
        
    candidates = list(range(N, 0, -1))
    
    def greedy_construct(order):
        S = []
        occupied_mask = 0
        current_score = 0.0
        for a in order:
            if (conflict_masks[a] & occupied_mask) == 0:
                S.append(a)
                occupied_mask |= (1 << a)
                current_score += weights[a]
        return S, occupied_mask, current_score

    S, occupied_mask, current_score = greedy_construct(candidates)
    best_S = S[:]
    best_score = current_score
    best_mask = occupied_mask
    
    T_start = 1.0
    T_end = 0.001
    cooling = 0.999
    T = T_start
    
    def is_valid_swap(u, v, occupied_mask):
        if (occupied_mask >> v) & 1:
            return False
        mask_without_u = occupied_mask ^ (1 << u)
        if (conflict_masks[v] & mask_without_u) != 0:
            return False
        return True

    def is_valid_add(v, occupied_mask):
        if (occupied_mask >> v) & 1:
            return False
        if (conflict_masks[v] & occupied_mask) != 0:
            return False
        return True

    while T > T_end and time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.7:
            if not S:
                continue
            u = random.choice(S)
            v = random.randint(1, N)
            if is_valid_swap(u, v, occupied_mask):
                delta = weights[v] - weights[u]
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.remove(u)
                    S.append(v)
                    occupied_mask = (occupied_mask ^ (1 << u)) | (1 << v)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_mask = occupied_mask
                        
        elif move_type < 0.85:
            v = random.randint(1, N)
            if is_valid_add(v, occupied_mask):
                delta = weights[v]
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.append(v)
                    occupied_mask |= (1 << v)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_mask = occupied_mask
        else:
            if S:
                u = random.choice(S)
                delta = -weights[u]
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.remove(u)
                    occupied_mask ^= (1 << u)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S[:]
                        best_mask = occupied_mask
                        
        T *= cooling
        
    return best_S
# EVOLVE-BLOCK-END
