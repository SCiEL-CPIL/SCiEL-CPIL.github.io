import time
import random

def construct():
    N = 512
    # Bit i represents element i+1
    
    # Precompute conflict masks
    # conflicts[x_idx] is a list of masks. 
    # A mask M in conflicts[x_idx] represents a set of 3 elements {a, b, c} 
    # such that {x, a, b, c} forms a 4-AP.
    # If (S & M) == M, then adding x to S creates a 4-AP.
    
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # 1 <= a, a+3d <= N
    # d >= 1
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Convert to 0-indexed bits
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            c_bit = 1 << (c - 1)
            e_bit = 1 << (e - 1)
            
            # For each element in the AP, the other 3 form a conflict mask
            # Element a (index a-1)
            mask_bce = b_bit | c_bit | e_bit
            conflicts[a - 1].append(mask_bce)
            
            # Element b (index b-1)
            mask_ace = a_bit | c_bit | e_bit
            conflicts[b - 1].append(mask_ace)
            
            # Element c (index c-1)
            mask_abe = a_bit | b_bit | e_bit
            conflicts[c - 1].append(mask_abe)
            
            # Element e (index e-1)
            mask_abc = a_bit | b_bit | c_bit
            conflicts[e - 1].append(mask_abc)

    def can_add(x_idx, S):
        """Check if element x_idx can be added to set S without creating a 4-AP."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def count_bits(S):
        return bin(S).count('1')

    def greedy_fill(S, order):
        """Add elements from order to S if they don't create a 4-AP."""
        for x_idx in order:
            if not (S & (1 << x_idx)):
                if can_add(x_idx, S):
                    S |= (1 << x_idx)
        return S

    start_time = time.time()
    # Leave a safety margin. The evaluator kills at 90s, but we want to be safe.
    # 55s is a good target to ensure we return a valid result.
    deadline = start_time + 55.0 

    best_S = 0
    best_size = 0
    current_S = 0
    current_size = 0

    # 1. Initial Greedy Constructions
    # Try a few random orderings to get a good starting point
    base_order = list(range(N))
    
    # Always try natural order first
    S_try = greedy_fill(0, base_order)
    size_try = count_bits(S_try)
    if size_try > best_size:
        best_size = size_try
        best_S = S_try
        current_S = S_try
        current_size = size_try

    # Try random orderings
    for _ in range(50):
        random.shuffle(base_order)
        S_try = greedy_fill(0, base_order)
        size_try = count_bits(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
            current_S = S_try
            current_size = size_try

    # 2. Iterated Local Search (ILS) with Adaptive Perturbation and Swaps
    # Strategy: 
    # 1. Monitor stagnation.
    # 2. If not stagnant, try small perturbations (swaps or small removes).
    # 3. If stagnant, increase perturbation strength (larger removes).
    # 4. Always refill greedily after perturbation.
    
    iterations = 0
    stagnation = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Determine move type based on stagnation
        # If we've been stuck for a while, we need a bigger jump.
        # Otherwise, try to fine-tune with swaps or small removals.
        
        if stagnation > 10:
            # High stagnation: Large perturbation
            # Remove 5-10 random elements
            k = random.randint(5, 10)
            move_type = 'remove_fill'
        elif stagnation > 5:
            # Medium stagnation: Medium perturbation
            # Remove 2-5 random elements
            k = random.randint(2, 5)
            move_type = 'remove_fill'
        else:
            # Low stagnation: Try swaps or very small removals
            # 50% chance of swap, 50% chance of small remove
            if random.random() < 0.5:
                move_type = 'swap'
                k = 1
            else:
                move_type = 'remove_fill'
                k = 1

        elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
        
        if not elements_in_set:
            continue

        if move_type == 'swap':
            # Swap Move: Remove one element, try to add a different one
            u = random.choice(elements_in_set)
            S_temp = current_S & ~(1 << u)
            
            # Try to find a v that can be added to S_temp
            # We prioritize v's that are not in the set
            # To speed up, we don't iterate all N, we just try a few randoms
            # Or better: iterate all candidates, but that's O(N). 
            # Given N=512, O(N) is fine.
            # Let's just try to add ANY valid element to S_temp to maximize size.
            # Actually, a true swap keeps size constant. 
            # But if we can add an element, size increases, which is even better.
            # So "Swap" here effectively means "Remove u, then Greedy Fill".
            # This is similar to remove_fill with k=1, but we specifically removed u.
            
            # To make it distinct and potentially better, let's just do a greedy fill 
            # after removing u. The greedy fill will try to add things back, 
            # possibly including u, or adding something else.
            
            order = list(range(N))
            random.shuffle(order)
            S_filled = greedy_fill(S_temp, order)
            size_filled = count_bits(S_filled)
            
            if size_filled > best_size:
                best_size = size_filled
                best_S = S_filled
                stagnation = 0
            else:
                stagnation += 1
                
            current_S = S_filled
            current_size = size_filled

        else: # move_type == 'remove_fill'
            # Remove and Fill
            if len(elements_in_set) < k:
                k = len(elements_in_set)
            if k == 0:
                continue
                
            removed = random.sample(elements_in_set, k)
            S_temp = current_S
            for idx in removed:
                S_temp &= ~(1 << idx)
            
            # Run multiple greedy fills to find the best local optimum
            # Since we have more time budget per iteration if we are efficient, 
            # we can try a few different orders.
            # However, to keep it fast, let's do 1-3 fills.
            num_fills = 3 if k > 2 else 5
            
            best_local_S = S_temp
            best_local_size = count_bits(S_temp)
            
            for _ in range(num_fills):
                order = list(range(N))
                random.shuffle(order)
                S_filled = greedy_fill(S_temp, order)
                size_filled = count_bits(S_filled)
                
                if size_filled > best_local_size:
                    best_local_size = size_filled
                    best_local_S = S_filled
            
            # Update global best
            if best_local_size > best_size:
                best_size = best_local_size
                best_S = best_local_S
                stagnation = 0
            else:
                stagnation += 1
            
            # Move to the new local optimum
            current_S = best_local_S
            current_size = best_local_size

    # Return the best set found
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
