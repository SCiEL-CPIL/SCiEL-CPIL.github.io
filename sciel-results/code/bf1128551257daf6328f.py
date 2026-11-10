import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit

    # 1. Precompute conflict masks
    # masks[x] is a list of bitmasks. Each bitmask represents the other 3 elements
    # in a 4-AP containing x.
    masks = [[] for _ in range(N + 1)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            # Bitmask for the full AP
            m = (1 << x1) | (1 << x2) | (1 << x3) | (1 << x4)
            
            # Store masks for each element (excluding the element itself)
            masks[x1].append(m ^ (1 << x1))
            masks[x2].append(m ^ (1 << x2))
            masks[x3].append(m ^ (1 << x3))
            masks[x4].append(m ^ (1 << x4))

    def can_add(x, S):
        """Check if x can be added to set S (represented as int bitset)"""
        for m in masks[x]:
            # If there are already 3 elements from this AP in S, we cannot add x
            if (m & S).bit_count() >= 3:
                return False
        return True

    def greedy_add(S, candidates):
        """Greedily add elements from candidates to S"""
        for v in candidates:
            if not (S >> v) & 1: # if v not in S
                if can_add(v, S):
                    S |= (1 << v)
        return S

    # 2. Initial Heuristics
    best_set = 0
    best_size = 0
    
    # Heuristic 1: Base-4 (Salem-Spencer)
    # Numbers with base-4 digits only 0,1
    S_base4 = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            S_base4 |= (1 << i)
    
    if S_base4.bit_count() > best_size:
        best_set = S_base4
        best_size = S_base4.bit_count()

    # Heuristic 2: Random Greedy
    # Try several random orders to find a good starting point
    for seed in range(20):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = 0
        for x in order:
            if can_add(x, S):
                S |= (1 << x)
        
        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # 3. Local Search with 2-Exchange
    # We try to improve the set by removing pairs of elements and repacking.
    # This helps escape local optima where single-element swaps are not sufficient.
    
    # List of all elements for convenience
    all_elements = list(range(1, N + 1))
    
    # We will iterate for a while, trying to improve the best set found so far
    # We maintain a "current" set that we modify.
    
    # To avoid getting stuck in a cycle, we track the best set found globally.
    # The local search operates on a copy.
    
    current_set = best_set
    current_size = best_size
    
    # We use a simple restart strategy:
    # 1. Take best_set
    # 2. Perturb (remove random elements)
    # 3. Repair (greedy add + 2-exchange local optimization)
    # 4. If better, update best_set
    
    # Pre-compute list of elements in best_set for perturbation
    # We'll do this dynamically to save memory and time in the loop
    
    while time.time() < deadline - 2:
        # Start from the best known set
        S = best_set
        
        # Perturb: Remove k random elements
        # k=2 or 3 is typical for 2-exchange search
        k = random.randint(2, 4)
        elements_in_S = [i for i in range(1, N + 1) if (S >> i) & 1]
        
        if len(elements_in_S) > k:
            to_remove = random.sample(elements_in_S, k)
            for u in to_remove:
                S &= ~(1 << u)
        
        # Repair Phase 1: Greedy Add
        # Try to add back elements. Random order helps explore different solutions.
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S = greedy_add(S, candidates)
        
        # Repair Phase 2: Local Optimization with 2-Exchange
        # We try to remove a pair (u, v) and see if we can add more elements.
        # We limit the number of attempts to keep it fast.
        
        improved = True
        no_improve_count = 0
        max_no_improve = 5
        
        while improved and time.time() < deadline - 2:
            improved = False
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
            elements_in_S = [i for i in range(1, N + 1) if (S >> i) & 1]
            if len(elements_in_S) < 2:
                break
                
            # Try a limited number of random 2-exchanges
            # We sample pairs to avoid O(N^2) check every iteration
            num_attempts = 200
            for _ in range(num_attempts):
                if time.time() > deadline - 2:
                    break
                    
                # Pick two distinct elements from S
                # Using random.sample is efficient for small k
                u, v = random.sample(elements_in_S, 2)
                
                # Remove u and v
                S_temp = S & ~(1 << u) & ~(1 << v)
                
                # Try to greedily add elements to S_temp
                # We prioritize elements that were just removed? No, just random/shuffled.
                # Actually, a good heuristic is to try adding the removed ones first,
                # then others. But random shuffle of all candidates is robust.
                
                # To speed up, we can just try to add ALL elements in random order.
                # Since S_temp is smaller, can_add might be faster on average? 
                # Not necessarily, but the set is smaller so fewer conflicts.
                
                candidates = list(range(1, N + 1))
                random.shuffle(candidates)
                
                S_new = greedy_add(S_temp, candidates)
                
                new_size = S_new.bit_count()
                old_size = S.bit_count()
                
                if new_size > old_size:
                    S = S_new
                    improved = True
                    no_improve_count = 0
                    # Restart the local optimization loop with the new set
                    # We break out of the attempt loop to restart the while loop
                    break
                elif new_size == old_size:
                    # Accepting equal size moves can help escape local optima,
                    # but it's risky. We accept with low probability to allow exploration.
                    if random.random() < 0.1:
                        S = S_new
                        # Do not set improved = True to prevent infinite loop on same size
                        # But we break to restart the sampling with new structure
                        break

        # Update global best
        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # Convert bitset to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    return sorted(result)

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify no 4-AP
    S = set(res)
    found_ap = False
    for a in S:
        for d in range(1, 512):
            if a + d in S and a + 2*d in S and a + 3*d in S:
                found_ap = True
                print(f"Found AP: {a}, {a+d}, {a+2*d}, {a+3*d}")
                break
        if found_ap:
            break
    if not found_ap:
        print("Valid: No 4-AP found")
