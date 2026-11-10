import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # 1. Precompute all 4-APs and conflict masks
    # masks[x] is a list of integers (bitsets) representing the other 3 elements
    # in each 4-AP containing x.
    masks = [[] for _ in range(N + 1)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            # Create bitmask for the AP
            m = (1 << x1) | (1 << x2) | (1 << x3) | (1 << x4)
            
            # Add mask for each element (the other 3 elements)
            masks[x1].append(m ^ (1 << x1))
            masks[x2].append(m ^ (1 << x2))
            masks[x3].append(m ^ (1 << x3))
            masks[x4].append(m ^ (1 << x4))

    def can_add(x, S):
        """Check if x can be added to set S (represented as int bitset)"""
        for m in masks[x]:
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

    # Heuristic 2: Random Greedy (Multiple seeds)
    for seed in range(50):
        if time.time() > deadline - 10:
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

    # Heuristic 3: Greedy from base4
    S = S_base4
    order = list(range(1, N + 1))
    random.shuffle(order)
    S = greedy_add(S, order)
    if S.bit_count() > best_size:
        best_set = S
        best_size = S.bit_count()

    # 3. Local Search
    # We will alternate between:
    # A. 1-Remove, Greedy Repair (Unlock move)
    # B. 2-Remove, Greedy Repair (Perturbation)
    # C. 1-Swap (Refinement)
    
    rng = random.Random(42)
    
    while time.time() < deadline - 2:
        # Choose a strategy based on time or random
        # Strategy A is usually the most effective for jumps
        if rng.random() < 0.7:
            # 1-Remove, Repair
            S = best_set
            s_size = S.bit_count()
            
            # Get list of elements in S
            elements_in_S = [i for i in range(1, N + 1) if (S >> i) & 1]
            
            if not elements_in_S:
                continue
                
            # Try removing a random subset of elements (just 1 usually, maybe 2)
            # Let's try 1 first
            u = rng.choice(elements_in_S)
            S_temp = S & ~(1 << u)
            
            # Repair: Add candidates in random order
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            S_new = greedy_add(S_temp, candidates)
            
            if S_new.bit_count() > best_size:
                best_set = S_new
                best_size = S_new.bit_count()
            elif S_new.bit_count() == best_size:
                # Accept with some probability to explore?
                # Or just keep best.
                pass
            else:
                # Try a second random removal if first didn't help?
                # Or try removing a different u
                # To save time, just restart loop.
                pass
                
        else:
            # Strategy B: 2-Remove, Repair
            S = best_set
            s_size = S.bit_count()
            
            elements_in_S = [i for i in range(1, N + 1) if (S >> i) & 1]
            
            if len(elements_in_S) < 2:
                continue
                
            u1, u2 = rng.sample(elements_in_S, 2)
            S_temp = S & ~(1 << u1) & ~(1 << u2)
            
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            S_new = greedy_add(S_temp, candidates)
            
            if S_new.bit_count() > best_size:
                best_set = S_new
                best_size = S_new.bit_count()

    # Final Polish: 1-Swaps
    # Try to find a swap that increases size or maintains size but allows future growth?
    # Given time constraints, we might skip complex polish if we hit deadline.
    # But a quick pass of 1-swaps might help.
    
    if time.time() < deadline - 0.5:
        S = best_set
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            elements_in_S = [i for i in range(1, N + 1) if (S >> i) & 1]
            elements_out_S = [i for i in range(1, N + 1) if not ((S >> i) & 1)]
            
            if not elements_in_S or not elements_out_S:
                break
                
            # Try a limited number of swaps
            for _ in range(50):
                if time.time() > deadline - 0.5:
                    break
                u = rng.choice(elements_in_S)
                v = rng.choice(elements_out_S)
                
                S_temp = S & ~(1 << u)
                if can_add(v, S_temp):
                    S_new = S_temp | (1 << v)
                    # Try to add more after swap
                    # Use a short greedy pass
                    candidates = elements_out_S[:]
                    rng.shuffle(candidates)
                    # Filter out u? u is already removed.
                    # We want to add anything not in S_new
                    # Note: v is already in S_new.
                    S_final = greedy_add(S_new, candidates)
                    
                    if S_final.bit_count() > S.bit_count():
                        S = S_final
                        improved = True
                        break
                    elif S_final.bit_count() == S.bit_count() and S_final != S:
                        # Accept neutral swap to change structure?
                        # Risky, might cycle.
                        # Let's only accept if it's strictly better or equal but different structure.
                        # For safety, only accept if strictly better.
                        pass

        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # Convert bitset to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    return sorted(result)
