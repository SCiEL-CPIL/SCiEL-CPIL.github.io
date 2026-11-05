import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph using bitmasks for speed
    # conflicts[i] is a bitmask of j such that i+j is a square
    # We use 0-indexed for internal representation, 1-indexed for values
    # val = idx + 1
    conflicts = [0] * N
    
    for i in range(N):
        val_i = i + 1
        mask = 0
        for j in range(N):
            val_j = j + 1
            if val_i + val_j in squares:
                mask |= (1 << j)
        conflicts[i] = mask
        
    # Helper to check if a set (bitmask) is an independent set
    # Not strictly needed if we maintain invariant, but good for verification
    
    # Strategy: Branch and Bound with Bit-Parallelism
    # We want to find the maximum independent set.
    # Since the graph is not sparse enough for simple greedy to be optimal, 
    # and N=256 is too large for pure exponential search, we use a hybrid:
    # 1. Greedy coloring to get an upper bound.
    # 2. Recursive search with pruning.
    # However, 2^256 is too big. We need to exploit structure.
    
    # Observation: The conflict graph has a strong modular structure.
    # If a+b is a square, a+b is in {4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256}
    # Actually squares up to 512: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484
    
    # Let's try a different approach: 
    # The graph is "almost" bipartite or has small cliques?
    # Let's use a heuristic search with strong pruning.
    # We will use a recursive backtracking with a greedy upper bound.
    
    # To make it fast, we order vertices by degree (most constrained first).
    # But we need to be careful with time.
    
    # Alternative: Use the fact that the graph is a "sum graph".
    # Let's try a local search that is deterministic and structured, 
    # combined with a bit-parallel check.
    
    # Actually, let's try a Max-Independent-Set solver using a specific library? No, stdlib only.
    
    # Let's try a Greedy + Local Search but with BITMASKS for speed.
    # This allows us to do many more iterations in 55s.
    
    # Initial set: Greedy by degree
    # Or use the modular seed: x % 3 == 1
    # Let's use multiple seeds and pick the best.
    
    best_set = set()
    best_size = 0
    
    def evaluate_and_optimize(initial_set):
        # Convert to bitmask for fast operations
        # current_mask: bits set for elements in current_set
        current_mask = 0
        for v in initial_set:
            current_mask |= (1 << (v - 1))
            
        # Precompute "conflict_mask" for each vertex? Already have conflicts[i].
        
        # Helper: can_add(v, current_mask)
        # v is 1-indexed value. idx = v-1.
        # Conflict if (conflicts[idx] & current_mask) != 0
        
        def can_add(idx, mask):
            return (conflicts[idx] & mask) == 0
            
        def get_size(mask):
            return bin(mask).count('1')
            
        # Local Search Loop
        improved = True
        while improved:
            improved = False
            # Try adding any vertex
            # To speed up, iterate through candidates not in set
            # We can find candidates by checking all 256, but that's slow in Python loop.
            # Better: maintain a list of candidates? Or just loop 0..255.
            # 256 checks is fast.
            
            for idx in range(N):
                if not (current_mask & (1 << idx)):
                    if can_add(idx, current_mask):
                        current_mask |= (1 << idx)
                        improved = True
            
            # If no single addition works, try 1-for-1 swaps
            # This is O(N^2) per iteration if done naively.
            # Let's do a random subset of swaps to keep it fast.
            
            if not improved:
                # Try swaps
                # Pick random u in set, random v not in set
                # If removing u allows adding v, and v doesn't conflict with rest
                # Actually, if we remove u, we can add v if v doesn't conflict with (set - u).
                # And we want to keep size same or increase. 
                # If we just swap, size stays same. We want to find a sequence that allows growth.
                # Simpler: Try to remove u, then greedily add as many as possible.
                
                # Let's try a "kick" move: remove a random element, then greedily add back.
                if current_mask:
                    # Pick a random element to remove
                    # Get list of set bits
                    set_bits = []
                    m = current_mask
                    while m:
                        lsb = m & -m
                        idx = (lsb.bit_length() - 1)
                        set_bits.append(idx)
                        m ^= lsb
                    
                    if set_bits:
                        u_idx = random.choice(set_bits)
                        # Remove u
                        new_mask = current_mask & ~(1 << u_idx)
                        
                        # Greedily add back
                        added = 0
                        # Try to add all possible
                        for idx in range(N):
                            if not (new_mask & (1 << idx)):
                                if can_add(idx, new_mask):
                                    new_mask |= (1 << idx)
                                    added += 1
                                    
                        if get_size(new_mask) > get_size(current_mask):
                            current_mask = new_mask
                            improved = True
                            break # Restart loop with new mask
        
        return current_mask, get_size(current_mask)

    # Run multiple iterations with different seeds
    # Seed 1: x % 3 == 1
    # Seed 2: x % 4 in {1, 2} ? Check validity.
    # Seed 3: Random
    
    seeds = []
    
    # Modular seeds
    for mod in [3, 4, 5, 6, 7, 8, 12]:
        for residue in range(mod):
            seed_set = set()
            for v in vertices:
                if v % mod == residue:
                    seed_set.add(v)
            # Check if seed is valid (independent set)
            valid = True
            s_list = list(seed_set)
            for i in range(len(s_list)):
                for j in range(i, len(s_list)):
                    if s_list[i] + s_list[j] in squares:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                seeds.append(seed_set)
                
    # Random seeds
    for _ in range(10):
        seed_set = set(random.sample(vertices, 50))
        # Clean it to be valid
        valid_set = set()
        for v in sorted(seed_set, key=lambda x: random.random()):
            if all((v + u) not in squares for u in valid_set):
                valid_set.add(v)
        seeds.append(valid_set)
        
    # Also include the original x%3==1
    seeds.append(set(v for v in vertices if v % 3 == 1))
    
    # Evaluate each seed with local search
    # We have ~55s. 
    # Let's distribute time.
    
    num_seeds = len(seeds)
    time_per_seed = (DEADLINE - (time.time() - start_time)) / max(1, num_seeds)
    
    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
            
        # Run local search on this seed
        # We'll run it for a fraction of the remaining time or a fixed number of steps
        # Since the local search above is a "while improved" loop, it might finish fast.
        # We should add a time check inside the loop.
        
        # Modify evaluate_and_optimize to respect time? 
        # Or just call it and hope it's fast. 
        # The bitmask operations are very fast.
        
        mask, size = evaluate_and_optimize(seed)
        
        if size > best_size:
            best_size = size
            # Convert mask back to set
            best_set = set()
            m = mask
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                best_set.add(idx + 1)
                m ^= lsb
                
        # If we have time, do more random restarts
        while time.time() - start_time < DEADLINE - 5.0:
            # Random restart
            seed_set = set(random.sample(vertices, 20))
            # Clean
            valid_set = set()
            for v in sorted(seed_set, key=lambda x: random.random()):
                if all((v + u) not in squares for u in valid_set):
                    valid_set.add(v)
            
            mask, size = evaluate_and_optimize(valid_set)
            
            if size > best_size:
                best_size = size
                best_set = set()
                m = mask
                while m:
                    lsb = m & -m
                    idx = (lsb.bit_length() - 1)
                    best_set.add(idx + 1)
                    m ^= lsb
                    
    # Final verification
    if not best_set:
        # Fallback
        best_set = set(v for v in vertices if v % 3 == 1)
        
    best_list = list(best_set)
    
    # Verify validity
    valid = True
    for i in range(len(best_list)):
        a = best_list[i]
        for j in range(i, len(best_list)):
            b = best_list[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed if something went wrong
        return list(range(1, 257, 3))
        
    return best_list
