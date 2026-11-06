import numpy as np
import time
import random
import sys

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # 1. Generate all APs of length >= 2
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            terms = list(range(start, n, step))
            if len(terms) >= 2:
                ap_indices.append(terms)
    num_aps = len(ap_indices)

    # Precompute inverse mapping: pos -> list of ap_ids
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)

    # 2. Helper: Compute AP sums
    def compute_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, idx in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[idx])
        return sums

    # 3. Helper: Greedy local search (fast)
    def greedy_local_search(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = compute_sums(signs)
        current_max = np.max(np.abs(ap_sums))
        best_signs = signs.copy()
        best_max = current_max
        
        # Precompute affected APs for each position for speed? 
        # pos_to_aps is already a list of lists.
        
        # To speed up max query, we can't easily use histogram in Python loops without overhead.
        # But np.max is fast enough if we do it less often? 
        # Actually, let's just compute max every time.
        
        no_improve = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random position
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = pos_to_aps[i]
            
            # Apply
            for ap_id in affected:
                ap_sums[ap_id] += delta
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                signs[i] = new_sign
                current_max = new_max
                no_improve = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            elif new_max == current_max and rng.random() < 0.1:
                # Random walk to escape plateaus
                signs[i] = new_sign
                current_max = new_max
                no_improve += 1
                # Revert if no improvement in last 100 steps?
            else:
                # Revert
                for ap_id in affected:
                    ap_sums[ap_id] -= delta
                no_improve += 1
            
            if no_improve > 50:
                break
                
        return best_signs, best_max

    # 4. Helper: SA local search (slower but better escape)
    def sa_local_search(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = compute_sums(signs)
        current_max = np.max(np.abs(ap_sums))
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 2.0
        cooling = 0.999
        min_temp = 0.05
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = pos_to_aps[i]
            
            # Apply
            for ap_id in affected:
                ap_sums[ap_id] += delta
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                signs[i] = new_sign
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                delta_max = new_max - current_max
                if temperature > min_temp and rng.random() < np.exp(-delta_max / temperature):
                    signs[i] = new_sign
                    current_max = new_max
                else:
                    # Revert
                    for ap_id in affected:
                        ap_sums[ap_id] -= delta
                    
            temperature *= cooling
            
        return best_signs, best_max

    # 5. Main loop
    best_overall_signs = None
    best_overall_max = n + 1
    
    # Start with Thue-Morse
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs, best_max = greedy_local_search(thue_morse, 2000, deadline, random.Random(42))
    best_overall_signs = best_signs
    best_overall_max = best_max
    
    # Try different initializations and search types
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 9999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10101, 20202]
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
            
        rng = random.Random(seed)
        
        # Random initialization
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        
        # Use SA for some, Greedy for others
        if seed % 2 == 0:
            best_signs, best_max = sa_local_search(signs, 3000, deadline, rng)
        else:
            best_signs, best_max = greedy_local_search(signs, 3000, deadline, rng)
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall_signs = best_signs.copy()
            
        # If we hit 2, try harder to get 1
        if best_overall_max <= 2:
             # Intensive SA on best solution
             rng2 = random.Random(seed * 7919)
             best_signs2, best_max2 = sa_local_search(best_overall_signs, 5000, deadline, rng2)
             if best_max2 < best_overall_max:
                 best_overall_max = best_max2
                 best_overall_signs = best_signs2.copy()

    # Final polish on best solution
    if best_overall_signs is not None:
        rng3 = random.Random(31337)
        final_signs, final_max = greedy_local_search(best_overall_signs, 10000, deadline, rng3)
        if final_max < best_overall_max:
            best_overall_max = final_max
            best_overall_signs = final_signs

    if best_overall_signs is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
        
    return [int(s) for s in best_overall_signs]
