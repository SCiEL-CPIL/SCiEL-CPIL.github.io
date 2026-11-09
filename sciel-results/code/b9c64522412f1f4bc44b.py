import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps_list.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps_list, dtype=np.int32)
    M = len(aps_arr)
    
    def count_mono(c):
        # c is array of 0/1
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
        
    def greedy_coordinate_descent(c):
        # Vectorized Greedy Coordinate Descent
        # Iteratively compute delta for flipping every bit i in 0..n-1
        # and flip those that improve the score.
        # Stop when no improvements are found.
        
        # Precompute position of each element in each AP for fast delta calculation
        # For each AP index k, we have indices p0, p1, p2, p3.
        # We want to know, for each position p, the sum of deltas if we flip c[p].
        
        # Delta for flipping p in AP k:
        # If c[p] is the unique value, flipping it makes the AP monochromatic (delta +1).
        # If c[p] is part of a majority (3 same, 1 different), flipping it breaks the AP (delta -1).
        # If c[p] is part of a 2-2 split, flipping it keeps it non-mono (delta 0).
        
        # We can compute this by checking the state of each AP before and after flip.
        # However, a simpler way is to compute the "pressure" on each bit.
        
        # Let's use the property:
        # A 4-term AP is monochromatic if all 4 are equal.
        # Flipping bit p changes the count by:
        # +1 if the AP was NOT mono, and becomes mono after flip.
        # -1 if the AP WAS mono, and becomes non-mono after flip.
        
        # Vectorized approach:
        # For each AP, we can determine if it is currently mono.
        # If we flip p, is it mono?
        
        # This is hard to vectorize directly for "all p at once" because each p belongs to many APs.
        # Instead, we can compute the delta for each p by summing contributions from all APs it belongs to.
        
        # Let's precompute for each position p, the list of (ap_idx, pos_in_ap).
        # This was done in the parent but removed. Let's recreate it efficiently or use a different trick.
        
        # Alternative: 
        # We can compute the current mono count.
        # Then, for each p, we can simulate the flip and check the APs.
        # But doing this for all p in a loop is slow (n * avg_aps_per_p).
        
        # Let's try a different vectorization:
        # For each AP, we can compute the number of 0s and 1s.
        # If count_0 == 4 or count_1 == 4, it is mono.
        
        # If we flip p, the count changes.
        
        # Let's stick to a slightly optimized loop for GCD if full vectorization is complex, 
        # but the prompt suggests vectorized NumPy.
        
        # Actually, we can compute the delta for all p at once using matrix operations if we represent the APs as a matrix.
        # But M is ~10k and n is 256.
        
        # Let's use the precomputed pos_to_aps structure but vectorize the delta calculation inside the loop over p? 
        # No, that's still a loop over p.
        
        # Let's try to compute the delta for all p in one go.
        # For each AP k, let the values be v0, v1, v2, v3.
        # If we flip p_i (where p_i is one of the 4 positions), the new values are v0', v1', v2', v3'.
        # The AP becomes mono if v0'=v1'=v2'=v3'.
        
        # This is equivalent to: 
        # The AP is mono if (v0==v1) and (v1==v2) and (v2==v3).
        
        # Let's define a function that, given the current colors c, computes the delta for flipping each p.
        
        # We can iterate over APs and accumulate the delta for each position.
        # This is O(M * 4) which is ~40k operations, very fast.
        
        delta = np.zeros(n, dtype=np.int32)
        
        # For each AP, check if it is currently mono.
        # If it is mono, flipping any of its 4 bits will make it non-mono (delta -1 for each of the 4 bits).
        # If it is not mono, we check if flipping a specific bit makes it mono.
        # An AP with values v0,v1,v2,v3 is not mono.
        # It becomes mono if and only if the 3 unflipped bits are all equal.
        # So if v1==v2==v3, flipping v0 makes it mono (delta +1 for v0).
        # If v0==v2==v3, flipping v1 makes it mono (delta +1 for v1).
        # If v0==v1==v3, flipping v2 makes it mono (delta +1 for v2).
        # If v0==v1==v2, flipping v3 makes it mono (delta +1 for v3).
        
        # We can vectorize this over all APs.
        
        v0 = c[aps_arr[:, 0]]
        v1 = c[aps_arr[:, 1]]
        v2 = c[aps_arr[:, 2]]
        v3 = c[aps_arr[:, 3]]
        
        # Current mono status
        is_mono = (v0 == v1) & (v1 == v2) & (v2 == v3)
        
        # Case 1: Currently mono. Flipping any bit makes it non-mono.
        # So for each AP that is mono, we add -1 to the delta of its 4 positions.
        mono_aps = aps_arr[is_mono]
        if len(mono_aps) > 0:
            # We need to add -1 to delta[p] for each p in mono_aps.
            # We can use np.add.at
            np.add.at(delta, mono_aps[:, 0], -1)
            np.add.at(delta, mono_aps[:, 1], -1)
            np.add.at(delta, mono_aps[:, 2], -1)
            np.add.at(delta, mono_aps[:, 3], -1)
            
        # Case 2: Currently not mono.
        non_mono_aps = aps_arr[~is_mono]
        if len(non_mono_aps) > 0:
            nv0 = c[non_mono_aps[:, 0]]
            nv1 = c[non_mono_aps[:, 1]]
            nv2 = c[non_mono_aps[:, 2]]
            nv3 = c[non_mono_aps[:, 3]]
            
            # Check if flipping v0 makes it mono: nv1 == nv2 == nv3
            flip_0_makes_mono = (nv1 == nv2) & (nv2 == nv3)
            if np.any(flip_0_makes_mono):
                np.add.at(delta, non_mono_aps[flip_0_makes_mono, 0], 1)
                
            # Check if flipping v1 makes it mono: nv0 == nv2 == nv3
            flip_1_makes_mono = (nv0 == nv2) & (nv2 == nv3)
            if np.any(flip_1_makes_mono):
                np.add.at(delta, non_mono_aps[flip_1_makes_mono, 1], 1)
                
            # Check if flipping v2 makes it mono: nv0 == nv1 == nv3
            flip_2_makes_mono = (nv0 == nv1) & (nv1 == nv3)
            if np.any(flip_2_makes_mono):
                np.add.at(delta, non_mono_aps[flip_2_makes_mono, 2], 1)
                
            # Check if flipping v3 makes it mono: nv0 == nv1 == nv2
            flip_3_makes_mono = (nv0 == nv1) & (nv1 == nv2)
            if np.any(flip_3_makes_mono):
                np.add.at(delta, non_mono_aps[flip_3_makes_mono, 3], 1)
                
        return delta

    def run_gcd(initial_colors):
        c = initial_colors.copy().astype(np.int8)
        current_score = count_mono(c)
        best_score = current_score
        best_c = c.copy()
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            delta = greedy_coordinate_descent(c)
            # Find all positions where delta < 0 (improvement)
            improve_mask = delta < 0
            if not np.any(improve_mask):
                improved = False
                break
            
            # Flip all improving positions simultaneously
            c[improve_mask] = 1 - c[improve_mask]
            # The score change is sum of deltas for those positions
            # Note: simultaneous flips might interact, but since we computed delta based on current state,
            # and we are doing a "greedy" step, we should recompute score to be safe or assume small interactions.
            # Actually, if two bits in the same AP are both flipped, the delta calculation might be slightly off 
            # if they were both -1. But usually, if an AP is mono, all 4 are -1. If we flip all 4, it becomes non-mono (correct).
            # If an AP is not mono, and two bits are flipped... e.g. 0001, flipping 0 and 1 gives 1101 (not mono). 
            # Delta for 0 was +1 (0001 -> 1001, wait. 0001, flip 0 -> 1001. 1,0,0,1 not mono. Delta 0? 
            # My logic: 0001. nv1=0,nv2=0,nv3=1. nv1!=nv3. So flip 0 does NOT make it mono. Delta 0.
            # So only bits that *make* the AP mono get +1.
            # If we flip multiple bits, we might create new mono APs or break existing ones in ways not captured.
            # To be safe, we should recompute the score.
            
            new_score = count_mono(c)
            if new_score < best_score:
                best_score = new_score
                best_c = c.copy()
            else:
                # If simultaneous flip didn't improve, we might need to be more careful.
                # But usually, it does. If it doesn't, we can break or revert.
                # For simplicity, if it doesn't improve, we break the loop for this run.
                break
                
            current_score = new_score
            
        return best_score, best_c

    # Generate initial colorings
    initial_colors_list = []
    
    # Thue-Morse
    def thue_morse(i):
        return bin(i).count('1') % 2
    initial_colors_list.append(np.array([thue_morse(i) for i in range(n)], dtype=np.int8))
    
    # Random
    initial_colors_list.append(np.random.randint(0, 2, n).astype(np.int8))
    
    # Random biased
    initial_colors_list.append((np.random.rand(n) < 0.3).astype(np.int8))
    initial_colors_list.append((np.random.rand(n) < 0.7).astype(np.int8))
    
    # Structured: Alternating blocks
    initial_colors_list.append(np.array([i % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8))
    initial_colors_list.append(np.array([(i // 8) % 2 for i in range(n)], dtype=np.int8))
    
    # Prime-based
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0: return False
        return True
    initial_colors_list.append(np.array([1 if is_prime(i+1) else 0 for i in range(n)], dtype=np.int8))
    
    # More randoms
    for _ in range(10):
        initial_colors_list.append(np.random.randint(0, 2, n).astype(np.int8))
        
    best_score_global = float('inf')
    best_colors_global = None
    
    # Run GCD for each initial coloring
    # GCD is deterministic and fast, so we can run many restarts
    for init_c in initial_colors_list:
        if time.time() > deadline:
            break
            
        score, c = run_gcd(init_c)
        if score < best_score_global:
            best_score_global = score
            best_colors_global = c
            
    # If we have time left, do more random restarts
    while time.time() < deadline - 5.0:
        init_c = np.random.randint(0, 2, n).astype(np.int8)
        score, c = run_gcd(init_c)
        if score < best_score_global:
            best_score_global = score
            best_colors_global = c
            
    if best_colors_global is None:
        return [int(c) for c in initial_colors_list[0]]
        
    return [int(c) for c in best_colors_global]
