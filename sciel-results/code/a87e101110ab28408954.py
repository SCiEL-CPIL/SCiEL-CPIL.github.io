import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps_list = []
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            aps_list.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps_list)
    aps_arr = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_violations_vec(colors_np):
        """Vectorized violation count using numpy."""
        # colors_np: shape (N,)
        # For each AP, get the 4 colors
        c = colors_np[aps_arr]  # shape (num_aps, 4)
        # Distinct count per row: violation if distinct <= 2
        # distinct <= 2 means: all same OR exactly 2 values
        # all same: c[:,0]==c[:,1] and c[:,1]==c[:,2] and c[:,2]==c[:,3]
        all_same = (c[:,0] == c[:,1]) & (c[:,1] == c[:,2]) & (c[:,2] == c[:,3])
        # exactly 2: not all same, but the set has size 2
        # This means: among the 4 values, there are exactly 2 distinct
        # Equivalently: not all same AND (c[0]==c[1] or c[0]==c[2] or c[0]==c[3] or c[1]==c[2] or c[1]==c[3] or c[2]==c[3])
        # But that's not sufficient. Let me think differently.
        # distinct <= 2 iff the number of pairwise-equal pairs is >= 3 (since with 4 elements, if 2 distinct, min pairs equal is 3: e.g., A,A,B,B has 3 equal pairs; A,A,A,B has 3)
        # Actually with 4 elements and k distinct:
        #   k=1: 6 equal pairs
        #   k=2: if pattern A,A,B,B -> 3 equal pairs (A-A twice? no: positions (0,1),(0,2)? no. Let me just count)
        #   Pattern A,A,B,B: equal pairs: (0,1), (2,3) = 2? No wait, (0,1) both A, (2,3) both B. That's 2. But also... no, only adjacent in value not position.
        #   Actually pairs of indices: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        #   A,A,B,B: (0,1)A=A yes, (0,2)A!=B no, (0,3)A!=B no, (1,2)A!=B no, (1,3)A!=B no, (2,3)B=B yes -> 2 equal pairs
        #   A,A,A,B: (0,1)yes,(0,2)yes,(0,3)no,(1,2)yes,(1,3)no,(2,3)no -> 3 equal pairs
        #   A,B,A,B: (0,1)no,(0,2)yes,(0,3)no,(1,2)no,(1,3)yes,(2,3)no -> 2 equal pairs
        #   A,B,B,A: (0,1)no,(0,2)no,(0,3)yes,(1,2)yes,(1,3)no,(2,3)no -> 2 equal pairs
        #   A,A,B,A: same as A,A,A,B by permutation -> 3
        # So k=2 can have 2 or 3 equal pairs. k=1 has 6.
        # So distinct <= 2 iff equal_pairs >= 2? No, k=3 can have 1 equal pair. k=4 has 0.
        # k=2: min equal pairs is 2 (A,A,B,B or A,B,A,B or A,B,B,A). k=3: max equal pairs is 1 (A,A,B,C).
        # So distinct <= 2 iff equal_pairs >= 2.
        # Let me verify: k=3, e.g., A,A,B,C: equal pairs: (0,1) yes, others no -> 1. Correct.
        # k=2, A,A,B,B: 2 equal pairs. k=2, A,A,A,B: 3. k=2, A,B,A,B: 2.
        # So yes: distinct <= 2 iff number of equal pairs >= 2.
        
        # Compute equal pairs vectorized
        # 6 pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        eq01 = (c[:,0] == c[:,1]).astype(np.int8)
        eq02 = (c[:,0] == c[:,2]).astype(np.int8)
        eq03 = (c[:,0] == c[:,3]).astype(np.int8)
        eq12 = (c[:,1] == c[:,2]).astype(np.int8)
        eq13 = (c[:,1] == c[:,3]).astype(np.int8)
        eq23 = (c[:,2] == c[:,3]).astype(np.int8)
        
        equal_pairs = eq01 + eq02 + eq03 + eq12 + eq13 + eq23
        violations = (equal_pairs >= 2).sum()
        return violations
    
    def is_valid_vec(colors_np):
        return count_violations_vec(colors_np) == 0
    
    # Run the seed greedy
    def run_seed():
        colors = []
        for x in range(1, N + 1):
            found = False
            for color in range(128):
                valid_4 = True
                for d in range(1, (x - 1) // 3 + 1):
                    if x - 1 - 3 * d < 0:
                        continue
                    prev_colors = {colors[x - 1 - j * d] for j in (1, 2, 3)}
                    if len({color} | prev_colors) < 3:
                        valid_4 = False
                        break
                if not valid_4:
                    continue
                colors.append(color)
                found = True
                break
            if not found:
                colors.append(random.randint(0, 127))
        return colors
    
    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    # Try to improve with local search using vectorized violation counting
    for target_k in range(3, best_k + 1):
        if time.time() > DEADLINE - 5:
            break
        
        k_start = time.time()
        k_budget = min(4.0, max(1.0, (DEADLINE - 5 - k_start)))
        
        for restart in range(500):
            if time.time() > k_start + k_budget:
                break
            
            colors = np.array([random.randint(0, target_k - 1) for _ in range(N)], dtype=np.int32)
            
            for step in range(20000):
                if time.time() > k_start + k_budget:
                    break
                
                violations = count_violations_vec(colors)
                if violations == 0:
                    best_colors = colors.tolist()
                    best_k = target_k
                    break
                
                # Pick a random violated AP
                # Find violated APs
                c = colors[aps_arr]
                eq01 = (c[:,0] == c[:,1]).astype(np.int8)
                eq02 = (c[:,0] == c[:,2]).astype(np.int8)
                eq03 = (c[:,0] == c[:,3]).astype(np.int8)
                eq12 = (c[:,1] == c[:,2]).astype(np.int8)
                eq13 = (c[:,1] == c[:,3]).astype(np.int8)
                eq23 = (c[:,2] == c[:,3]).astype(np.int8)
                equal_pairs = eq01 + eq02 + eq03 + eq12 + eq13 + eq23
                violated_mask = equal_pairs >= 2
                violated_indices = np.where(violated_mask)[0]
                
                if len(violated_indices) == 0:
                    break
                
                ap_idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
                ap = aps_arr[ap_idx]
                
                # Try all positions in the AP and all colors, find best move
                best_new_violations = violations
                best_idx = -1
                best_new_c = -1
                
                for pos in range(4):
                    idx = int(ap[pos])
                    old_c = int(colors[idx])
                    for new_c in range(target_k):
                        if new_c == old_c:
                            continue
                        colors[idx] = new_c
                        new_violations = count_violations_vec(colors)
                        if new_violations < best_new_violations:
                            best_new_violations = new_violations
                            best_idx = idx
                            best_new_c = new_c
                        colors[idx] = old_c
                
                if best_idx >= 0:
                    colors[best_idx] = best_new_c
                    violations = best_new_violations
                else:
                    # Random move
                    pos = random.randint(0, 3)
                    idx = int(ap[pos])
                    old_c = int(colors[idx])
                    new_c = random.randint(0, target_k - 1)
                    if new_c != old_c:
                        colors[idx] = new_c
                        new_violations = count_violations_vec(colors)
                        if new_violations <= violations:
                            violations = new_violations
                        else:
                            colors[idx] = old_c
            
            if best_k == target_k:
                break
    
    # Final validation
    best_np = np.array(best_colors, dtype=np.int32)
    if not is_valid_vec(best_np):
        seed_np = np.array(seed_colors, dtype=np.int32)
        if is_valid_vec(seed_np):
            best_colors = seed_colors
        else:
            # Should not happen, but fallback
            best_colors = seed_colors
    
    return best_colors
