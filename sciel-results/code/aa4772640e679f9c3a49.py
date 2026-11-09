import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 60s limit (evaluator kills at 90s, but let's be safe)
    
    best_score = float('inf')
    best_poly = None
    
    # Helper to count nonzeros in square using NumPy
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return np.count_nonzero(sq)

    # Strategy: Borwein-Erdelyi style structural search.
    # Represent P(x) = sum c_i x^i.
    # We want to minimize nonzero terms in P(x)^2.
    # This is equivalent to finding a set of 12 indices I and coefficients C (non-zero)
    # such that the number of distinct sums i+j (where i,j in I) with non-zero resulting coefficient is minimized.
    # Note: If i+j = k+l and c_i c_j + c_k c_l = 0, the term vanishes.
    # To maximize cancellations, we can look for structures where the index set I has many collisions (i+j = k+l).
    # The maximum number of collisions occurs when I is an arithmetic progression (AP).
    # Let I = {a, a+d, ..., a+11d}.
    # The sums are 2a, 2a+d, ..., 2a+22d.
    # Total 23 distinct sums.
    # Can we make some of these coefficients zero?
    # Coeff of x^(2a+kd) is sum_{i=0}^{11} c_i c_{k-i} (with valid indices).
    # For an AP with 12 terms, the number of non-zero terms in the square is at least 23?
    # Actually, if we choose coefficients carefully, we can cancel some.
    # Example: P(x) = 1 - x^d + x^{2d} - ... + x^{11d} (geometric series).
    # P(x)^2 = (1-x^{12d})^2 / (1-x^d)^2. This doesn't directly help with sparse square in standard basis.
    
    # Let's try specific small structures.
    # 1. Pure AP with alternating signs: 1, -1, 1, -1 ...
    # 2. Symmetric structures.
    
    # We will perform a focused search over:
    # 1. Arithmetic Progressions (AP) with various steps and offsets, and coefficient patterns.
    # 2. Random sets of indices that are "dense" (many collisions) - e.g. subsets of APs.
    # 3. Local search on promising starts.
    
    # Step 1: Generate candidate index sets
    # Since degree <= 40, max index is 40.
    # 12 terms.
    
    candidates = []
    
    # Pattern A: Arithmetic Progression
    # indices: a, a+d, ..., a+11d <= 40
    # d must be small.
    # If d=1, a=0..28.
    # If d=2, a=0..18.
    # If d=3, a=0..7.
    # If d=4, a=0..2.
    # If d=5, a=0.
    for d in range(1, 6):
        max_a = 40 - 11*d
        for a in range(0, max_a + 1):
            indices = [a + i*d for i in range(12)]
            # Try different coefficient patterns for these indices
            # 1. All 1s
            # 2. Alternating signs
            # 3. Random small integers
            
            # All 1s
            p = [0]*41
            for idx in indices: p[idx] = 1
            candidates.append(p)
            
            # Alternating
            p = [0]*41
            for i, idx in enumerate(indices):
                p[idx] = 1 if i % 2 == 0 else -1
            candidates.append(p)
            
            # 1, -1, 1, 1, -1, -1 ... (period 4)
            p = [0]*41
            for i, idx in enumerate(indices):
                p[idx] = 1 if i % 4 < 2 else -1
            candidates.append(p)
            
            # Random coefficients from {-1, 1}
            if random.random() < 0.1: # Only for some to save time
                p = [0]*41
                for idx in indices:
                    p[idx] = random.choice([-1, 1])
                candidates.append(p)
                
            # Random coefficients from {-2, -1, 1, 2}
            if random.random() < 0.1:
                p = [0]*41
                for idx in indices:
                    p[idx] = random.choice([-2, -1, 1, 2])
                candidates.append(p)

    # Pattern B: Two interleaved APs? Or other structured sets.
    # Let's add some random "clumped" sets.
    for _ in range(100):
        # Pick a base and spread
        base = random.randint(0, 20)
        # Choose 12 indices in a small window
        window = random.randint(10, 20)
        max_idx = min(40, base + window)
        # Ensure we can pick 12 distinct indices
        if max_idx - base + 1 >= 12:
            indices = random.sample(range(base, max_idx + 1), 12)
            p = [0]*41
            for idx in indices:
                p[idx] = random.choice([-1, 1])
            candidates.append(p)

    # Ensure last coeff non-zero and valid for all candidates
    valid_candidates = []
    for p in candidates:
        nz_count = sum(1 for x in p if x != 0)
        if nz_count != 12: continue
        if p[-1] == 0:
            nz_indices = [i for i in range(41) if p[i] != 0]
            if nz_indices:
                k = random.choice(nz_indices)
                p[40] = p[k]
                p[k] = 0
            else:
                p[40] = 1
        if any(abs(x) > 64 for x in p): continue
        valid_candidates.append(p)
        
    # Evaluate initial candidates
    for p in valid_candidates:
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_poly = p[:]
            
    # If we haven't found a good one, default to seed
    if best_poly is None:
        best_poly = [0]*41
        for i in range(12): best_poly[i] = 1
        best_score = count_nz_square(best_poly)

    # Step 2: Local Search (Simulated Annealing style)
    # Start from best_poly
    current_poly = best_poly[:]
    current_score = best_score
    
    # Temperature for SA
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart or cool down
            if T > T_min:
                T = max(T_min, T * cooling_rate)
            else:
                # Random restart
                # Pick a random valid candidate from valid_candidates
                if valid_candidates:
                    current_poly = random.choice(valid_candidates).copy()
                else:
                    current_poly = best_poly.copy()
                current_score = count_nz_square(current_poly)
                no_improve_count = 0
                T = 5.0
            continue

        # Generate neighbor
        # Move types:
        # 1. Change coefficient of a non-zero term
        # 2. Move a term to a new index
        # 3. Swap two indices (if both non-zero, effectively same as move? No, just move one)
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Change coefficient
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            
            # Perturb coefficient
            # Try small changes first
            if random.random() < 0.5:
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
            else:
                new_val = random.choice([-2, -1, 1, 2])
            
            if new_val == 0:
                new_val = old_val + random.choice([-1, 1])
                if new_val == 0: new_val = 1
            if abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
            
            current_poly[idx] = new_val
            new_score = count_nz_square(current_poly)
            
            # Accept if better, or with probability exp(-(new_score - current_score)/T)
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                    # Do not update best
                else:
                    current_poly[idx] = old_val
                    no_improve_count += 1
        else:
            # Move index
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            z_indices = [i for i in range(41) if current_poly[i] == 0]
            if not z_indices:
                no_improve_count += 1
                continue
                
            src_idx = random.choice(nz_indices)
            dst_idx = random.choice(z_indices)
            
            val = current_poly[src_idx]
            current_poly[src_idx] = 0
            current_poly[dst_idx] = val
            
            if src_idx == 40:
                # Revert
                current_poly[dst_idx] = 0
                current_poly[src_idx] = val
                no_improve_count += 1
                continue

            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                else:
                    current_poly[dst_idx] = 0
                    current_poly[src_idx] = val
                    no_improve_count += 1

    return best_poly
