import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s kill, 60s contract
    
    best_score = float('inf')
    best_poly = None
    
    # Helper to count nonzero terms in square
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return int(np.count_nonzero(sq))
    
    # Helper to validate polynomial
    def is_valid(poly):
        if len(poly) != 41:
            return False
        if poly[40] == 0:
            return False
        nz_count = sum(1 for x in poly if x != 0)
        if nz_count != 12:
            return False
        if any(abs(x) > 64 for x in poly):
            return False
        return True
    
    # Generate all possible support sets of size 12 that include index 40
    # But that's C(40, 11) which is too many.
    # Instead, we will use a heuristic search over supports.
    
    # Strategy:
    # 1. Pre-compute a set of "good" supports based on random sampling and local search on support structure.
    # 2. For a fixed support, solve for signs (and possibly small integer values) to minimize nonzero square terms.
    #    Since values are in [-64, 64], we can start with +/- 1 and try to adjust if needed, 
    #    but typically +/- 1 is best for cancellation.
    
    # Let's focus on binary coefficients (+/-1) first, as they are most likely to yield minimal square support.
    # If we find a solution with score S, we can try to improve it by allowing values in {-2, -1, 1, 2} etc.
    
    # Step 1: Find a good support structure.
    # A support is a set of 12 indices including 40.
    # The square of a polynomial with support S has support contained in S+S = {i+j | i,j in S}.
    # We want S+S to be as small as possible.
    
    # Heuristic: Try to find S such that |S+S| is minimized.
    # This is a known problem (additive combinatorics). Sets with small sumset are arithmetic progressions.
    # Let's generate candidate supports that are close to arithmetic progressions.
    
    candidate_supports = set()
    
    # Add arithmetic progressions
    for start in range(-29, 29):
        for step in range(1, 5): # Steps 1, 2, 3, 4
            s = set()
            for i in range(12):
                idx = start + i * step
                if 0 <= idx < 41:
                    s.add(idx)
            if len(s) == 12:
                # Ensure 40 is in s? No, we can just check later or force it.
                # For now, just add valid sets.
                candidate_supports.add(frozenset(s))
                
    # Add random supports
    for _ in range(200):
        s = set(random.sample(range(41), 12))
        candidate_supports.add(frozenset(s))
        
    # Add supports that include 40 specifically
    for _ in range(200):
        rest = random.sample(range(40), 11)
        s = set(rest) | {40}
        candidate_supports.add(frozenset(s))
        
    # Evaluate each support with optimal sign assignment (Greedy/Local Search)
    # For a fixed support S, let x_i be the coefficient. We want to minimize || (sum x_i t^i)^2 ||_0.
    # This is equivalent to minimizing the number of nonzero coefficients in the convolution.
    # With x_i in {-1, 1}, we can use local search on signs.
    
    def solve_signs_for_support(support):
        # support is a frozenset of indices
        indices = sorted(support)
        if len(indices) != 12:
            return None, float('inf')
            
        # Initialize signs randomly
        signs = [1] * 12
        best_signs = signs[:]
        best_score = float('inf')
        
        # Local search on signs
        # We can also try to move indices if the support is bad, but let's stick to fixed support first.
        
        # To speed up, we can precompute the structure of the square.
        # The square coefficients are at indices i+j.
        # There are at most 2*12-1 = 23 unique sums if indices are close? No, up to 40+40=80, but degree is 40, so max index 80?
        # Wait, polynomial length is 41 (degree 40). Square length is 81 (degree 80).
        # The contract says "length<=41" for the input polynomial.
        # The square can be longer. We count nonzero terms in the square.
        
        # Let's just use numpy convolve for speed.
        
        def eval_signs(signs):
            p = [0] * 41
            for idx, sgn in zip(indices, signs):
                p[idx] = sgn
            return count_nz_square(p)
        
        current_score = eval_signs(signs)
        best_score = current_score
        best_signs = signs[:]
        
        # Simple local search
        no_improve = 0
        for _ in range(2000):
            if no_improve > 50:
                break
            idx = random.randint(0, 11)
            signs[idx] *= -1
            new_score = eval_signs(signs)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_signs = signs[:]
                no_improve = 0
            else:
                # Revert
                signs[idx] *= -1
                no_improve += 1
                
        # Reconstruct polynomial
        p = [0] * 41
        for idx, sgn in zip(indices, best_signs):
            p[idx] = sgn
        return p, best_score

    # Iterate through candidate supports
    # To ensure we find something good, we prioritize supports with small sumset size.
    
    def sumset_size(support):
        s = set()
        support_list = list(support)
        for i in support_list:
            for j in support_list:
                s.add(i + j)
        return len(s)
        
    # Sort supports by sumset size
    supports_list = list(candidate_supports)
    # Filter to those containing 40? Not strictly required for the search, but the final poly must have p[40]!=0.
    # If the optimal sign assignment for a support not containing 40 is found, we can't use it directly.
    # So let's only consider supports that contain 40.
    
    valid_supports = [s for s in supports_list if 40 in s]
    
    # If we didn't generate many, add some more
    if len(valid_supports) < 100:
        for _ in range(100):
            rest = random.sample(range(40), 11)
            s = frozenset(set(rest) | {40})
            if s not in valid_supports:
                valid_supports.append(s)
                
    # Sort by sumset size
    valid_supports.sort(key=sumset_size)
    
    # Process top N supports
    # We have ~50s. Solving signs takes ~2000 * eval_time.
    # eval_time is ~10us (numpy convolve on small arrays).
    # So 2000 * 10us = 20ms per support.
    # We can process ~2000 supports in 40s.
    
    for support in valid_supports:
        if time.time() > deadline - 5.0:
            break
            
        p, score = solve_signs_for_support(support)
        if p is None:
            continue
            
        if score < best_score:
            best_score = score
            best_poly = p[:]
            
    # If we found a good one, try to improve it by allowing non-binary values?
    # Or just do a general local search on the best_poly found so far.
    
    if best_poly is None:
        # Fallback: random search
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            p = [0] * 41
            indices = random.sample(range(41), 12)
            if 40 not in indices:
                indices[-1] = 40
            for idx in indices:
                p[idx] = random.choice([-1, 1])
            score = count_nz_square(p)
            if score < best_score:
                best_score = score
                best_poly = p[:]
    else:
        # Refine best_poly with general local search (moving terms, changing values)
        current_poly = best_poly[:]
        current_score = best_score
        T = 1.0
        cooling = 0.999
        no_improve = 0
        
        while time.time() < deadline - 1.0:
            if no_improve > 1000:
                break
                
            neighbor = current_poly[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Change value
                nz_indices = [i for i in range(41) if neighbor[i] != 0]
                idx = random.choice(nz_indices)
                old_val = neighbor[idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if new_val == 0:
                    continue
                if abs(new_val) > 64:
                    continue
                neighbor[idx] = new_val
            else:
                # Move term
                nz_indices = [i for i in range(41) if neighbor[i] != 0]
                z_indices = [i for i in range(41) if neighbor[i] == 0]
                if not z_indices:
                    continue
                src = random.choice(nz_indices)
                if src == 40:
                    continue
                dst = random.choice(z_indices)
                val = neighbor[src]
                neighbor[src] = 0
                neighbor[dst] = val
                
            if not is_valid(neighbor):
                continue
                
            new_score = count_nz_square(neighbor)
            
            if new_score < current_score:
                current_poly = neighbor
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta = new_score - current_score
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_poly = neighbor
                    current_score = new_score
                no_improve += 1
                T *= cooling
                if T < 0.01:
                    T = 0.01

    return best_poly
