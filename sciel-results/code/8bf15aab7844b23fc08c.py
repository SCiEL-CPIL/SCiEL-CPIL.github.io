"""
Strategy:
1. Generate all vectors from the D_11 lattice with squared norm 4.
   These are vectors with:
   - Two coordinates being 1 or -1, rest 0 (Type A: 11*10*4/2 * 2^2? No, 11C2 * 4 = 220 * 4 = 880? 
     Wait, D_n vectors have even weight of odd coordinates? No, D_n is even sum of coordinates.
     Standard D_n: integer vectors with even sum of coordinates.
     Norm squared 4 in D_11:
     Case 1: One coord is 2, rest 0. Sum = 2 (even). Count: 11 * 2 = 22.
     Case 2: Two coords are 1, rest 0. Sum = 2 (even). Count: 11C2 * 4 = 55 * 4 = 220.
     Case 3: Two coords are -1, rest 0. Sum = -2 (even). Count: 11C2 * 1 = 55.
     Case 4: One coord 1, one coord -1? Sum = 0 (even). Count: 11*10*1 = 110? No, (1, -1) is distinct from (-1, 1).
             Pairs (i, j) with i!=j. v_i=1, v_j=-1. Sum=0. Count: 11*10 = 110.
             Pairs (i, j) with i!=j. v_i=-1, v_j=1. Sum=0. Count: 11*10 = 110.
     Case 5: Four coords are 1, rest 0. Sum = 4 (even). Count: 11C4 * 1 = 330.
     Case 6: Three coords 1, one coord -1? Sum = 2 (even). 
             Choose 4 coords: 11C4 = 330. Choose which one is -1: 4 ways.
             Count: 330 * 4 = 1320.
     Case 7: Two coords 1, two coords -1? Sum = 0 (even).
             Choose 4 coords: 11C4 = 330. Choose 2 of them to be 1: 4C2 = 6.
             Count: 330 * 6 = 1980.
     
     Let's re-verify the "Inspiration" candidate generation.
     Inspiration 1 generated:
     - Type 1: One ±2 (22 vectors)
     - Type 2: Four ±1's (11C4 * 2^4 = 330 * 16 = 5280 vectors)
     Total = 5302.
     
     Does Inspiration 1 cover ALL D_11 vectors of norm 4?
     D_11 requires sum of coordinates to be even.
     - One ±2: Sum ±2 (even). OK.
     - Four ±1's: Sum is sum of 4 signs. Possible sums: 4, 2, 0, -2, -4. All even. OK.
     
     What about vectors with two ±1's?
     - Two 1's: Sum 2 (even). OK. (220 vectors)
     - Two -1's: Sum -2 (even). OK. (55 vectors)
     - One 1, one -1: Sum 0 (even). OK. (220 vectors)
     
     The Inspiration 1 code *only* generated Type 1 (one ±2) and Type 2 (four ±1's).
     It MISSED the vectors with exactly two non-zero coordinates (±1, ±1).
     
     Let's check if adding the "two ±1" vectors helps.
     Total candidates in D_11 norm 4:
     1. (±2, 0...): 22
     2. (±1, ±1, 0...): 
        - (1, 1): 55 * 1 = 55? No, 11C2 * 1 (signs fixed +,+) = 55.
        - (-1, -1): 55.
        - (1, -1): 11*10 = 110.
        - (-1, 1): 11*10 = 110.
        Total type 2: 330.
     3. (±1, ±1, ±1, ±1, 0...):
        11C4 * 2^4 = 330 * 16 = 5280.
        
     Total = 22 + 330 + 5280 = 5632.
     
     Why did Inspiration 1 skip the 2-nonzero vectors? Maybe they are less useful?
     Or maybe the code in Inspiration 1 was just a heuristic subset?
     The score was 1.9909. The seed score is 1.0.
     1.9909 * (k_seed) = k_new.
     If k_seed was small, k_new is small.
     The kissing number of 11 is known to be 440? No, kissing number of 11 is 440? 
     Actually, kissing number of 10 is 460, 11 is 440? 
     Wait, kissing number of 11 is 440? 
     Let's check known values.
     k(1)=2, k(2)=6, k(3)=12, k(4)=24, k(5)=40, k(6)=72, k(7)=126, k(8)=240, k(9)=460, k(10)=460, k(11)=440?
     Actually, k(11) is 440? No, k(11) is not definitively 440. 
     Bounds for k(11): 440 <= k(11) <= 460? 
     Actually, the Leech lattice gives 196560 in dim 24.
     In dim 11, the best known is 440?
     
     If the target is ~440, and Inspiration 1 got 1.99, it found ~440 vectors?
     If seed found ~220, then 440/220 = 2.0.
     So Inspiration 1 likely found ~440 vectors.
     
     My plan:
     1. Generate ALL valid D_11 vectors with norm 4 (5632 vectors).
     2. Use a fast greedy MIS algorithm.
     3. Use vectorized numpy operations for speed.
     4. Add a simple local search if time permits.
"""

import numpy as np
import itertools
import time

D = 11
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _generate_d11_norm4_candidates():
    """Generate integer vectors in D_11 with squared norm 4.
    Note: We restrict to vectors with squared norm 4 to ensure uniform max norm
    and simplify the compatibility check (dot product <= 2).
    Vectors with squared norm 2 (Type 2) are excluded to avoid mixed-norm
    constraint violations in the greedy selection process.
    """
    vecs = []
    
    # Type 1: One coordinate is ±2, rest 0
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
            
    # Type 3: Four coordinates are non-zero (±1), rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vecs.append(v)
            
    return np.array(vecs, dtype=np.int64)

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN
    
    # 1. Generate Candidates
    candidates = _generate_d11_norm4_candidates()
    N = len(candidates)
    
    # 2. Precompute Dot Product Matrix
    # M[i, j] = dot(candidates[i], candidates[j])
    # We need M[i, j] <= 2 for i != j to be compatible (independent set in conflict graph)
    # Note: M[i, i] = 4
    M = candidates @ candidates.T
    
    # Conflict matrix: True if incompatible (dot > 2)
    conflict = (M > 2).astype(bool)
    np.fill_diagonal(conflict, False)
    
    # Compatibility matrix: True if compatible (dot <= 2)
    compat = (~conflict).astype(bool)
    np.fill_diagonal(compat, False)
    
    # 3. Greedy MIS with multiple strategies
    best_size = 0
    best_set = []
    
    # Strategy A: Order by degree (ascending) - pick vertices with fewest conflicts first
    degrees = np.sum(conflict, axis=1)
    order_deg_asc = np.argsort(degrees)
    
    # Strategy B: Order by degree (descending) - pick vertices with most conflicts first?
    # Usually for MIS, picking low degree first is better for packing.
    # But let's try both.
    order_deg_desc = np.argsort(-degrees)
    
    # Strategy C: Random restarts
    rng = np.random.default_rng(42)
    
    def _greedy_mis(order):
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if selected_mask[idx]:
                continue
            # Check if idx is compatible with all currently selected
            # If selected is empty, it's always compatible
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
            else:
                # Check compatibility with selected vertices
                # compat[idx, selected_mask]
                # We need all True
                if np.all(compat[idx, selected_mask]):
                    selected.append(idx)
                    selected_mask[idx] = True
        return selected
    
    # Run Strategy A
    if time.time() < deadline:
        s = _greedy_mis(order_deg_asc)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Run Strategy B
    if time.time() < deadline:
        s = _greedy_mis(order_deg_desc)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Run Random Restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = rng.permutation(N)
        s = _greedy_mis(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            # If we reach a high number, we can break early or continue to improve
            if best_size >= 440:
                # Keep searching a bit more to see if we can go higher
                pass
                
    # 4. Local Search: Try to add more vectors or swap
    # If we have time, try to improve the best set found.
    # This is a simple 1-swap local search.
    # For each selected vertex v, and each unselected vertex u:
    # If u is compatible with all selected vertices except possibly v,
    # and u is NOT compatible with v, then swapping v with u might allow adding more?
    # Actually, a simple addition search is better if we missed some.
    
    # Let's try a "fill" pass:
    # Iterate through all unselected vertices, if compatible with current set, add it.
    # This is essentially what greedy does, but maybe the order matters.
    
    # More effective local search:
    # 1. Convert best_set to a set for fast lookup? No, numpy boolean mask is faster.
    # 2. Try to remove one vertex and add two? Too complex for Python loop.
    
    # Let's stick to a robust greedy with many restarts. The D_11 lattice is highly symmetric.
    
    if not best_set:
        # Fallback: Return a small valid set
        # Just the 22 vectors of type (±2, 0...) are mutually orthogonal?
        # Dot product of (2,0..) and (-2,0..) is -4. Distance squared = 16. Norm squared = 4.
        # Condition: dist^2 >= max_norm^2. 16 >= 4. Yes.
        # But (2,0..) and (2,0..) are same vector? No, distinct indices.
        # Wait, if I pick (2,0..) and (-2,0..), they are distinct vectors.
        # Dist^2 = (2 - (-2))^2 = 16.
        # Max norm^2 = 4.
        # 16 >= 4. Valid.
        # Can I pick all 22?
        # Any two distinct type-1 vectors have dot product 0 (if different coords) or -4 (if same coord, opposite sign).
        # If same coord, opposite sign: dot = -4. Dist^2 = 16.
        # If diff coords: dot = 0. Dist^2 = 8.
        # 8 >= 4. Valid.
        # So the 22 type-1 vectors form a valid kissing configuration.
        return candidates[:22]
        
    result = candidates[best_set]
    
    # Final Verification
    # Check if the set is valid: min pairwise distance >= max norm
    if len(result) > 0:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        if len(result) >= 2:
            # Compute pairwise squared distances
            # Using broadcasting for clarity, memory is small for k~440, d=11
            dists_sq = np.sum((result[:, None, :] - result[None, :, :])**2, axis=2)
            np.fill_diagonal(dists_sq, 0)
            min_dist_sq = np.min(dists_sq)
            
            if min_dist_sq < max_norm_sq:
                # Fallback to known valid set: The 22 Type 1 vectors
                # They are all norm squared 4.
                # Pairwise dot products are 0 or -4.
                # Min dist squared is 8 (for dot 0) or 16 (for dot -4).
                # 8 >= 4. Valid.
                return candidates[:22].astype(np.int64)
    
    return result.astype(np.int64)

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found {len(res)} vectors")
    # Verify
    if len(res) > 0:
        dots = res @ res.T
        np.fill_diagonal(dots, 0)
        max_dot = np.max(dots)
        print(f"Max dot product: {max_dot} (should be <= 2)")
        norms = np.sum(res**2, axis=1)
        print(f"Norms: {np.unique(norms)} (should be [4])")
