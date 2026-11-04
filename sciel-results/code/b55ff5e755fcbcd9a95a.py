"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Type A vectors: ±2e_i for i in 0..10 (22 vectors, norm 2)
    # Type B vectors: ±2e_i ± 2e_j for i<j (44 vectors, norm 4)
    # Type D vectors: weight-4 supports with all 16 sign patterns (norm 2)
    # D3 vectors: 12 vectors from D3 lattice component (norm 2)
    
    # Generate all weight-4 supports
    all_supports = list(combinations(range(D), 4))
    N_sup = len(all_supports)
    
    # Convert supports to bitmasks
    masks = []
    for s in all_supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
    
    # Precompute all pairs (i,j) for Type B compatibility check
    all_pairs = list(combinations(range(D), 2))
    
    # A support S is compatible with Type B vector ±2e_i ± 2e_j iff |S ∩ {i,j}| <= 1
    # i.e., S doesn't contain both i and j
    # So a support is "B-compatible" if it doesn't contain any pair (i,j)
    # Wait - every pair of elements in a 4-element set... no.
    # A support S of size 4 contains C(4,2)=6 pairs. 
    # For Type B vector on pair {i,j}, we need i,j NOT both in S.
    # So S is compatible with Type B(i,j) iff not (i in S and j in S).
    
    # For the full construction, we need:
    # - Type D support S must be compatible with ALL Type B vectors
    #   This means for every pair {i,j}, NOT both i,j in S.
    #   But S has 4 elements, so it contains 6 pairs. 
    #   This is IMPOSSIBLE - any 4-element set contains pairs.
    
    # Wait, I'm confusing myself. Let me re-read the constraint.
    # The distance constraint is: for any two vectors u, v in the set:
    #   ||u - v|| >= max_norm
    # where max_norm is the maximum norm over all vectors.
    
    # If max_norm = 4 (from Type B vectors), then we need ||u-v|| >= 4 for all pairs.
    
    # For Type A (norm 2) and Type D (norm 2):
    #   ||u_A - u_D||^2 = ||u_A||^2 + ||u_D||^2 - 2<u_A, u_D> = 4 + 4 - 2<u_A, u_D>
    #   Need >= 16, so <u_A, u_D> <= -4
    #   u_A = ±2e_i, u_D has entries ±1 on support S
    #   <u_A, u_D> = ±2 * u_D[i] if i in S, else 0
    #   So |<u_A, u_D>| <= 2 if i in S, 0 if not.
    #   <u_A, u_D> <= -4 is IMPOSSIBLE since max absolute value is 2.
    
    # Hmm, this means Type A and Type D can't coexist if max_norm = 4.
    
    # Let me reconsider. Maybe we don't include Type B vectors, or we use a different scaling.
    
    # Actually, let me reconsider the whole approach. The task says:
    # "minimum pairwise distance is at least the maximum vector norm"
    
    # Let me think about what configurations are known for kissing number in dimension 11.
    # The kissing number in dimension 11 is known to be at least 252 and at most 342.
    
    # The current program gets score 2.4727, which with 16 vectors per support gives about 16*155 = 2480? No...
    # Score = k/seed. If seed is some baseline, and score is 2.4727...
    
    # Let me just focus on maximizing the number of vectors.
    
    # Strategy: Use only Type D vectors (weight 4, norm 2) with max_norm = 2.
    # Constraint: ||u - v|| >= 2 for all pairs.
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 + 4 - 2<u,v> = 8 - 2<u,v>
    # Need 8 - 2<u,v> >= 4, so <u,v> <= 2.
    
    # For two weight-4 vectors u, v with supports S_u, S_v:
    # <u,v> = sum over i in S_u ∩ S_v of u[i]*v[i]
    # Each term is ±1, so |<u,v>| <= |S_u ∩ S_v|
    # We need <u,v> <= 2 for all pairs.
    
    # If |S_u ∩ S_v| >= 3, we need to ensure <u,v> <= 2.
    # With 3 common positions, <u,v> can be -3,-1,1,3. We need it <= 2, so we can't have <u,v> = 3.
    # <u,v> = 3 means all three signs match.
    
    # This is complex. Let me use the simpler constraint: |S_u ∩ S_v| <= 2 guarantees <u,v> <= 2.
    # This is what the current program does.
    
    # But we can do better! If |S_u ∩ S_v| = 3, we just need to avoid the case where all 3 signs match.
    # This allows more supports to coexist.
    
    # Let me implement a more careful approach:
    # 1. Find a large set of weight-4 supports with |S_i ∩ S_j| <= 2 (guaranteed compatible)
    # 2. For supports with |S_i ∩ S_j| = 3, we can still include them if we choose signs carefully
    
    # Actually, for a given pair of supports with intersection of size 3, 
    # there are 2^3 = 8 sign combinations on the intersection for each vector.
    # The dot product on the intersection ranges from -3 to 3.
    # We need the total dot product <= 2.
    # If intersection is size 3 and no other overlap, dot product = sum of 3 ±1 terms.
    # Values: -3, -1, 1, 3. We need <= 2, so we exclude the case where all match (dot = 3).
    
    # This means for each pair of supports with intersection 3, we need to avoid one of the 16*16=256 sign combinations.
    # This is a constraint satisfaction problem.
    
    # For simplicity and robustness, let me stick with |S_i ∩ S_j| <= 2.
    
    # Precompute incompatibility: supports are incompatible if |S_i ∩ S_j| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.int8)
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            inter = masks[i] & masks[j]
            if bin(inter).count('1') >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Precompute neighbors
    neighbors = [np.where(incompat[i] == 1)[0] for i in range(N_sup)]
    
    # Greedy MIS with randomized tie-breaking
    def greedy_min_degree(available, rng):
        selected = []
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = sum(1 for v in neighbors[u] if v in avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            u = rng.choice(candidates)
            selected.append(u)
            
            to_remove = {u}
            for v in neighbors[u]:
                if v in avail:
                    to_remove.add(v)
            avail -= to_remove
            
        return selected
    
    # Local search: 1-2 swap to improve
    def local_search(selected, rng):
        selected_set = set(selected)
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 15.0:
            improved = False
            iterations += 1
            if iterations > 100:
                break
                
            selected_list = list(selected_set)
            rng.shuffle(selected_list)
            
            for v in selected_list:
                if time.time() > deadline - 15.0:
                    break
                
                # Find nodes compatible with selected \ {v}
                potential_adds = []
                for x in range(N_sup):
                    if x in selected_set:
                        continue
                    compatible = True
                    for y in selected_set:
                        if y == v:
                            continue
                        if incompat[x, y]:
                            compatible = False
                            break
                    if compatible:
                        potential_adds.append(x)
                        
                if len(potential_adds) >= 2:
                    found = False
                    for i in range(len(potential_adds)):
                        u1 = potential_adds[i]
                        for j in range(i + 1, len(potential_adds)):
                            u2 = potential_adds[j]
                            if not incompat[u1, u2]:
                                selected_set.remove(v)
                                selected_set.add(u1)
                                selected_set.add(u2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                if improved:
                    break
                    
        return list(selected_set)
    
    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different seeds
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        rng_r = np.random.default_rng(42 + r * 1000)
        avail = set(range(N_sup))
        
        iset = greedy_min_degree(avail, rng_r)
        iset = local_search(iset, rng_r)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
    
    if best_iset is None:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 2
        return v.reshape(1, -1)
    
    # Generate Type D vectors: 16 sign patterns per support
    vectors = []
    for i in best_iset:
        sup = all_supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    # Add Type A vectors: ±2e_i (22 vectors, norm 2)
    # Check compatibility with Type D vectors
    # Type A: u = ±2e_i, norm 2
    # Type D: v with support S, norm 2
    # ||u - v||^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>
    # <u,v> = ±2*v[i] if i in S, else 0
    # If i in S: <u,v> = ±2*(±1) = ±2, so ||u-v||^2 = 8-2*(±2) = 4 or 12
    #   We need ||u-v|| >= 2, i.e., ||u-v||^2 >= 4. Both 4 and 12 satisfy this.
    # If i not in S: <u,v> = 0, ||u-v||^2 = 8 >= 4. OK.
    # So Type A vectors are ALWAYS compatible with Type D vectors!
    
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            vectors.append(v)
    
    if not vectors:
        return np.array([[]], dtype=np.int64)

    result = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint
    # M = result @ result.T
    # norms_sq = np.diag(M)
    # max_norm = np.sqrt(np.max(norms_sq))
    # Fill diagonal with 0 for distance check
    # M_off = M.copy()
    # np.fill_diagonal(M_off, 0)
    # dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * M_off
    # np.fill_diagonal(dist_sq, np.inf)
    # assert np.all(dist_sq >= max_norm**2 - 1e-9)
    
    return result

# EVOLVE-BLOCK-END
