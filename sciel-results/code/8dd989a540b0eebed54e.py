"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # 1. Generate Candidates: Weight-3 and Weight-4 vectors
    candidates = []
    
    # Weight 3
    for support in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    # Weight 4
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)
    
    # 2. Build Conflict Graph
    # Conflict if 2 * dot(v_i, v_j) > norm_i * norm_j
    # For weight 3: norm^2 = 3. Condition: 2 dot > 3 => dot >= 2 (since dot is int)
    # For weight 4: norm^2 = 4. Condition: 2 dot > 4 => dot >= 3
    # For mixed (3,4): norm^2 = 3, 4. Condition: 2 dot > sqrt(12) ~ 3.46 => dot >= 2
    
    # We can compute dot products and check conditions based on weights.
    # weights: 0 for w3, 1 for w4?
    # Actually, just compute norms squared.
    norms_sq = np.sum(cand_arr * cand_arr, axis=1)
    
    # G = cand_arr @ cand_arr.T
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Conflict matrix
    # Conflict[i,j] is True if 2*G[i,j] > sqrt(norms_sq[i]*norms_sq[j])
    # To avoid sqrt, square both sides: 4*G[i,j]^2 > norms_sq[i]*norms_sq[j]
    # But G can be negative. Conflict only if G > 0.
    # So: G[i,j] > 0 and 4*G[i,j]^2 > norms_sq[i]*norms_sq[j]
    
    # Vectorized check
    # H = 4 * G**2
    # N_i = norms_sq[:, None]
    # N_j = norms_sq[None, :]
    # Conflict = (G > 0) & (4 * G**2 > N_i * N_j)
    
    # Note: G is int8. G**2 might overflow int8? Max G is 4. 4*16=64. Fits in int8? No, int8 max 127. OK.
    # But norms_sq are 3 or 4. Product is 9, 12, 16.
    # 4*G^2 > 9, 12, 16.
    
    # Let's just use float for safety and simplicity, memory is 6600*6600*8 = 348MB. Too much?
    # 6600*6600*1 (bool) is 43MB.
    # Let's compute the condition carefully.
    
    # G is int8.
    # G_sq = G.astype(np.int16) ** 2
    # G_sq = 4 * G_sq
    # N_prod = norms_sq[:, None].astype(np.int16) * norms_sq[None, :].astype(np.int16)
    # Conflict = (G > 0) & (G_sq > N_prod)
    
    G_sq = (G.astype(np.int16) ** 2) * 4
    N_prod = norms_sq[:, None].astype(np.int16) * norms_sq[None, :].astype(np.int16)
    Conflict = (G > 0) & (G_sq > N_prod)
    
    # Build adjacency lists
    neighbors = [[] for _ in range(n_cands)]
    for i in range(n_cands):
        js = np.where(Conflict[i])[0]
        js = js[js != i]
        neighbors[i] = js.tolist()

    # 3. Greedy Independent Set
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0

    def run_greedy_fast():
        if time.time() > deadline - 5:
            return None
            
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Find min degree among remaining
            min_deg = min(deg[i] for i in remaining)
            
            # Get all nodes with min degree
            candidates_min = [i for i in remaining if deg[i] == min_deg]
            
            # Pick randomly
            u = rng.choice(candidates_min)
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    # Try multiple restarts
    for r in range(100):
        if time.time() > deadline - 10:
            break
        result = run_greedy_fast()
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
