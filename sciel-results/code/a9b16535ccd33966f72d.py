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

    # 1. Generate Candidates: Weight-4 vectors
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # 2. Build Conflict Graph
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    neighbors = [[] for _ in range(n_cands)]
    for i in range(n_cands):
        js = np.where(G[i] > 2)[0]
        js = js[js != i]
        neighbors[i] = js.tolist()
        
    init_deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)

    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy_heap_random(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        local_rng = np.random.default_rng(42 + seed_offset)
        deg = init_deg.copy()
        removed = np.zeros(n_cands, dtype=bool)
        selected = []
        
        tie_breakers = local_rng.random(n_cands)
        
        heap = [(deg[i], tie_breakers[i], i) for i in range(n_cands)]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 2:
                break
                
            d, tb, u = heapq.heappop(heap)
            
            if removed[u] or deg[u] != d:
                continue
            
            selected.append(u)
            removed[u] = True
            
            for v in neighbors[u]:
                if not removed[v]:
                    deg[v] -= 1
                    heapq.heappush(heap, (deg[v], tie_breakers[v], v))
                    
        return selected

    # Run multiple restarts
    n_restarts = 100
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        result = run_greedy_heap_random(r)
        
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()

    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
