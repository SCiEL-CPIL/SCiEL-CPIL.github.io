"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Generate candidates of various weights
    # Weight w: C(11,w) * 2^w vectors, each with norm^2 = w
    # Conflict condition for two vectors with norms^2 = w1, w2:
    #   ||v-w||^2 >= max(w1,w2)
    #   w1 + w2 - 2*v.dot(w) >= max(w1,w2)
    #   2*v.dot(w) <= min(w1,w2)
    
    candidates = []
    norms_sq = []  # norm^2 for each candidate
    
    # Weight 2: C(11,2)*4 = 55*4 = 220 vectors, norm^2=2
    for support in combinations(range(D), 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(2)
    
    # Weight 3: C(11,3)*8 = 165*8 = 1320 vectors, norm^2=3
    for support in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(3)
    
    # Weight 4: C(11,4)*16 = 330*16 = 5280 vectors, norm^2=4
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(4)
    
    # Weight 5: C(11,5)*32 = 462*32 = 14784 vectors, norm^2=5
    # This might be too many, let's include but be careful
    for support in combinations(range(D), 5):
        for signs in product([1, -1], repeat=5):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(5)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)
    norms_sq_arr = np.array(norms_sq, dtype=np.int64)
    
    # Build conflict graph
    # Conflict if 2*dot > min(norm_sq_i, norm_sq_j)
    # G = cand_arr @ cand_arr.T gives dot products
    # For pair (i,j): conflict if 2*G[i,j] > min(norms_sq[i], norms_sq[j])
    
    # Compute dot product matrix
    G = (cand_arr.astype(np.int32) @ cand_arr.T.astype(np.int32))
    
    # Build adjacency lists
    neighbors = [[] for _ in range(n_cands)]
    for i in range(n_cands):
        # Find j such that 2*G[i,j] > min(norms_sq[i], norms_sq[j]) and j != i
        min_norms = np.minimum(norms_sq_arr[i], norms_sq_arr)
        conflict_mask = (2 * G[i] > min_norms)
        js = np.where(conflict_mask)[0]
        js = js[js != i]
        neighbors[i] = js.tolist()
    
    # Greedy independent set with heap and randomization
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        local_rng = np.random.default_rng(42 + seed_offset)
        
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        # Heap of (degree, random_tiebreak, node)
        # Using random tiebreak to handle equal degrees
        heap = []
        for i in range(n_cands):
            tiebreak = local_rng.random()
            heap.append((deg[i], tiebreak, i))
        heapq.heapify(heap)
        
        # Track last pushed degree for staleness check
        last_deg_in_heap = np.full(n_cands, -1, dtype=np.int32)
        for i in range(n_cands):
            last_deg_in_heap[i] = deg[i]
        
        selected = []
        
        while heap:
            if time.time() > deadline - 2:
                break
                
            d, tb, u = heapq.heappop(heap)
            
            # Check if node is still active and degree is fresh
            if not active[u]:
                continue
            if d != deg[u]:
                continue  # Stale entry
            
            selected.append(u)
            active[u] = False
            
            # Remove neighbors
            for v in neighbors[u]:
                if active[v]:
                    active[v] = False
                    for w in neighbors[v]:
                        if active[w]:
                            deg[w] -= 1
                            # Push updated degree
                            tiebreak = local_rng.random()
                            heapq.heappush(heap, (deg[w], tiebreak, w))
        
        return selected
    
    # Try multiple restarts
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        result = run_greedy(r)
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
    
    # Local search: try to add vectors that are compatible with the current set
    if best_size > 0 and time.time() < deadline - 15:
        current_set = set(np.arange(n_cands)[np.isin(np.arange(n_cands), best_set_indices)] if len(best_set) > 0 else [])
        # Actually, let's work with the actual vectors
        current_vectors = best_set
        current_norms_sq = (current_vectors ** 2).sum(axis=1)
        max_norm_sq = current_norms_sq.max()
        
        # Try to add candidates not in the set
        added = []
        for i in range(n_cands):
            if time.time() > deadline - 10:
                break
            v = cand_arr[i]
            v_norm_sq = norms_sq_arr[i]
            
            # Check if v is compatible with all current vectors
            compatible = True
            for j, cv in enumerate(current_vectors):
                cv_norm_sq = current_norms_sq[j]
                dot = int(np.dot(v, cv))
                min_norm_sq = min(v_norm_sq, cv_norm_sq)
                if 2 * dot > min_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                # Also check with newly added vectors
                ok = True
                for av in added:
                    av_norm_sq = (av ** 2).sum()
                    dot = int(np.dot(v, av))
                    min_norm_sq = min(v_norm_sq, av_norm_sq)
                    if 2 * dot > min_norm_sq:
                        ok = False
                        break
                if ok:
                    added.append(v)
        
        if added:
            new_set = np.vstack([current_vectors, np.array(added)])
            # Verify
            new_norms_sq = (new_set ** 2).sum(axis=1)
            new_max_norm_sq = new_norms_sq.max()
            valid = True
            for i in range(len(new_set)):
                for j in range(i+1, len(new_set)):
                    dist_sq = int(np.dot(new_set[i] - new_set[j], new_set[i] - new_set[j]))
                    if dist_sq < new_max_norm_sq:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_set = new_set
                best_size = len(new_set)
    
    # Fallback
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
