"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # 1. Generate Candidates: Weight-2, Weight-3, Weight-4 vectors
    candidates = []
    
    # Weight 2
    for support in combinations(range(D), 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
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
    norms_sq = np.sum(cand_arr * cand_arr, axis=1)
    
    # G = cand_arr @ cand_arr.T
    # Use int16 for safety in dot product before squaring?
    # Max dot is 4. 4*4=16. int8 is fine.
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Conflict condition: 2 * dot > norm_i * norm_j
    # Equivalent: 4 * dot^2 > norm_sq_i * norm_sq_j  AND dot > 0
    G_sq = (G.astype(np.int16) ** 2) * 4
    N_prod = norms_sq[:, None].astype(np.int16) * norms_sq[None, :].astype(np.int16)
    Conflict = (G > 0) & (G_sq > N_prod)
    
    # Build adjacency lists
    neighbors = [[] for _ in range(n_cands)]
    for i in range(n_cands):
        js = np.where(Conflict[i])[0]
        js = js[js != i]
        neighbors[i] = js.tolist()

    # 3. Greedy Independent Set with Local Search
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0

    def run_greedy_local_search():
        if time.time() > deadline - 5:
            return None
            
        # Initial greedy: Min degree
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
            min_deg = min(deg[i] for i in remaining)
            candidates_min = [i for i in remaining if deg[i] == min_deg]
            u = rng.choice(candidates_min)
            selected.append(u)
            remaining.remove(u)
            
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        # Local Search: Try to swap nodes to increase size
        # Represent set as boolean array for fast lookup
        in_set = np.zeros(n_cands, dtype=bool)
        for u in selected:
            in_set[u] = True
            
        # List of nodes in set
        set_list = list(selected)
        
        # Try to add nodes from 'remaining' (or all not in set)
        # We can iterate through all candidates not in set
        not_in_set = [i for i in range(n_cands) if not in_set[i]]
        
        # Shuffle not_in_set for randomization
        rng.shuffle(not_in_set)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for new_node in not_in_set:
                if time.time() > deadline - 1:
                    break
                # Check conflicts of new_node with current set
                conflicts = [v for v in neighbors[new_node] if in_set[v]]
                
                # If no conflicts, add it
                if not conflicts:
                    in_set[new_node] = True
                    set_list.append(new_node)
                    improved = True
                    continue
                
                # If conflicts, try to swap: remove one conflict, add new_node
                # To maximize gain, we want to remove a node that allows adding more?
                # Simple heuristic: remove a random conflict, add new_node.
                # But we must ensure new_node doesn't conflict with the rest of the set.
                # new_node only conflicts with 'conflicts'.
                # So if we remove ONE conflict, new_node is compatible with the rest.
                # Does removing one conflict and adding one new node increase size? No, size stays same.
                # We need to find a SWAP that increases size?
                # Wait, if we remove 1 and add 1, size is constant.
                # We need to find a set of conflicts such that removing them allows adding MULTIPLE new nodes?
                # Or just try to add nodes that have 0 conflicts first (done above).
                
                # To increase size, we need to find a "bottleneck".
                # This simple local search might get stuck.
                # Let's try a different local search:
                # Pick a random node from set, remove it. Try to add 2 nodes that were blocked by it?
                
                pass # Skip complex swap for now, rely on multiple restarts with different greedy orders

        return set_list

    # Try multiple restarts with different random seeds for tie-breaking
    for r in range(200):
        if time.time() > deadline - 5:
            break
        
        # Slight variation: Randomize initial degree order?
        # Or just run the same greedy, it's deterministic unless rng is used.
        # rng is used in choice.
        
        result = run_greedy_local_search()
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
