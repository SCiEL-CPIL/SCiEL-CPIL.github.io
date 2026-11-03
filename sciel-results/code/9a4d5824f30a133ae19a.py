import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s margin (10s safety before 150s limit)
    
    # Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Precompute conflict as bitmasks for fast operations
    # conflict_mask[i] is a Python int bitmask of all j that conflict with i
    conflict_mask = [0] * n_supports
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            inter = len(si.intersection(sj))
            if inter >= 3:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute degree
    degree = [bin(cm).count('1') for cm in conflict_mask]
    
    # Also precompute a numpy boolean matrix for the local search
    conflict_np = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        cm = conflict_mask[i]
        # Set bits
        j = 0
        while cm:
            if cm & 1:
                conflict_np[i][j] = True
            cm >>= 1
            j += 1
    
    best_mis = []
    best_size = 0
    
    def greedy_mis(order):
        """Greedy MIS following a given order of nodes."""
        in_mis = [False] * n_supports
        mis = []
        for node in order:
            if in_mis[node]:
                continue
            # Check if node conflicts with any in MIS
            cm = conflict_mask[node]
            # Check if any bit in cm is set where in_mis is True
            conflict_found = False
            j = 0
            while cm:
                if cm & 1:
                    if in_mis[j]:
                        conflict_found = True
                        break
                cm >>= 1
                j += 1
            if not conflict_found:
                in_mis[node] = True
                mis.append(node)
        return mis
    
    def local_search(mis):
        """2-opt local search: try to swap one out, one in to improve."""
        in_mis = [False] * n_supports
        for node in mis:
            in_mis[node] = True
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try removing each node in MIS and adding a non-MIS node
            for i, out_node in enumerate(mis):
                if time.time() > deadline:
                    break
                
                in_mis[out_node] = False
                
                # Find best candidate to add
                best_candidate = -1
                best_score = -1
                
                # Candidates: nodes not in MIS that don't conflict with MIS (after removal)
                cm_out = conflict_mask[out_node]
                
                for cand in range(n_supports):
                    if in_mis[cand]:
                        continue
                    
                    # Check if cand conflicts with any in current MIS (without out_node)
                    cm_cand = conflict_mask[cand]
                    conflict_with_mis = False
                    j = 0
                    while cm_cand:
                        if cm_cand & 1:
                            if in_mis[j]:
                                conflict_with_mis = True
                                break
                        cm_cand >>= 1
                        j += 1
                    
                    if conflict_with_mis:
                        continue
                    
                    # Score: prefer nodes with fewer conflicts overall (more central)
                    score = -degree[cand]
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                
                if best_candidate >= 0:
                    in_mis[best_candidate] = True
                    mis[i] = best_candidate
                    improved = True
                else:
                    in_mis[out_node] = True
        
        return mis
    
    num_restarts = 3000
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random ordering with some structure
        # Mix of random and degree-based ordering
        if r % 3 == 0:
            # Pure random order
            order = list(range(n_supports))
            random.shuffle(order)
        elif r % 3 == 1:
            # Degree-ascending with noise
            scored = [(degree[i] + random.random() * 5, i) for i in range(n_supports)]
            scored.sort()
            order = [x[1] for x in scored]
        else:
            # Random permutation but biased toward low-degree nodes first
            order = list(range(n_supports))
            random.shuffle(order)
            # Partially sort by degree
            for i in range(len(order) - 1):
                if random.random() < 0.5:
                    if degree[order[i]] > degree[order[i+1]]:
                        order[i], order[i+1] = order[i+1], order[i]
        
        mis = greedy_mis(order)
        
        # Local search to improve
        mis = local_search(mis)
        
        if len(mis) > best_size:
            best_size = len(mis)
            best_mis = list(mis)
    
    # Construct vectors
    vectors = []
    for s_idx in best_mis:
        support = supports[s_idx]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
