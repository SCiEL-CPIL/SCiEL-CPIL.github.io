import numpy as np
import itertools
import time
import sys
import scipy.sparse as sp

D = 11

def generate_w4_vectors():
    vecs = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vecs.append(v)
    return np.array(vecs, dtype=np.int8)

def construct():
    t0 = time.time()
    deadline = t0 + 145
    
    # Try Weight 4
    vectors = generate_w4_vectors()
    N = len(vectors)
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Threshold for w4: dot <= 2
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    conflicts_csr = sp.csr_matrix(conflicts)
    
    # Greedy MIS
    def greedy_mis(conflicts_csr, rng):
        N = conflicts_csr.shape[0]
        alive = np.ones(N, dtype=bool)
        degrees = np.array(conflicts_csr.sum(axis=1)).flatten().astype(np.int32)
        
        selected = []
        while np.any(alive):
            if time.time() > deadline:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            min_deg = np.min(degrees[alive_indices])
            candidates = alive_indices[degrees[alive_indices] == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
            selected.append(chosen)
            
            start = conflicts_csr.indptr[chosen]
            end = conflicts_csr.indptr[chosen + 1]
            neighbors = conflicts_csr.indices[start:end]
            
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            
            for node in to_remove:
                s = conflicts_csr.indptr[node]
                e = conflicts_csr.indptr[node + 1]
                nbrs = conflicts_csr.indices[s:e]
                mask = alive[nbrs]
                if np.any(mask):
                    degrees[nbrs[mask]] -= 1
                    
        return selected

    rng = np.random.default_rng(42)
    best_set = greedy_mis(conflicts_csr, rng)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        k_remove = max(1, len(best_set) // 20)
        remove_indices = rng.choice(len(best_set), size=k_remove, replace=False)
        
        base_set = [v for i, v in enumerate(best_set) if i not in remove_indices]
        base_set_set = set(base_set)
        
        # Candidates: all not in base_set
        all_indices = set(range(N))
        candidates = list(all_indices - base_set_set)
        
        # Filter candidates compatible with base_set
        valid_candidates = []
        for c in candidates:
            s_idx = conflicts_csr.indptr[c]
            e_idx = conflicts_csr.indptr[c + 1]
            nbrs = conflicts_csr.indices[s_idx:e_idx]
            if len(set(nbrs.tolist()) & base_set_set) == 0:
                valid_candidates.append(c)
                
        if not valid_candidates:
            continue
            
        # Random Greedy on valid_candidates
        rng.shuffle(valid_candidates)
        
        current_set = base_set.copy()
        current_set_set = base_set_set.copy()
        
        # To speed up removal, maintain a set of available candidates
        available = set(valid_candidates)
        
        for c in valid_candidates:
            if c not in available:
                continue
            # c is available, check if compatible with current_set
            # (It was compatible with base_set, but might conflict with newly added ones)
            s_idx = conflicts_csr.indptr[c]
            e_idx = conflicts_csr.indptr[c + 1]
            nbrs = conflicts_csr.indices[s_idx:e_idx]
            
            if len(set(nbrs.tolist()) & current_set_set) == 0:
                current_set.append(c)
                current_set_set.add(c)
                
                # Remove c and its neighbors from available
                available.discard(c)
                for n in nbrs:
                    available.discard(n)
                    
        if len(current_set) > len(best_set):
            best_set = current_set
            improved = True

    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
