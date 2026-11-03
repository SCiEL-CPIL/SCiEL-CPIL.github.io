import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Generate candidate vectors with squared norm 5
    # Type: One entry +/-2, one entry +/-1, rest 0.
    # Count: 11 * 10 * 2 * 2 = 440 vectors.
    
    candidates = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for s1 in [2, -2]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    V = np.array(candidates, dtype=np.int8)
    N = len(V)
    
    # Compute Conflict Matrix
    # Max norm squared is 5. Max norm is sqrt(5).
    # Min distance squared must be >= 5.
    # ||u-v||^2 = 5 + 5 - 2 u.v = 10 - 2 u.v >= 5
    # 5 >= 2 u.v => u.v <= 2.5
    # Since u.v is integer, u.v <= 2.
    # Conflict if u.v > 2.
    
    M = V @ V.T
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Greedy Maximum Independent Set
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        
        selected = []
        
        while active.any():
            if time.time() > deadline - 5:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            neighbors_mask[idx] = True
            
            removed_indices = np.where(neighbors_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg -= removal_counts
                
            active[removed_indices] = False
            
        return selected

    best_size = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        selected = run_greedy(i)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    if best_size > 0:
        final_vectors = V[best_set]
    else:
        # Fallback: just return one vector or empty? 
        # Better to return the 220 norm sqrt(2) vectors if this fails?
        # But this shouldn't fail.
        final_vectors = V[:1]
        
    return final_vectors

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found {len(res)} vectors")
