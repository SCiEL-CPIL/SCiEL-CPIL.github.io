import time
import itertools
import numpy as np

D = 11

def _build_candidates():
    """Build all nonzero integer vectors with norm^2 = 4."""
    cands = []

    # Type 1: weight-4 vectors in {-1, 0, 1}^11 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            cands.append(v)

    # Type 2: weight-1 vectors with entry ±2 (norm^2 = 4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)

    return np.array(cands, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    candidates = _build_candidates()
    n = len(candidates)
    
    # Compute conflict graph: edge if dot > 2
    dots = candidates @ candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    degrees = conflicts.sum(axis=1)
    
    def greedy_mis(seed):
        rng = np.random.default_rng(seed)
        alive = np.ones(n, dtype=bool)
        selected = []
        # Use a copy of degrees to update? 
        # For speed, we just pick min degree from alive.
        # We don't update degrees dynamically in this simple version, 
        # but we can.
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Pick vertex with min degree among alive
            # degrees[alive_indices]
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected.append(chosen)
            
            # Remove chosen and neighbors
            to_remove = np.where(conflicts[chosen] | (np.arange(n) == chosen))[0]
            alive[to_remove] = False
            
        return selected

    # Find best initial solution from a few greedy runs
    best_initial_set = None
    best_initial_size = 0
    for s in range(10):
        if time.time() > deadline - 10: break
        sel = greedy_mis(s)
        if len(sel) > best_initial_size:
            best_initial_size = len(sel)
            best_initial_set = sel
            
    if best_initial_set is None:
        best_initial_set = greedy_mis(0)
        best_initial_size = len(best_initial_set)

    # Simulated Annealing
    in_set = np.zeros(n, dtype=bool)
    for v in best_initial_set:
        in_set[v] = True
        
    current_size = best_initial_size
    best_in_set = in_set.copy()
    best_size_sa = current_size
    
    T = 1.0
    cooling = 0.9995 # Slower cooling to explore more
    
    rng = np.random.default_rng(123)
    
    # Precompute neighbor lists? 
    # conflicts is a boolean matrix. 
    # conflicts[v] is a boolean array.
    
    while time.time() < deadline:
        if T < 0.01:
            T = 0.01
            
        move_type = rng.integers(3)
        
        if move_type == 0: # Add
            candidates_add = np.where(~in_set)[0]
            if len(candidates_add) == 0: continue
            v = candidates_add[rng.integers(len(candidates_add))]
            
            if np.any(conflicts[v] & in_set):
                continue
            
            in_set[v] = True
            current_size += 1
            delta = 1
            accepted = True
            
        elif move_type == 1: # Remove
            candidates_rem = np.where(in_set)[0]
            if len(candidates_rem) == 0: continue
            v = candidates_rem[rng.integers(len(candidates_rem))]
            
            in_set[v] = False
            current_size -= 1
            delta = -1
            if rng.random() < np.exp(delta / T):
                accepted = True
            else:
                in_set[v] = True
                current_size += 1
                accepted = False
                
        else: # Swap
            candidates_in = np.where(in_set)[0]
            candidates_out = np.where(~in_set)[0]
            if len(candidates_in) == 0 or len(candidates_out) == 0: continue
            
            u = candidates_in[rng.integers(len(candidates_in))]
            v = candidates_out[rng.integers(len(candidates_out))]
            
            common = conflicts[v] & in_set
            if np.sum(common) > 1:
                continue
            if np.sum(common) == 1:
                if not common[u]:
                    continue
            
            in_set[u] = False
            in_set[v] = True
            delta = 0
            accepted = True
            
        if accepted:
            if current_size > best_size_sa:
                best_size_sa = current_size
                best_in_set = in_set.copy()
                
        T *= cooling
        
    best_indices = np.where(best_in_set)[0]
    return candidates[best_indices]

# EVOLVE-BLOCK-END
