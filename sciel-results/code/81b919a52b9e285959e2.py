import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate candidate vectors
    candidates = []
    
    # Type A: ±2e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int8)
        v[i] = 2
        candidates.append(v)
        v2 = np.zeros(D, dtype=np.int8)
        v2[i] = -2
        candidates.append(v2)
    
    # Type D: ±1 on 4 coordinates
    supports = list(itertools.combinations(range(D), 4))
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(s, signs):
                v[idx] = sign
            candidates.append(v)
    
    n = len(candidates)
    V = np.array(candidates, dtype=np.int8)
    
    # All norms squared = 4
    # Conflict if dot product > 2
    dots = V @ V.T
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    # adj_bool[i] = True for j if i and j conflict
    adj_bool = conflict  # (n, n) boolean
    
    # Local search with vectorized operations
    def local_search(init_set, max_iter=1000):
        current = np.zeros(n, dtype=bool)
        for v in init_set:
            current[v] = True
        
        best_set = set(init_set)
        best_size = len(init_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute blocked: vertices that conflict with any in current
            if current.any():
                blocked = np.any(adj_bool[current], axis=0)
            else:
                blocked = np.zeros(n, dtype=bool)
            
            # Candidates: not in current and not blocked
            cand_mask = ~(current | blocked)
            cand_indices = np.where(cand_mask)[0]
            
            if len(cand_indices) > 0:
                v_add = int(cand_indices[np.random.randint(len(cand_indices))])
                current[v_add] = True
                cur_size = current.sum()
                if cur_size > best_size:
                    best_size = int(cur_size)
                    best_set = set(np.where(current)[0].tolist())
                continue
            
            # Stuck: try 1-swap
            if current.any():
                cur_indices = np.where(current)[0]
                v_remove = int(cur_indices[np.random.randint(len(cur_indices))])
                current[v_remove] = False
                
                # Recompute blocked
                if current.any():
                    blocked2 = np.any(adj_bool[current], axis=0)
                else:
                    blocked2 = np.zeros(n, dtype=bool)
                
                cand_mask2 = ~(current | blocked2)
                cand_indices2 = np.where(cand_mask2)[0]
                
                if len(cand_indices2) > 0:
                    v_add = int(cand_indices2[np.random.randint(len(cand_indices2))])
                    current[v_add] = True
                    cur_size = current.sum()
                    if cur_size > best_size:
                        best_size = int(cur_size)
                        best_set = set(np.where(current)[0].tolist())
                else:
                    current[v_remove] = True
        
        return best_set
    
    def greedy_init():
        """Randomized greedy initialization."""
        current = np.zeros(n, dtype=bool)
        order = np.random.permutation(n)
        for v in order:
            if not current[v]:
                # Check if v conflicts with any in current
                if not np.any(adj_bool[v] & current):
                    current[v] = True
        return set(np.where(current)[0].tolist())
    
    np.random.seed(42)
    
    best_set = set()
    best_size = 0
    
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        init = greedy_init()
        result = local_search(init, max_iter=800)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    if not best_set:
        return np.array([], dtype=np.int8).reshape(0, D)
    
    final_vectors = V[list(best_set)]
    return final_vectors
