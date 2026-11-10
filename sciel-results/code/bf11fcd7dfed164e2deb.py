# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

START_TIME = time.time()
TIME_LIMIT = 590  # Safety margin


def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm
    
    Strategy: Use greedy + local search on weight-4 and weight-5 vector pools.
    For weight-w vectors: constraint is dot(u,v) <= floor(w/2) for all pairs.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    d = 11
    
    def generate_pool(w):
        """Generate all weight-w vectors in d dimensions."""
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)
    
    def greedy_select(pool, max_dot, time_limit, n_restarts=5):
        """
        Greedy selection with random restarts.
        Returns best set of indices found.
        """
        n = len(pool)
        if n == 0:
            return np.array([], dtype=np.int64)
        
        # Precompute dot product matrix if feasible (n <= 8000)
        use_matrix = n <= 8000
        if use_matrix:
            D = pool @ pool.T
        
        best_indices = np.array([], dtype=np.int64)
        rng = np.random.RandomState(42)
        
        for restart in range(n_restarts):
            if time.time() > time_limit:
                break
            
            order = rng.permutation(n)
            selected = []
            
            for idx in order:
                if time.time() > time_limit:
                    break
                
                if len(selected) == 0:
                    selected.append(idx)
                    continue
                
                if use_matrix:
                    dots = D[idx, selected]
                    if np.all(dots <= max_dot):
                        selected.append(idx)
                else:
                    # On-the-fly computation
                    dots = pool[selected] @ pool[idx]
                    if np.all(dots <= max_dot):
                        selected.append(idx)
            
            if len(selected) > len(best_indices):
                best_indices = np.array(selected, dtype=np.int64)
        
        return best_indices
    
    def local_search(indices, pool, max_dot, time_limit, n_rounds=3):
        """
        Try to improve the set by swapping or adding vectors.
        Returns improved indices.
        """
        n = len(pool)
        if len(indices) == 0:
            return indices
        
        use_matrix = n <= 8000
        if use_matrix:
            D = pool @ pool.T
        
        selected_set = set(indices.tolist())
        selected_list = list(indices)
        
        for rnd in range(n_rounds):
            if time.time() > time_limit:
                break
            
            improved = False
            # Try adding vectors not in the set
            unselected = [i for i in range(n) if i not in selected_set]
            rng = np.random.RandomState(42 + rnd)
            rng.shuffle(unselected)
            
            for idx in unselected:
                if time.time() > time_limit:
                    break
                
                if use_matrix:
                    dots = D[idx, selected_list]
                    if np.all(dots <= max_dot):
                        selected_list.append(idx)
                        selected_set.add(idx)
                        improved = True
                else:
                    dots = pool[selected_list] @ pool[idx]
                    if np.all(dots <= max_dot):
                        selected_list.append(idx)
                        selected_set.add(idx)
                        improved = True
            
            # Try swapping: remove one, try to add two
            if not improved and len(selected_list) > 2:
                # Skip complex swaps for now, just return
                pass
        
        return np.array(selected_list, dtype=np.int64)
    
    # Generate pools for different weights
    # w=4: pool=5280, max_dot=2
    # w=5: pool=14784, max_dot=2
    # w=6: pool=29568, max_dot=3
    
    best_result = None
    best_k = 0
    
    # Try weight 4 first (smaller pool, faster)
    pool4 = generate_pool(4)
    idx4 = greedy_select(pool4, 2, deadline, n_restarts=8)
    if len(idx4) > 0:
        idx4 = local_search(idx4, pool4, 2, deadline, n_rounds=2)
        k4 = len(idx4)
        if k4 > best_k:
            best_k = k4
            best_result = pool4[idx4]
    
    # Try weight 5 if time permits
    if time.time() < deadline - 30:
        pool5 = generate_pool(5)
        idx5 = greedy_select(pool5, 2, deadline, n_restarts=5)
        if len(idx5) > 0:
            idx5 = local_search(idx5, pool5, 2, deadline, n_rounds=2)
            k5 = len(idx5)
            if k5 > best_k:
                best_k = k5
                best_result = pool5[idx5]
    
    # Try weight 6 if time permits
    if time.time() < deadline - 30:
        pool6 = generate_pool(6)
        idx6 = greedy_select(pool6, 3, deadline, n_restarts=3)
        if len(idx6) > 0:
            idx6 = local_search(idx6, pool6, 3, deadline, n_rounds=1)
            k6 = len(idx6)
            if k6 > best_k:
                best_k = k6
                best_result = pool6[idx6]
    
    # Fallback: weight 2 vectors (always valid, k=220)
    if best_result is None or best_k < 220:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        best_result = np.array(points, dtype=np.int64)
        best_k = 220
    
    # Final validation and formatting
    best_result = np.around(best_result).astype(np.int64)
    
    # Quick validation
    if best_result.shape[1] != 11 or best_result.shape[0] < 2:
        # Fallback to weight 2
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        best_result = np.array(points, dtype=np.int64)
    
    return best_result


# EVOLVE-BLOCK-END
