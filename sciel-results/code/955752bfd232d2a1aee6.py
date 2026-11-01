import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Generate candidate vectors
    # We focus on vectors with small norms to maximize density
    # Weight 2: norm sqrt(2)
    # Weight 3: norm sqrt(3)
    # Weight 4: norm sqrt(4)=2 (e.g. 1,1,1,1) or sqrt(5) (1,2,0..)
    
    candidates = []
    seen = set()
    
    # Add Weight 2 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                key = tuple(v)
                if key not in seen:
                    seen.add(key)
                    candidates.append(np.array(v, dtype=np.int64))
    
    # Add Weight 3 vectors
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    key = tuple(v)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(np.array(v, dtype=np.int64))
                        
    # Add Weight 4 vectors (1,1,1,1) type
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        key = tuple(v)
                        if key not in seen:
                            seen.add(key)
                            candidates.append(np.array(v, dtype=np.int64))

    # Initial state: All Weight 2 vectors (the seed)
    # Identify which candidates are weight 2
    w2_indices = [i for i, v in enumerate(candidates) if np.sum(v**2) == 2]
    
    current_set = set(w2_indices)
    
    # Precompute norms
    norms = np.array([np.sum(v**2) for v in candidates])
    
    def is_valid(indices):
        if not indices:
            return True
        idx_list = list(indices)
        # Max norm
        max_norm_sq = max(norms[i] for i in idx_list)
        # Min dist
        min_dist_sq = float('inf')
        for i in range(len(idx_list)):
            vi = candidates[idx_list[i]]
            for j in range(i+1, len(idx_list)):
                vj = candidates[idx_list[j]]
                dist_sq = np.sum((vi - vj)**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        return min_dist_sq >= max_norm_sq

    # Initial check
    if not is_valid(current_set):
        return np.array([candidates[i] for i in current_set], dtype=np.int64)

    best_set = set(current_set)
    best_count = len(current_set)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_set = set(current_set)
        
        if move_type == 'add':
            # Pick a random candidate not in set
            avail = [i for i in range(len(candidates)) if i not in current_set]
            if not avail:
                continue
            idx = random.choice(avail)
            new_set.add(idx)
        elif move_type == 'remove':
            if not current_set:
                continue
            idx = random.choice(list(current_set))
            new_set.remove(idx)
        elif move_type == 'swap':
            if not current_set:
                continue
            avail = [i for i in range(len(candidates)) if i not in current_set]
            if not avail:
                continue
            idx_in = random.choice(list(current_set))
            idx_out = random.choice(avail)
            new_set.remove(idx_in)
            new_set.add(idx_out)
            
        # Check validity
        if is_valid(new_set):
            # Accept if valid (greedy) or with probability
            # Since we want max count, we always accept valid moves that increase count
            # If count is same, we might accept to escape local optima
            if len(new_set) > len(current_set):
                current_set = new_set
                if len(current_set) > best_count:
                    best_set = set(current_set)
                    best_count = len(current_set)
            else:
                # Same or smaller count, but valid
                # Accept with probability exp(-delta/T) ? 
                # Here delta is 0 or negative count change.
                # We prefer larger sets.
                # If count is same, accept with prob 0.5 to explore
                if random.random() < 0.5:
                    current_set = new_set
        else:
            # Invalid move, reject
            pass
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return np.array([candidates[i] for i in best_set], dtype=np.int64)
