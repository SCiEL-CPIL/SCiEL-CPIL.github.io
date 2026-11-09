import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 78 second deadline with 12s safety margin
    
    random.seed(42)
    
    # Precompute pair information
    pairs_i = []
    pairs_j = []
    pair_sums = []
    for i in range(129):
        for j in range(i, 129):
            pairs_i.append(i)
            pairs_j.append(j)
            pair_sums.append(i + j)
    
    pairs_i = np.array(pairs_i, dtype=np.int32)
    pairs_j = np.array(pairs_j, dtype=np.int32)
    pair_sums = np.array(pair_sums, dtype=np.int32)
    
    # For each target, get indices of pairs that sum to it
    target_to_pair_idx = {t: [] for t in range(128, 257)}
    for idx, s in enumerate(pair_sums):
        if 128 <= s <= 256:
            target_to_pair_idx[s].append(idx)
    
    for t in target_to_pair_idx:
        target_to_pair_idx[t] = np.array(target_to_pair_idx[t], dtype=np.int32)
    
    def get_rep_counts(A_bool):
        counts = np.zeros(129, dtype=np.int32)
        for idx, t in enumerate(range(128, 257)):
            idxs = target_to_pair_idx[t]
            if len(idxs) == 0:
                continue
            active = A_bool[pairs_i[idxs]] & A_bool[pairs_j[idxs]]
            counts[idx] = np.sum(active)
        return counts
    
    # Initial: A = [64, 128]
    A_bool = np.zeros(129, dtype=bool)
    A_bool[64:129] = True
    current_max = int(np.max(get_rep_counts(A_bool)))
    best_A_bool = A_bool.copy()
    best_max = current_max
    
    # Phase 1: Greedy removal
    elements_in_A = list(np.where(A_bool)[0])
    random.shuffle(elements_in_A)
    
    for elem in elements_in_A:
        if time.time() > deadline - 15:
            break
        A_bool[elem] = False
        counts = get_rep_counts(A_bool)
        if np.all(counts > 0):
            new_max = int(np.max(counts))
            if new_max <= current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A_bool = A_bool.copy()
            else:
                A_bool[elem] = True
        else:
            A_bool[elem] = True
    
    # Phase 2: Local search
    for iteration in range(50000):
        if time.time() > deadline - 3:
            break
        
        move_type = random.choice(['remove', 'add', 'swap'])
        
        if move_type == 'remove':
            in_A = np.where(A_bool)[0]
            if len(in_A) <= 15:
                continue
            elem = in_A[random.randint(0, len(in_A)-1)]
            A_bool[elem] = False
        elif move_type == 'add':
            not_in_A = np.where(~A_bool)[0]
            if len(not_in_A) == 0:
                continue
            elem = not_in_A[random.randint(0, len(not_in_A)-1)]
            A_bool[elem] = True
        else:
            in_A = np.where(A_bool)[0]
            not_in_A = np.where(~A_bool)[0]
            if len(in_A) == 0 or len(not_in_A) == 0:
                continue
            elem_remove = in_A[random.randint(0, len(in_A)-1)]
            elem_add = not_in_A[random.randint(0, len(not_in_A)-1)]
            A_bool[elem_remove] = False
            A_bool[elem_add] = True
        
        counts = get_rep_counts(A_bool)
        
        if np.any(counts == 0):
            if move_type == 'remove':
                A_bool[elem] = True
            elif move_type == 'add':
                A_bool[elem] = False
            else:
                A_bool[elem_remove] = True
                A_bool[elem_add] = False
            continue
        
        new_max = int(np.max(counts))
        
        if new_max < current_max or (new_max == current_max and random.random() < 0.05):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_A_bool = A_bool.copy()
        else:
            if move_type == 'remove':
                A_bool[elem] = True
            elif move_type == 'add':
                A_bool[elem] = False
            else:
                A_bool[elem_remove] = True
                A_bool[elem_add] = False
    
    return list(np.where(best_A_bool)[0])
