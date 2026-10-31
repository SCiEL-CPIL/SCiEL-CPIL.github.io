# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    import random
    import math
    import numpy as np

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def _score_set(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        if len(U_set) != len(U):
            return 0.0
        if min(U_set) < 0:
            return 0.0
            
        max_u = max(U_set)
        if max_u == 0:
            return 1.0
            
        # Use NumPy for vectorized computation if size is reasonable
        # For very small sizes, pure python might be faster due to overhead, but let's try numpy
        U_arr = np.array(sorted(U_set), dtype=np.int64)
        
        # Calculate sums and diffs
        # Broadcasting: U_arr[:, None] + U_arr[None, :]
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        size_sums = len(np.unique(sums))
        size_diffs = len(np.unique(diffs))
        
        if size_sums == 0:
            return 0.0
            
        ratio = size_diffs / size_sums
        denom_arg = 2 * max_u + 1
        if denom_arg <= 1:
            return 1.0
            
        try:
            score = 1.0 + math.log(ratio) / math.log(denom_arg)
        except Exception:
            return 0.0
            
        return score

    # Known optimal Golomb rulers for small n
    # These provide high difference set sizes
    golomb_rulers = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 9, 11, 17],
        7: [0, 1, 4, 9, 11, 17, 22],
        8: [0, 1, 4, 9, 11, 17, 22, 30],
        9: [0, 1, 4, 9, 11, 17, 22, 30, 40],
        10: [0, 1, 4, 9, 11, 17, 22, 30, 40, 52],
        11: [0, 1, 4, 9, 11, 17, 22, 30, 40, 52, 64],
        12: [0, 1, 4, 9, 11, 17, 22, 30, 40, 52, 64, 77]
    }

    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    if time.time() > deadline - 5:
        return best_set

    def improve_set(U, max_iters=200):
        U = sorted(set(U))
        current_score = _score_set(U)
        max_val = max(U) * 2 + 10 # Dynamic max val for search
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.choice(['add', 'remove', 'shift'])
            
            if move_type == 'add' and len(U) < 15:
                # Add a random number, biased towards larger values to increase span?
                # Or just uniform in range
                new_val = random.randint(1, max_val)
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
            elif move_type == 'remove' and len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_U = U[:idx] + U[idx+1:]
            elif move_type == 'shift' and len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Larger shifts to explore space
                delta = random.randint(-5, 5)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in U and new_val != old_val:
                    continue
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_U = sorted(set(new_U))
            else:
                continue
                
            if not new_U or 0 not in new_U:
                continue
                
            new_score = _score_set(new_U)
            
            if new_score > current_score:
                U = new_U
                current_score = new_score
            else:
                # Accept worse with small probability to escape local minima
                if random.random() < 0.02 and new_score > 0.95:
                     U = new_U
                     current_score = new_score

        return U, current_score

    # Strategy:
    # 1. Evaluate and improve all known Golomb rulers
    # 2. Run random restarts with random sets
    
    # 1. Golomb rulers
    for n in range(3, 13):
        if time.time() > deadline - 2:
            break
        if n in golomb_rulers:
            init_set = golomb_rulers[n][:]
            improved_set, score = improve_set(init_set, max_iters=100)
            if score > best_score:
                best_score = score
                best_set = improved_set

    # 2. Random restarts
    restarts = 0
    max_restarts = 100 # Reduced count because scoring is faster, but we have strong starts
    
    while time.time() < deadline - 2 and restarts < max_restarts:
        N = random.randint(3, 12)
        # Max val should be large enough to allow distinct differences
        max_val = N * N * 2 
        
        if N > max_val + 1:
            max_val = N + 1
            
        # Generate random set
        candidates = random.sample(range(1, max_val + 1), N - 1)
        init_set = [0] + candidates
        init_set.sort()
        
        improved_set, score = improve_set(init_set, max_iters=50)
        
        if score > best_score:
            best_score = score
            best_set = improved_set
            
        restarts += 1

    return best_set
# EVOLVE-BLOCK-END
