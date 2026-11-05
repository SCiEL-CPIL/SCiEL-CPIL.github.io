import time
import random
import numpy as np

def _count_nonzero_square_fast(coeffs):
    if not coeffs:
        return 0
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def _validate(coeffs):
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    return True

def _sparse_to_dense(sparse, max_deg=40):
    coeffs = [0] * (max_deg + 1)
    for exp, c in sparse:
        if 0 <= exp <= max_deg:
            coeffs[exp] = c
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _dense_to_sparse(coeffs):
    return [(i, c) for i, c in enumerate(coeffs) if c != 0]

def _local_search(sparse_init, deadline):
    current_sparse = list(sparse_init)
    current_dense = _sparse_to_dense(current_sparse)
    current_score = _count_nonzero_square_fast(current_dense)
    
    best_sparse = list(current_sparse)
    best_score = current_score
    
    coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
    max_exp = 40
    
    no_improve = 0
    max_no_improve = 100 # Reduced to allow more restarts
    
    while time.time() < deadline - 1.0:
        if no_improve > max_no_improve:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            idx = random.randrange(12)
            e, c = current_sparse[idx]
            shift = random.randint(-5, 5)
            if shift == 0:
                continue
            new_e = e + shift
            if 0 <= new_e <= max_exp:
                other_exps = set(x[0] for i, x in enumerate(current_sparse) if i != idx)
                if new_e not in other_exps:
                    current_sparse[idx] = (new_e, c)
                    trial_dense = _sparse_to_dense(current_sparse)
                    trial_score = _count_nonzero_square_fast(trial_dense)
                    if trial_score < current_score:
                        current_score = trial_score
                        current_dense = trial_dense
                        no_improve = 0
                        if trial_score < best_score:
                            best_score = trial_score
                            best_sparse = list(current_sparse)
                    else:
                        current_sparse[idx] = (e, c)
                        no_improve += 1
        elif move_type < 0.8:
            idx = random.randrange(12)
            e, c = current_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c == c:
                continue
            current_sparse[idx] = (e, new_c)
            trial_dense = _sparse_to_dense(current_sparse)
            trial_score = _count_nonzero_square_fast(trial_dense)
            if trial_score < current_score:
                current_score = trial_score
                current_dense = trial_dense
                no_improve = 0
                if trial_score < best_score:
                    best_score = trial_score
                    best_sparse = list(current_sparse)
            else:
                current_sparse[idx] = (e, c)
                no_improve += 1
        else:
            i, j = random.sample(range(12), 2)
            ei, ci = current_sparse[i]
            ej, cj = current_sparse[j]
            current_sparse[i] = (ej, ci)
            current_sparse[j] = (ei, cj)
            trial_dense = _sparse_to_dense(current_sparse)
            trial_score = _count_nonzero_square_fast(trial_dense)
            if trial_score < current_score:
                current_score = trial_score
                current_dense = trial_dense
                no_improve = 0
                if trial_score < best_score:
                    best_score = trial_score
                    best_sparse = list(current_sparse)
            else:
                current_sparse[i] = (ei, ci)
                current_sparse[j] = (ej, cj)
                no_improve += 1
                
    return best_sparse, best_score

def _generate_seed(strategy):
    if strategy == 0:
        exps = random.sample(range(0, 41), 12)
        coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        return list(zip(exps, coeffs))
    elif strategy == 1:
        center = random.randint(10, 25)
        sparse = []
        for i in range(6):
            offset = random.randint(1, 10)
            e1 = center - offset
            e2 = center + offset
            if e1 < 0 or e2 > 40:
                continue
            c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            sparse.append((e1, c))
            sparse.append((e2, -c))
        if len(sparse) != 12:
            return _generate_seed(0)
        return sparse
    elif strategy == 2:
        start = random.randint(0, 20)
        step = random.randint(1, 5)
        exps = []
        for i in range(12):
            e = start + i * step
            if e > 40:
                break
            exps.append(e)
        if len(exps) != 12:
            return _generate_seed(0)
        coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        return list(zip(exps, coeffs))
    else:
        center = random.randint(10, 30)
        exps = set()
        while len(exps) < 12:
            e = center + random.randint(-5, 5)
            if 0 <= e <= 40:
                exps.add(e)
        exps = list(exps)
        coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        return list(zip(exps, coeffs))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = float('inf')
    best_sparse = None
    
    random.seed(12345)
    
    # Run for as many restarts as possible
    while time.time() < deadline - 5.0:
        strategy = random.randint(0, 3)
        seed_sparse = _generate_seed(strategy)
        
        dense_seed = _sparse_to_dense(seed_sparse)
        if not _validate(dense_seed):
            continue
            
        sparse_res, score_res = _local_search(seed_sparse, deadline)
        
        if score_res < best_score:
            best_score = score_res
            best_sparse = sparse_res
            
        if best_score <= 14:
            break
            
    if best_sparse is None:
        best_sparse = [(i, 1) for i in range(12)]
        
    best_dense = _sparse_to_dense(best_sparse)
    if not _validate(best_dense):
        best_dense = [1] * 12
        
    return best_dense
