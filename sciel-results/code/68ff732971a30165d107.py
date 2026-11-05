import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    def get_counts(mask):
        """
        mask: numpy array of 0s and 1s of length 129
        returns: array of counts for sums 128..256
        """
        conv = np.convolve(mask, mask, mode='full')
        counts = np.zeros(max_sum - min_sum + 1, dtype=np.int64)
        for i, s in enumerate(range(min_sum, max_sum + 1)):
            if s < len(conv):
                ordered = conv[s]
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= n and mask[half]:
                        diag = 1
                    else:
                        diag = 0
                else:
                    diag = 0
                counts[i] = (ordered + diag) // 2
        return counts

    def compute_cost(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

    best_score = float('inf')
    best_mask = None

    def update_best(mask):
        nonlocal best_score, best_mask
        if not np.any(mask):
            return
        cost = compute_cost(mask)
        if cost < best_score:
            best_score = cost
            best_mask = mask.copy()

    # 1. Heuristic Initializations
    # Symmetric ranges
    for offset in range(0, 30):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[lo:hi+1] = 1
        update_best(mask)
        
    # Modular
    for mod in [3, 4, 5, 6, 7, 8]:
        for keep in range(1, mod):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(n + 1):
                if x % mod != mod - keep:
                    mask[x] = 1
            update_best(mask)
            
    # Random
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 50: break
        size = random.randint(20, 60)
        A = random.sample(range(n + 1), size)
        mask = list_to_mask(A)
        update_best(mask)

    # 2. Differential Evolution
    # We define the objective for DE.
    # DE works on continuous variables [0, 1].
    def de_objective(x):
        # x is a numpy array of floats
        mask = (x > 0.5).astype(np.int64)
        if not np.any(mask):
            return 10000
        counts = get_counts(mask)
        if np.any(counts == 0):
            return 10000 + np.sum(counts == 0)
        return float(np.max(counts))

    if time.time() < deadline - 30:
        try:
            bounds = [(0, 1) for _ in range(n + 1)]
            # Use 'best1bin' for binary-like problems
            res = differential_evolution(
                de_objective, 
                bounds, 
                strategy='best1bin',
                popsize=15,
                maxiter=30,
                tol=1e-4,
                seed=42,
                polish=False, # Disable polish to save time, we do our own local search
                init='latinhypercube'
            )
            if res.success or res.fun < 10000:
                best_de_mask = (res.x > 0.5).astype(np.int64)
                update_best(best_de_mask)
        except Exception:
            pass

    # 3. Local Search Refinement
    if best_mask is None:
        # Fallback
        best_mask = np.zeros(n + 1, dtype=np.int64)
        best_mask[64:129] = 1
        best_score = compute_cost(best_mask)

    current_mask = best_mask.copy()
    current_cost = best_score
    in_set = set(mask_to_list(current_mask))
    
    def try_local_search(mask, cost, time_limit):
        nonlocal best_score, best_mask
        current_mask = mask.copy()
        current_cost = cost
        in_set = set(mask_to_list(current_mask))
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try removing
            # Iterate in reverse to handle index shifts
            for i in range(len(current_mask) - 1, -1, -1):
                if time.time() > time_limit: break
                if current_mask[i] == 1:
                    trial_mask = current_mask.copy()
                    trial_mask[i] = 0
                    if not np.any(trial_mask): continue
                    tc = compute_cost(trial_mask)
                    if tc <= current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        in_set.discard(i)
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        break
            
            # 2. Try adding
            if not improved:
                for i in range(n + 1):
                    if time.time() > time_limit: break
                    if current_mask[i] == 0:
                        trial_mask = current_mask.copy()
                        trial_mask[i] = 1
                        tc = compute_cost(trial_mask)
                        if tc < current_cost:
                            current_mask = trial_mask
                            current_cost = tc
                            in_set.add(i)
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            break
                            
            # 3. Try swapping
            if not improved:
                # Try swapping each 1 with a 0
                ones = [i for i in range(n+1) if current_mask[i] == 1]
                zeros = [i for i in range(n+1) if current_mask[i] == 0]
                
                # Limit swaps to speed up if sets are large
                if len(ones) > 20:
                    ones = random.sample(ones, 20)
                if len(zeros) > 20:
                    zeros = random.sample(zeros, 20)
                    
                for i in ones:
                    if time.time() > time_limit: break
                    for j in zeros:
                        if time.time() > time_limit: break
                        trial_mask = current_mask.copy()
                        trial_mask[i] = 0
                        trial_mask[j] = 1
                        tc = compute_cost(trial_mask)
                        if tc < current_cost:
                            current_mask = trial_mask
                            current_cost = tc
                            in_set.discard(i)
                            in_set.add(j)
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            break
                    if improved:
                        break
                        
            if not improved:
                # If no simple move improves, try random perturbation to escape local minimum
                if time.time() < time_limit - 5:
                    # Perturb: flip 2-3 random bits
                    flip_count = random.randint(2, 4)
                    indices = random.sample(range(n+1), flip_count)
                    trial_mask = current_mask.copy()
                    for idx in indices:
                        trial_mask[idx] = 1 - trial_mask[idx]
                    tc = compute_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        in_set = set(mask_to_list(current_mask))
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                    elif tc == current_cost and random.random() < 0.1:
                        # Accept neutral move sometimes to explore
                        current_mask = trial_mask
                        in_set = set(mask_to_list(current_mask))
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                    
                if not improved:
                    break

    try_local_search(best_mask, best_score, deadline - 5)

    # Final check
    if best_mask is None or not np.any(best_mask):
        return list(range(64, 129))
    
    return mask_to_list(best_mask)
