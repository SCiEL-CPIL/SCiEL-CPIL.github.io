import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # 1. Generate candidates with squared norm 4
    # Type 1: One 2, rest 0: ±2e_i
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    
    candidates = []
    
    # Type 1: 2 * D vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: C(11, 4) * 16 vectors
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    if N == 0:
        return np.array([], dtype=np.int64)
        
    # 2. Compute dot product matrix M
    # M[i,j] = candidates[i] . candidates[j]
    # Constraint for valid kissing config with max norm squared 4:
    # |u - v|^2 >= 4  =>  |u|^2 + |v|^2 - 2u.v >= 4
    # Since |u|^2 = |v|^2 = 4, this becomes 8 - 2u.v >= 4 => 2u.v <= 4 => u.v <= 2
    M = candidates @ candidates.T
    
    # 3. Greedy selection strategies
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy A: Multiple random orderings
    # We run several random trials to find a good greedy packing
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            
            # Check if idx is compatible with all currently selected vectors
            # M[idx, selected_mask] gives dot products with selected vectors
            # We need all dot products <= 2
            if np.all(M[idx, selected_mask] <= 2):
                selected_mask[idx] = True
                count += 1
                
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
            
    # Strategy B: Greedy with "least conflicts" priority
    # Vectors that conflict with fewer others are "easier" to place? 
    # Actually, we want to place vectors that have high compatibility with the rest.
    # Let's try sorting by the number of conflicts (degree in conflict graph).
    # Conflict if M[i,j] > 2.
    if time.time() < deadline - 20:
        conflict_counts = np.sum(M > 2, axis=1) - 1  # Exclude self
        # Sort by conflict count ascending (least conflicts first)
        order_b = np.argsort(conflict_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order_b:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            
            if np.all(M[idx, selected_mask] <= 2):
                selected_mask[idx] = True
                count += 1
                
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
            
    # Strategy C: Greedy with "most central" (highest degree in compatibility graph)
    # Vectors compatible with many others might form a dense core.
    if time.time() < deadline - 10:
        compat_counts = np.sum(M <= 2, axis=1) - 1
        order_c = np.argsort(-compat_counts, kind='stable')  # Highest compat first
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order_c:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            
            if np.all(M[idx, selected_mask] <= 2):
                selected_mask[idx] = True
                count += 1
                
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Strategy D: Local improvement / Swap search
    # Try to swap out selected vectors for unselected ones to gain more vectors?
    # Or simply try to add any unselected vector that fits.
    if best_set is not None and time.time() < deadline - 5:
        # First, try to add any remaining compatible vectors
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 2:
                break
            if np.all(M[idx, best_set] <= 2):
                best_set[idx] = True
                best_count += 1
        
        # Simple local search: try to replace one selected vector with two unselected ones?
        # This is expensive. Let's stick to the greedy + add remaining for now, 
        # as the random greedy is usually quite strong for this problem size.
        
    if best_set is None:
        # Fallback: just return Type 1 vectors (22 vectors)
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Number of vectors: {len(res)}")
    # Verify
    if len(res) > 0:
        norms_sq = np.sum(res**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        dists_sq = np.sum((res[:, np.newaxis, :] - res[np.newaxis, :, :])**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        print(f"Max norm squared: {max_norm_sq}")
        print(f"Min dist squared: {min_dist_sq}")
        print(f"Valid: {min_dist_sq >= max_norm_sq}")
