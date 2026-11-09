"""Combined approach: Vectorized Greedy Construction + Aggressive Local Search.
Uses NumPy for O(1) isosceles checks to enable extensive multi-start and local search
refinement, aiming to find smaller maximal no-isosceles sets than pure Python heuristics."""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # Safety margin before 330s kill

# Precompute all points and indices
ALL_POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=np.int32)
NUM_POINTS = len(ALL_POINTS)
POINT_TO_IDX = {tuple(p): i for i, p in enumerate(ALL_POINTS)}

# Precompute squared distances between all pairs
# D[i, j] is squared distance between ALL_POINTS[i] and ALL_POINTS[j]
diff = ALL_POINTS[:, np.newaxis, :] - ALL_POINTS[np.newaxis, :, :]
DIST_MATRIX = np.sum(diff * diff, axis=2)

def _can_add_indices(p_idx, S_indices):
    """
    Check if point p_idx can be added to set S_indices (list of indices).
    Returns True if no isosceles triangle is formed.
    Vectorized check using DIST_MATRIX.
    """
    if len(S_indices) < 2:
        return True
    
    # Distances from p to all points in S
    d_p_s = DIST_MATRIX[p_idx, S_indices]
    
    # Check 1: Is p equidistant to any two points in S?
    # This means d_p_s has duplicate values.
    # Note: d_p_s values are squared distances.
    if len(np.unique(d_p_s)) < len(d_p_s):
        return False

    # Check 2 & 3: Is d(p, S[i]) == d(S[i], S[j]) or d(p, S[j]) == d(S[i], S[j])?
    # We need to check if any d_p_s[i] matches any d(S[i], S[j]) for i != j.
    # D_S is the submatrix of distances within S.
    D_S = DIST_MATRIX[np.ix_(S_indices, S_indices)]
    
    # We want to check if any value in d_p_s appears in D_S (excluding diagonal which is 0, 
    # but d_p_s is distance from p to S, so p not in S implies d_p_s > 0 usually, 
    # unless p coincides with a point in S, which is not the case here).
    # Actually, we must be careful: d_p_s[i] is distance between p and S[i].
    # D_S[i, j] is distance between S[i] and S[j].
    # Condition: d_p_s[i] == D_S[i, j] OR d_p_s[j] == D_S[i, j] for any i != j.
    
    # Efficient check:
    # Create a boolean mask for matches between d_p_s and D_S.
    # match[i, j] = (d_p_s[i] == D_S[i, j]) | (d_p_s[j] == D_S[i, j])
    # We can vectorize this.
    
    # d_p_s shape: (k,)
    # D_S shape: (k, k)
    
    # We need to check if there exists i, j such that i != j and (d_p_s[i] == D_S[i, j] or d_p_s[j] == D_S[i, j])
    
    # Let's create a matrix M where M[i, j] = D_S[i, j]
    # We want to check if d_p_s is in the set of values { D_S[i, j] | i != j }? 
    # No, the index matters.
    # If d_p_s[i] == D_S[i, j], that forms an isosceles triangle with vertices p, S[i], S[j] where sides p-S[i] and S[i]-S[j] are equal.
    # If d_p_s[j] == D_S[i, j], that forms an isosceles triangle with vertices p, S[j], S[i] where sides p-S[j] and S[i]-S[j] are equal.
    
    # So, for each pair (i, j) in S, we check if d_p_s[i] == D_S[i, j] or d_p_s[j] == D_S[i, j].
    
    # Vectorized approach:
    # d_p_s_col = d_p_s[:, np.newaxis]  # shape (k, 1)
    # d_p_s_row = d_p_s[np.newaxis, :]  # shape (1, k)
    # matches = (d_p_s_col == D_S) | (d_p_s_row == D_S)
    # We need to exclude the diagonal i==j? 
    # If i==j, D_S[i, i] = 0. d_p_s[i] is distance p-S[i]. 
    # If p != S[i], d_p_s[i] > 0, so d_p_s[i] != 0. 
    # So diagonal matches are impossible for valid p not in S.
    
    matches = (d_p_s[:, np.newaxis] == D_S) | (d_p_s[np.newaxis, :] == D_S)
    
    # If any match is True, return False
    if np.any(matches):
        return False
        
    return True

def _build_greedy_order(order, rng=None):
    """Simple greedy in a given order."""
    S_indices = []
    for p in order:
        idx = POINT_TO_IDX[p]
        if _can_add_indices(idx, S_indices):
            S_indices.append(idx)
    return S_indices

def _build_minimal_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    S_indices = []
    addable = set(range(NUM_POINTS))
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if not _can_add_indices(p_idx, S_indices):
                continue
            
            new_S = S_indices + [p_idx]
            block_count = 0
            for q_idx in addable:
                if q_idx == p_idx:
                    continue
                if not _can_add_indices(q_idx, new_S):
                    block_count += 1
            
            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p = p_idx
        
        if best_p is None:
            break
        
        S_indices.append(best_p)
        addable.discard(best_p)
        
        # Update addable
        new_addable = set()
        for q_idx in addable:
            if _can_add_indices(q_idx, S_indices):
                new_addable.add(q_idx)
        addable = new_addable
        
    return S_indices

def _local_search(S_indices, max_iterations=100, time_limit=30, start_time=0):
    """
    Local search refinement: try to reduce set size by removing points and refilling.
    """
    S = list(S_indices)
    best_size = len(S)
    best_S = list(S)
    
    # Precompute addable set for current S
    def compute_addable(S_list):
        S_set = set(S_list)
        addable = set()
        for p_idx in range(NUM_POINTS):
            if p_idx not in S_set and _can_add_indices(p_idx, S_list):
                addable.add(p_idx)
        return addable

    addable = compute_addable(S)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing each point in S
        for idx, removed in enumerate(S):
            if time.time() - start_time > time_limit:
                break
            
            S_trial = S[:idx] + S[idx+1:]
            
            # Check if S_trial is maximal
            # If it is maximal, we have a smaller set!
            # If it is not maximal, we can try to refill it to see if we get a smaller set?
            # Actually, if we remove a point and the set is NOT maximal, we MUST add points back to make it maximal.
            # The resulting maximal set might be smaller, equal, or larger.
            # If we remove a point and the set IS maximal, we immediately have a smaller set.
            
            # Check maximality of S_trial
            is_maximal = True
            for p_idx in range(NUM_POINTS):
                if p_idx not in S_trial:
                    if _can_add_indices(p_idx, S_trial):
                        is_maximal = False
                        break
            
            if is_maximal:
                S = S_trial
                best_size = len(S)
                best_S = list(S)
                addable = compute_addable(S)
                break  # Restart iteration after a successful removal
        
        # Random perturbation: Remove a random point, then greedily refill
        if random.random() < 0.5:
            if len(S) > 0:
                idx = random.randrange(len(S))
                S_trial = S[:idx] + S[idx+1:]
                
                # Greedily refill S_trial to make it maximal
                S_rebuilt = list(S_trial)
                S_rebuilt_set = set(S_rebuilt)
                addable = compute_addable(S_rebuilt)
                
                # Shuffle addable for random refill
                add_list = list(addable)
                random.shuffle(add_list)
                
                for p_idx in add_list:
                    if _can_add_indices(p_idx, S_rebuilt):
                        S_rebuilt.append(p_idx)
                        S_rebuilt_set.add(p_idx)
                        # Recompute addable
                        addable = compute_addable(S_rebuilt)
                        if not addable:
                            break
                
                if len(S_rebuilt) < best_size:
                    S = S_rebuilt
                    best_size = len(S)
                    best_S = list(S)
                    addable = compute_addable(S)
    
    return best_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    
    # 1. Baseline: Row-major greedy
    row_major_order = list(itertools.product(range(N), repeat=2))
    S1 = _build_greedy_order(row_major_order)
    best_set = S1
    best_size = len(S1)
    
    # 2. Multi-start with Minimum-Degree Greedy + Local Search
    # We can afford more attempts due to vectorization
    attempts = 50
    for seed in range(attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        S = _build_minimal_greedy(order_seed=seed)
        
        # Apply local search refinement
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 5:
            S_refined = _local_search(S, max_iterations=50, time_limit=min(remaining, 15), start_time=start_time)
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
        elif len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # 3. Random Orderings + Local Search
    rng_global = random.Random(42)
    for _ in range(20):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        random_order = list(itertools.product(range(N), repeat=2))
        rng_global.shuffle(random_order)
        S = _build_greedy_order(random_order)
        
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 5:
            S_refined = _local_search(S, max_iterations=50, time_limit=min(remaining, 10), start_time=start_time)
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
    
    # Convert indices back to (x, y) tuples
    return [tuple(ALL_POINTS[i]) for i in best_set]

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
