import numpy as np
import time
import random

def _greedy_init(n, m, M, rng):
    """
    Greedily construct a family of m subsets.
    At each step, add the subset that maximizes the number of comparable pairs
    with the current family.
    """
    N = 1 << n
    family = []
    in_family = np.zeros(N, dtype=bool)
    count_vec = np.zeros(N, dtype=np.int32)
    
    # Start with a random set to break symmetry
    first = rng.choice(N)
    family.append(int(first))
    in_family[first] = True
    count_vec += M[first, :]
    
    for _ in range(1, m):
        # Find the best candidate not in family
        # We want to maximize count_vec[j] for j not in family
        candidates = count_vec.copy()
        candidates[in_family] = -1
        
        # To add diversity, we can pick from top K candidates
        # For now, just pick the best, but maybe add some randomness if tied
        # or use a soft-max selection if we want more diversity
        best_idx = int(np.argmax(candidates))
        
        # If there are ties, pick randomly among them?
        # Let's just pick the argmax for now, as greedy is strong.
        # However, to avoid identical restarts, we can shuffle the initial choice
        # or pick randomly from top 10.
        
        # Let's pick randomly from top 5 to diversify restarts
        # Get indices of top 5
        # This is O(N log N) or O(N) if we just find top 5
        # Since N=4096, we can just sort or use argpartition
        if len(family) > 0:
             # Get top 5 indices
             # Note: candidates has -1 for in_family, so they won't be in top 5 unless all are -1 (impossible)
             top_k_indices = np.argpartition(candidates, -5)[-5:]
             # Filter out any that are in family (shouldn't happen due to -1, but safe)
             valid_indices = [int(i) for i in top_k_indices if not in_family[i]]
             if valid_indices:
                 best_idx = rng.choice(valid_indices)
        
        family.append(best_idx)
        in_family[best_idx] = True
        count_vec += M[best_idx, :]
        
    return family

def construct():
    n = 12
    m = 40
    N = 1 << n
    deadline = time.time() + 55.0  # Safety margin for 60s limit, 5s buffer
    
    # 1. Precompute Comparability Matrix
    all_masks = np.arange(N, dtype=np.int32)
    M = np.zeros((N, N), dtype=np.uint8)
    
    for i in range(N):
        sub1 = (i & all_masks) == i
        sub2 = (all_masks & i) == all_masks
        row = sub1 | sub2
        row[i] = 0
        M[i, :] = row.astype(np.uint8)
        
    best_score = -1
    best_family = None
    
    # 2. Multiple Restarts with Greedy Init and Simulated Annealing
    num_restarts = 20  # Fewer restarts but longer/better search per restart
    
    rng = np.random.default_rng()
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Greedy Initialization
        family = _greedy_init(n, m, M, rng)
        family = [int(x) for x in family]
        
        in_family = np.zeros(N, dtype=bool)
        for x in family:
            in_family[x] = True
            
        count_vec = np.zeros(N, dtype=np.int32)
        score = 0
        
        # Calculate initial score and count_vec
        for A in family:
            count_vec += M[A, :]
        score = int(np.sum(count_vec) // 2)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
            
        # Simulated Annealing Local Search
        # Temperature schedule: Start high, cool down
        T = 5.0 
        cooling_rate = 0.95
        min_T = 0.1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                    
                A = family[i]
                old_loss = count_vec[A]
                
                # Remove A
                count_vec -= M[A, :]
                in_family[A] = False
                
                # Find best B
                masked = count_vec.copy()
                masked[in_family] = -1
                B = int(np.argmax(masked))
                new_gain = count_vec[B]
                
                delta = new_gain - old_loss
                
                # Simulated Annealing Acceptance
                accept = False
                if delta >= 0:
                    accept = True
                else:
                    # Accept with probability exp(delta / T)
                    if T > min_T:
                        if random.random() < np.exp(delta / T):
                            accept = True
                    # If T is very low, only accept improvements (or equal)
                
                if accept:
                    # Accept swap
                    family[i] = B
                    in_family[B] = True
                    count_vec += M[B, :]
                    score += delta
                    improved = True
                    
                    if score > best_score:
                        best_score = score
                        best_family = family[:]
                else:
                    # Revert
                    count_vec += M[A, :]
                    in_family[A] = True
                    
            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T
                # If we've cooled down, we might break early if no improvements in last pass?
                # The 'improved' flag handles this. If no improvements, loop exits.
                
    # Convert best_family to lists
    if best_family is None:
        # Fallback
        best_family = list(range(m))
        
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
