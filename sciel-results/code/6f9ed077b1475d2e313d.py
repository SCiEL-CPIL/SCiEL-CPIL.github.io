import numpy as np
import time

def simulate_aco(demands, capacity, H, seed):
    np.random.seed(seed)
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(30):
        bins_counts = []
        for _ in range(20):
            remaining = list(range(n))
            bins = 0
            current_bin_items = []
            current_load = 0
            last_item = -1
            
            while remaining:
                if not current_bin_items:
                    # Random fitting item
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_bin_items.append(i)
                    current_load = demands[i]
                    last_item = i
                    bins += 1
                else:
                    # Find feasible j
                    feasible_indices = []
                    weights = []
                    for k, j in enumerate(remaining):
                        if current_load + demands[j] <= capacity:
                            feasible_indices.append(k)
                            # prob proportional to tau[last_item, j] * H[last_item, j]
                            w = tau[last_item, j] * H[last_item, j]
                            # Ensure positive, prompt says non-positive becomes 1e-9 in eval, 
                            # but here we use H directly. If H is 0, w is 0.
                            if w <= 0: w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        # Open new bin
                        current_bin_items = []
                        current_load = 0
                        last_item = -1
                        continue
                    
                    # Select j
                    total_w = sum(weights)
                    r = np.random.rand() * total_w
                    cumulative = 0
                    chosen_k = feasible_indices[0]
                    for k_idx in range(len(feasible_indices)):
                        cumulative += weights[k_idx]
                        if r <= cumulative:
                            chosen_k = feasible_indices[k_idx]
                            break
                    
                    j = remaining[chosen_k]
                    remaining.pop(chosen_k)
                    current_bin_items.append(j)
                    current_load += demands[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        # Update Pheromone
        tau *= 0.9
        tau += 1.0 / best_this_iter
        
    return best_overall

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 285 # Safety margin (300s limit, leave 15s buffer)
    
    n = len(demands)
    d = np.array(demands)
    
    # Precompute sum matrix
    pair_sum = d[:, None] + d[None, :]
    feasible_mask = pair_sum <= capacity
    
    candidates = []
    
    # Candidate 1: Sum
    h1 = np.where(feasible_mask, pair_sum, 1e-9)
    candidates.append(h1)
    
    # Candidate 2: Sum Squared
    h2 = np.where(feasible_mask, pair_sum**2, 1e-9)
    candidates.append(h2)
    
    # Candidate 3: Sum Cubed
    h3 = np.where(feasible_mask, pair_sum**3, 1e-9)
    candidates.append(h3)
    
    # Candidate 4: Inverse Waste (eps=1)
    waste = capacity - pair_sum
    h4 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h4)
    
    # Candidate 5: Inverse Waste (eps=10)
    h5 = np.where(feasible_mask, 1.0 / (waste + 10.0), 1e-9)
    candidates.append(h5)
    
    # Candidate 6: Inverse Waste (eps=50)
    h6 = np.where(feasible_mask, 1.0 / (waste + 50.0), 1e-9)
    candidates.append(h6)
    
    # Candidate 7: Next Item Size
    h7 = np.where(feasible_mask, d[None, :], 1e-9)
    candidates.append(h7)
    
    # Candidate 8: Product
    h8 = np.where(feasible_mask, d[:, None] * d[None, :], 1e-9)
    candidates.append(h8)
    
    # Candidate 9: Exp Decay Waste
    h9 = np.where(feasible_mask, np.exp(-waste / 10.0), 1e-9)
    candidates.append(h9)
    
    # Candidate 10: Exp Decay Waste (slower)
    h10 = np.where(feasible_mask, np.exp(-waste / 50.0), 1e-9)
    candidates.append(h10)
    
    best_score = float('inf')
    best_h = h1 # Fallback
    
    # We can test with a few seeds to be robust
    seeds = [42, 123, 7]
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline - 30: # Need time to return
            break
            
        # Evaluate candidate
        scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, h, seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        # print(f"Candidate {idx}: {avg_score}")
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
        # If we have a very good score (e.g., optimal packing is likely around 120*50/150 = 40 bins),
        # and we get close, we might stop early? 
        # Min bins for 120 items of avg 60 in cap 150 is 120*60/150 = 48.
        # If we get 50, it's good.
        
    return best_h
