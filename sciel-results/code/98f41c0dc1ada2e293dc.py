import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Precompute neighbors as boolean masks for fast set operations
    # A_bool[i] is a boolean array of length N, True where neighbor
    A_bool = A > 0.5
    
    # Helper: Compute the size of the independent set given a permutation
    # perm is an array of vertex indices, sorted by priority (highest first)
    def compute_is_size(perm):
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in perm:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                # Mark neighbors as picked (blocked)
                is_picked[A_bool[v]] = True
        return count

    # Helper: Compute a "regret" score which is inversely related to IS size
    # We want to maximize IS size, so minimize regret = N - IS_size?
    # Actually, let's just maximize IS size directly in the annealing loop.
    # But annealing usually minimizes energy. Let's define Energy = -IS_size.
    
    # Initial Strategy: 
    # A good starting point is often sorting by degree (ascending) or some heuristic.
    # Let's try a few initial heuristics and pick the best one.
    
    degrees = A.sum(axis=1)
    
    heuristics = [
        np.argsort(degrees, kind='stable'),          # Low degree first
        np.argsort(-degrees, kind='stable'),         # High degree first
        np.argsort(degrees / (A @ degrees + 1e-9), kind='stable'), # Low avg neighbor degree
        np.random.permutation(N)                     # Random
    ]
    
    best_init_size = -1
    best_init_perm = None
    
    for h in heuristics:
        size = compute_is_size(h)
        if size > best_init_size:
            best_init_size = size
            best_init_perm = h.copy()
            
    # Simulated Annealing
    # State: permutation of vertices
    # Move: Swap two vertices in the permutation
    # Cost: Change in IS size
    
    current_perm = best_init_perm
    current_size = best_init_size
    best_perm = best_init_perm.copy()
    best_size = best_init_size
    
    # Annealing parameters
    # We need to be fast. N=200.
    # Swapping two vertices i, j in the permutation.
    # To evaluate the change efficiently, we might need to re-evaluate the whole thing 
    # or use incremental updates. 
    # Given N=200, re-evaluating the whole IS size takes O(N^2) worst case (or O(N+E)).
    # With 150s budget, we can afford many evaluations.
    # Let's estimate speed.
    
    # Test speed of compute_is_size
    test_start = time.time()
    iterations_test = 0
    while time.time() - test_start < 0.1:
        compute_is_size(current_perm)
        iterations_test += 1
    
    evals_per_sec = iterations_test / 0.1
    # print(f"Evals per sec: {evals_per_sec}")
    
    # If evals_per_sec is high (e.g., > 1000), we can do a lot of swaps.
    # If low, we need fewer.
    
    # Let's assume we can do ~1000-5000 evals/sec.
    # Total time available: ~140s.
    # Max swaps: ~100,000 - 500,000.
    
    # Simulated Annealing Logic
    T_start = 10.0  # Temperature (since we maximize size, delta is integer, T should be comparable)
    T_end = 0.1
    alpha = 0.9999  # Cooling rate, needs to be tuned based on N_swaps
    
    # We will run for a fixed number of swaps or until time runs out.
    # Let's aim for ~200,000 swaps if possible, otherwise time-bound.
    
    rng = np.random.default_rng(42)
    
    # To make swaps effective, we should probably swap vertices that are "close" in the permutation
    # or random swaps. Random swaps in a permutation of size 200 might be too disruptive.
    # Let's try swapping adjacent elements first (local search) then random.
    # Actually, for IS, the order matters globally. 
    # Let's use a mix: 50% adjacent swaps, 50% random swaps.
    
    swaps_done = 0
    max_swaps = int(evals_per_sec * 130) # Use 130s for search
    
    # Adaptive cooling
    # T = T_start * (T_end/T_start)^(swaps/max_swaps)
    log_ratio = np.log(T_end / T_start)
    
    while swaps_done < max_swaps:
        if time.time() > DEADLINE - 2.0:
            break
            
        # Generate move
        if rng.random() < 0.5:
            # Adjacent swap
            if N < 2: break
            i = rng.integers(0, N - 1)
            j = i + 1
        else:
            # Random swap
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
        
        # Perform swap
        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        # Evaluate
        new_size = compute_is_size(current_perm)
        
        # Calculate delta (we want to maximize)
        delta = new_size - current_size
        
        # Update temperature
        # Instead of pre-computing T for each step, we can just use a simple schedule
        # Or compute T on the fly.
        # T = T_start * (T_end/T_start)^(swaps_done / max_swaps)
        # This is expensive to compute log/exp every time.
        # Let's use a simple linear or geometric cooling if we know max_swaps.
        # Since max_swaps is estimated, let's just use a simple decay.
        
        # Simplified: Use a fixed number of swaps per "epoch" and cool down?
        # Or just use the formula. np.exp is fast.
        progress = swaps_done / max_swaps
        T = T_start * np.exp(log_ratio * progress)
        
        # Acceptance
        if delta > 0:
            # Improvement
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_perm = current_perm.copy()
        else:
            # Worse, accept with probability exp(delta/T)
            # delta is negative
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_size = new_size
                else:
                    # Reject, swap back
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        swaps_done += 1
        
        # Occasionally check if we are stuck? 
        # If no improvement for long time, maybe restart?
        # For now, keep it simple.
        
    # Final refinement: Local search on the best permutation found
    # Try swapping all adjacent pairs and accept improvements (Hill Climbing)
    perm = best_perm.copy()
    size = best_size
    improved = True
    while improved and time.time() < DEADLINE - 0.5:
        improved = False
        for i in range(N - 1):
            if time.time() > DEADLINE - 0.5:
                break
            perm[i], perm[i+1] = perm[i+1], perm[i]
            new_size = compute_is_size(perm)
            if new_size > size:
                size = new_size
                improved = True
                if size > best_size:
                    best_size = size
                    best_perm = perm.copy()
            else:
                # Swap back
                perm[i], perm[i+1] = perm[i+1], perm[i]
                
    # Convert best permutation to scores
    # The score should be such that sorting by descending score yields best_perm
    # If best_perm[0] is the highest priority, it should have the highest score.
    scores = np.zeros(N)
    for rank, v in enumerate(best_perm):
        # Higher rank (lower index) = higher priority = higher score
        scores[v] = N - rank
        
    return scores

# EVOLVE-BLOCK-END
