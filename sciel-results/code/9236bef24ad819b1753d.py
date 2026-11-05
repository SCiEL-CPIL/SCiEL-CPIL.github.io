import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a structural heuristic that favors vertices with
       low degree and low local density (specifically, vertices adjacent to hubs
       are penalized less than hubs themselves, promoting 'hub-blocking').
    2. Refine these scores using local search (pair swaps) to directly maximize
       the size of the independent set produced by the greedy decoder.
    3. Use multiple restarts with noise to escape local optima.
    
    The key insight is that the greedy decoder's output is highly sensitive to
    small changes in the priority ordering, especially for vertices with similar
    scores. Local search allows us to navigate this landscape effectively.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(A_bool[i])[0].tolist() for i in range(N)]
    
    # --- 1. Initial Heuristic using Structural Features + Low-Discrepancy Mapping ---
    # We want to favor vertices that are "safe" to pick.
    # Low degree is generally good.
    # Low local density is good.
    # Eigenvector centrality captures "influence" - low centrality means less entangled.
    
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors (local clustering/density proxy)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Compute eigenvector centrality via power iteration
    # This captures the "hub-ness" of each vertex in the global structure
    eig_vec = np.ones(N) / N
    for _ in range(20):  # 20 iterations should converge for N=200
        eig_vec_new = A.dot(eig_vec)
        norm = np.linalg.norm(eig_vec_new)
        if norm > 1e-10:
            eig_vec = eig_vec_new / norm
        else:
            break
    
    # Normalize all features to [0, 1]
    def normalize(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        else:
            return np.zeros_like(x)
    
    norm_deg = normalize(deg)
    norm_avg_ndeg = normalize(avg_neighbor_deg)
    norm_eig = normalize(eig_vec)
    
    # We want LOW degree, LOW local density, LOW eigenvector centrality.
    # Combine into a single "badness" score (higher = worse)
    badness = 0.5 * norm_deg + 0.25 * norm_avg_ndeg + 0.25 * norm_eig
    
    # Now map to scores using a low-discrepancy approach.
    # Instead of linear mapping, we use the rank of badness to create a more uniform spread.
    # This ensures vertices are spread across the structural spectrum.
    # 
    # The key insight: by using rank-based mapping, we avoid clustering of similar
    # vertices in the priority order, which helps the greedy decoder explore more
    # of the graph's structure.
    
    # Get ranks (0 = best/lowest badness, N-1 = worst/highest badness)
    ranks = np.argsort(np.argsort(badness))  # Double argsort gives ranks
    
    # Map ranks to scores using a slightly non-linear transform to emphasize
    # the spread. We use a power transform that stretches the middle range.
    # This creates a low-discrepancy-like distribution in the score space.
    rank_frac = ranks / (N - 1 + 1e-10)  # [0, 1]
    
    # Apply a gentle S-curve to stretch the middle and compress extremes
    # This ensures that vertices with similar structural features don't all
    # cluster at similar priority levels
    scores = 1.0 - rank_frac  # Higher score = lower badness = picked first
    
    # Apply a subtle non-linearity to create more spread in the middle
    # Use a sine-based transform that peaks at 0.5
    scores = 0.5 + 0.5 * np.sin(np.pi * scores)
    
    # Final normalization to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)
    
    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    # Deactivate neighbors
                    for nb in nbrs:
                        if active[nb]:
                            active[nb] = False
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # --- 3. Local Search ---
    rng = np.random.default_rng(42)
    
    # Precompute score order for structured swaps
    score_order = np.argsort(-scores)
    
    # Try multiple restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # For restart 0, use original. For others, perturb from best found.
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Add noise to escape local optima
            # Noise scale decreases slightly for later restarts? Or stays constant?
            # Let's keep it moderate.
            noise_scale = 0.1
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        # Local search loop
        no_improve = 0
        max_no_improve = 2000 + 500 * restart
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose swap candidates
            # 60% structured (close in rank), 40% random
            if rng.random() < 0.6:
                base_idx = rng.integers(0, N - 1)
                # Pick a nearby index
                offset = rng.integers(1, min(20, N - 1))
                i = score_order[base_idx]
                j = score_order[min(base_idx + offset, N - 1)]
            else:
                i, j = rng.integers(0, N, size=2)
            
            if i == j:
                continue
            
            # Swap scores
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                    # Update score order if we want to keep it fresh?
                    # It's expensive to re-sort every time.
                    # The structured swaps rely on the *initial* or *last updated* order.
                    # Since scores change, the rank order changes.
                    # However, re-sorting is O(N log N). N=200, so it's fast.
                    # Let's update score_order occasionally or if we find a new best.
                    score_order = np.argsort(-current_scores)
            else:
                # Revert swap
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # Early stopping if no improvement in this restart
        if current_val <= best_val and restart > 2:
            break
    
    return best_scores
