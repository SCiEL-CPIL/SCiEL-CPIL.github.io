import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    A = adjacency.astype(float)
    degrees = A.sum(axis=1)
    neighbor_degree_sums = A @ degrees
    
    # Normalize features
    deg_norm = degrees / (degrees.max() + 1e-10)
    nds_norm = neighbor_degree_sums / (neighbor_degree_sums.max() + 1e-10)
    
    # Base score: Low degree is good, High neighbor degree sum is good (removes hubs)
    # We want to pick vertices that are low degree but connected to high degree nodes?
    # Or just low degree?
    # Let's use a mix.
    base_score = -0.7 * deg_norm + 0.3 * nds_norm
    
    # Ensemble Simulation
    # We will run multiple greedy simulations with noise.
    # Count how often each vertex is picked, weighted by the size of the resulting set.
    
    vote_counts = np.zeros(n)
    weights_sum = 0.0
    
    num_simulations = 200
    start_time = time.time()
    
    # Pre-compute neighbor lists for speed
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]
    
    for sim in range(num_simulations):
        # Check time
        if time.time() - start_time > 120:
            break
            
        # Add noise to base score
        # Noise scale: small enough to explore, large enough to change order
        noise = np.random.normal(0, 0.01, n)
        scores = base_score + noise
        
        # Greedy Decoder Simulation
        picked = set()
        remaining = set(range(n))
        current_scores = scores.copy()
        
        # Optimization: Instead of re-sorting every time, we can just pick max from remaining.
        # But n=200, sorting 200 items is fast.
        # However, we need to skip neighbors.
        
        # Efficient simulation:
        # 1. Sort vertices by score descending.
        # 2. Iterate through sorted list.
        # 3. If vertex is not in picked and not neighbor of any picked, pick it.
        
        order = np.argsort(-current_scores)
        
        for v in order:
            if v in picked:
                continue
            # Check if any neighbor is picked
            is_neighbor_of_picked = False
            for nb in neighbors[v]:
                if nb in picked:
                    is_neighbor_of_picked = True
                    break
            if not is_neighbor_of_picked:
                picked.add(v)
        
        # Update votes
        set_size = len(picked)
        # Weight by set size (better solutions vote more)
        w = set_size
        weights_sum += w
        for v in picked:
            vote_counts[v] += w
    
    # Normalize votes
    if weights_sum > 0:
        vote_counts /= weights_sum
    else:
        vote_counts = np.zeros(n)
        
    # Final score: Combine vote counts with original features?
    # The vote counts already reflect the "goodness" of picking the vertex.
    # We can just use vote_counts as the score.
    # But we need to handle ties or ensure a good distribution.
    
    # Add a small deterministic tie-breaker based on index
    # Or combine with base score to stabilize?
    # If we just use vote_counts, vertices with 0 votes get 0.
    # Vertices with high votes get high scores.
    
    final_scores = vote_counts.copy()
    
    # Add tiny index based epsilon to break ties deterministically
    final_scores += np.arange(n) * 1e-10
    
    return final_scores
