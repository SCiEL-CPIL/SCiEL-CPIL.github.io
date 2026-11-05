import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores based on negative degree (low degree is better for IS).
    2. Use Simulated Annealing to optimize the continuous score vector.
       The objective is the size of the independent set produced by the greedy decoder.
       Since the objective is discrete, we use Metropolis-Hastings acceptance.
    3. Finish with a local search phase using random swaps to fine-tune the ordering.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    # Using lists of numpy arrays for fast boolean indexing
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    deg = adj.sum(axis=1)
    
    def simulate_fast(s):
        """Fast greedy simulation using numpy boolean array."""
        order = np.argsort(-s)
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count
    
    # 1. Initialization
    # Start with negative degree, which is a strong heuristic for sparse graphs
    # Add small random noise to break symmetries
    rng = np.random.default_rng(42)
    best_scores = -deg.astype(float) + rng.normal(0, 0.01, n)
    best_val = simulate_fast(best_scores)
    
    # 2. Simulated Annealing on Continuous Scores
    # We optimize the scores directly.
    # Move: Add Gaussian noise to a random subset of vertices or all vertices.
    # To be efficient, we perturb all vertices slightly.
    
    max_time = 110.0
    T = 1.0
    cooling_rate = 0.9995
    sigma = 0.1
    
    current_scores = best_scores.copy()
    current_val = best_val
    
    while time.time() - start_time < max_time:
        # Generate new scores
        # Perturb all scores with small noise
        new_scores = current_scores + rng.normal(0, sigma, n)
        new_val = simulate_fast(new_scores)
        
        # Metropolis Acceptance
        delta = new_val - current_val
        if delta >= 0:
            current_scores = new_scores
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_scores = new_scores.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative integer, so exp(negative) is small
            if rng.random() < np.exp(delta / T):
                current_scores = new_scores
                current_val = new_val
        
        # Cool down
        T *= cooling_rate
        sigma *= 0.9999
        
        # If T is too low, restart with higher T to escape local optima
        if T < 0.01:
            T = 0.1
            sigma = 0.05
            # Optionally restart from best
            current_scores = best_scores + rng.normal(0, 0.05, n)
            current_val = simulate_fast(current_scores)
            
    # 3. Final Local Search (Random Swaps)
    # Fine-tune the discrete ordering
    max_time_end = 118.0
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() - start_time < max_time_end and no_improve < max_no_improve:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Swap scores
        best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
        new_val = simulate_fast(best_scores)
        
        if new_val > best_val:
            best_val = new_val
            no_improve = 0
        else:
            # Revert
            best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
            no_improve += 1
            
    return best_scores
