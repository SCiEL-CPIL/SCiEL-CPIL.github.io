import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Precompute neighbor indices for fast access in greedy evaluation
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order (array of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count
    
    def evaluate_scores(scores):
        """Evaluate the greedy MIS size for a given score vector."""
        # Sort vertices by descending score
        order = np.argsort(-scores)
        return greedy_size_from_order(order)
    
    # Initial scores: Random perturbation around -degree to break symmetry
    # Low degree is generally good for MIS, so -degree is a reasonable baseline
    scores = -degrees.copy() + np.random.uniform(-0.1, 0.1, n)
    
    current_size = evaluate_scores(scores)
    best_size = current_size
    best_scores = scores.copy()
    
    # Simulated Annealing Parameters
    # Temperature starts high enough to accept bad moves, decays over time
    T_start = 1.0
    T_end = 0.01
    # We don't know exactly how many steps we can take, so we'll iterate until time runs out
    # But we need to decay T. Let's estimate max steps.
    # Each step involves sorting (N log N) and greedy pass (N). N=200.
    # 200 * log(200) ~ 1500 ops. Greedy pass ~ 200 ops. Total ~ 2000 ops per step.
    # 140s / 2000 ops = 70,000 steps if 1 op/us. But Python overhead is high.
    # Let's assume ~1000 steps/sec. 140s -> 140,000 steps.
    # Let's just run a fixed number of iterations per "batch" and check time.
    
    # To make T decay smooth, we can use a logarithmic or linear decay based on iteration count.
    # Since we don't know exact speed, we'll use a simple heuristic:
    # Run in chunks.
    
    current_T = T_start
    cooling_rate = 0.999 # Decay per iteration
    
    iter_count = 0
    
    while time.time() < deadline:
        if current_T < T_end:
            current_T = T_end
            
        # Generate a random perturbation to the scores
        # Perturb a random subset of vertices to allow for larger jumps
        # Or perturb all vertices with small noise?
        # Let's perturb a random single vertex significantly, or a small set.
        # Perturbing all vertices with small noise is like a random walk in high-dim space.
        # Perturbing one vertex is more local.
        # Let's try perturbing a random vertex's score by a random amount.
        
        idx = np.random.randint(n)
        delta = np.random.uniform(-1.0, 1.0)
        new_scores = scores.copy()
        new_scores[idx] += delta
        
        new_size = evaluate_scores(new_scores)
        
        # Calculate energy difference
        # We want to MAXIMIZE size, so "energy" is -size
        # delta_E = -new_size - (-current_size) = current_size - new_size
        delta_E = current_size - new_size
        
        # Acceptance probability
        if delta_E < 0:
            # Improvement
            accept = True
        else:
            # Worse
            # P = exp(-delta_E / T)
            # If delta_E is large and T is small, P is small
            # Note: delta_E is in units of "set size", which is small integers.
            # T is in same units?
            # If T=1.0, delta_E=1, P = exp(-1) ~ 0.37
            # If T=0.1, delta_E=1, P = exp(-10) ~ 0
            # This seems reasonable.
            p = np.exp(-delta_E / current_T)
            accept = np.random.random() < p
        
        if accept:
            scores = new_scores
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_scores = scores.copy()
        else:
            # Reject, keep old scores
            pass
        
        current_T *= cooling_rate
        iter_count += 1
        
        # Optional: Check time more frequently if we want to be safer
        if iter_count % 100 == 0:
            if time.time() > deadline:
                break
    
    return best_scores.astype(float)
