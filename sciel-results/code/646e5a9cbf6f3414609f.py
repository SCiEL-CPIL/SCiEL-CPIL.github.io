import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = adjacency.shape[0]
    
    # Convert to adjacency lists for faster neighbor lookups
    adj_list = [np.nonzero(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(neighbors) for neighbors in adj_list])
    
    def evaluate_scores(scores):
        """
        Evaluates the size of the independent set produced by the greedy decoder
        given a vector of scores.
        The decoder picks vertices in descending score order.
        """
        # Get indices sorted by score descending
        order = np.argsort(-scores)
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Block neighbors
                for u in adj_list[v]:
                    selected[u] = True
        return count

    # Initial Scores: Low degree vertices should have higher scores to be picked first
    # This is a strong heuristic for MIS.
    # We map degrees to scores. Low degree -> High score.
    # Add some noise to break ties and allow exploration.
    initial_scores = -degrees.astype(float) + np.random.normal(0, 0.1, n)
    
    current_scores = initial_scores
    current_size = evaluate_scores(current_scores)
    best_scores = current_scores.copy()
    best_size = current_size
    
    # Simulated Annealing Parameters
    # We are optimizing a discrete objective (integer count) using continuous moves.
    # The "energy" landscape is very flat (plateaus) with sharp steps.
    # We need a temperature schedule that allows enough exploration.
    
    # Time budget for SA
    sa_deadline = start_time + 140
    
    # Initial temperature
    # Since the objective is integer, the "energy difference" is small integers.
    # T should be comparable to the typical change in size.
    # Let's start high to allow random walks, then cool.
    T = 1.0
    
    # Number of iterations per temperature step
    # Each eval is ~ n log n + E. n=200, E~20000. ~20k ops.
    # 140s / 20k ops is huge. We can do many iterations.
    # Let's estimate: 1 eval ~ 0.005s (conservative). 140s / 0.005s = 28000 evals.
    
    # We will run until time limit, cooling T exponentially.
    # alpha = cooling rate
    alpha = 0.9995
    min_T = 0.01
    
    while time.time() < sa_deadline:
        # Generate a neighbor
        # Perturb scores with Gaussian noise
        # The scale of noise matters. If too small, we get stuck. If too large, we random walk.
        # Let's adapt the noise scale slightly or keep it fixed.
        noise_scale = 0.5
        new_scores = current_scores + np.random.normal(0, noise_scale, n)
        
        new_size = evaluate_scores(new_scores)
        
        # Calculate energy difference
        # We want to maximize size, so we minimize -size.
        # Delta E = E_new - E_old = (-new_size) - (-current_size) = current_size - new_size
        delta_e = current_size - new_size
        
        # Acceptance criterion
        accept = False
        if delta_e <= 0:
            # Improvement or equal
            accept = True
        else:
            # Worse solution, accept with probability exp(-delta_e / T)
            # delta_e is positive here
            prob = np.exp(-delta_e / T)
            if np.random.rand() < prob:
                accept = True
        
        if accept:
            current_scores = new_scores
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_scores = current_scores.copy()
                
        # Cool down
        T *= alpha
        if T < min_T:
            # Reheat or reset?
            # If we cool too much, we stop exploring.
            # Let's reset T to a moderate value to allow escaping local minima.
            T = 0.5
            # Also, we can restart from best_scores to avoid drifting too far
            # But SA usually stays in the current state.
            # Let's bias towards best scores occasionally.
            if np.random.rand() < 0.1:
                current_scores = best_scores.copy()
                current_size = best_size

    return best_scores
