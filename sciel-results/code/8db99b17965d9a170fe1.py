import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    # 150s limit, use 145s for safety
    DEADLINE = 145.0
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Symmetrize and zero out diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbors as lists of integers
    neighbors = [np.where(A_bool[i])[0].tolist() for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=float)
    
    # --- 1. Generate Initial Scores using "Significance" Heuristic ---
    # Theoretical basis: Low degree vertices are "cheaper" to include.
    # We want high scores for low degree vertices.
    # Also, vertices connected to other low degree vertices are better.
    
    # Base score: Inverse degree (smoothed)
    inv_deg = 1.0 / (degrees + 1.0)
    
    # Neighborhood score: Average inverse degree of neighbors
    # score_nb(v) = sum_{u in N(v)} inv_deg[u]
    # Use matrix multiplication for speed
    nb_inv_deg_sum = A @ inv_deg
    # Average or Sum? Sum captures "total blocked value".
    # Let's normalize by degree to get average, or keep sum for "domination".
    # For MIS, blocking high-value neighbors is bad.
    # So, high inv_deg neighbors are "good" (they are cheap to block? No, they are cheap to include).
    # If I pick v, I block u. If u is "cheap" (low deg), I didn't lose much potential.
    # So, picking v is better if its neighbors are "cheap" (low degree).
    # Thus, high sum of inv_deg in neighborhood is GOOD.
    
    nb_score = nb_inv_deg_sum / (degrees + 1.0)  # Average inverse degree of neighbors
    
    # Combine
    w1, w2 = 1.0, 0.5
    raw_scores = w1 * inv_deg + w2 * nb_score
    
    # Normalize to [0, 1] range for stability in noise addition
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones(N) / N

    # --- 2. Fast Simulation Function ---
    def eval_mis_size(s):
        """
        Greedy decoder: pick highest score, remove it and its neighbors.
        Returns count of picked vertices.
        """
        # Stable sort: higher scores first.
        # If scores are tied, stable sort preserves original order (0..N-1).
        order = np.argsort(-s, kind='stable')
        
        active = [True] * N
        count = 0
        
        for v in order:
            v = int(v)
            if not active[v]:
                continue
            count += 1
            active[v] = False
            # Deactivate neighbors
            for u in neighbors[v]:
                if active[u]:
                    active[u] = False
        return count

    # --- 3. Local Search: Simulated Annealing ---
    
    best_scores = scores.copy()
    best_val = eval_mis_size(best_scores)
    
    rng = np.random.default_rng(42)
    
    # Parameters
    num_restarts = 25
    current_temp = 1.0
    cooling_rate = 0.98
    
    for r in range(num_restarts):
        if time.time() > DEADLINE - 2:
            break
            
        if r == 0:
            cur_scores = best_scores.copy()
            cur_val = best_val
        else:
            # Perturb the best known solution
            noise_scale = 0.1 + 0.05 * r
            noise = rng.normal(0, noise_scale, N)
            cur_scores = best_scores + noise
            cur_val = eval_mis_size(cur_scores)
            
        temp = current_temp
        no_improve = 0
        max_iters = 20000  # Cap iterations to ensure we check time often
        
        while time.time() < DEADLINE - 3 and no_improve < max_iters:
            # Move: Swap two random vertices in the score vector
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
                
            # Swap
            cur_scores[i], cur_scores[j] = cur_scores[j], cur_scores[i]
            new_val = eval_mis_size(cur_scores)
            
            delta = new_val - cur_val
            
            # Acceptance probability
            if delta > 0 or (temp > 1e-4 and rng.random() < np.exp(delta / temp)):
                cur_val = new_val
                no_improve = 0
                if cur_val > best_val:
                    best_val = cur_val
                    best_scores = cur_scores.copy()
            else:
                # Reject, revert swap
                cur_scores[i], cur_scores[j] = cur_scores[j], cur_scores[i]
                no_improve += 1
                
            temp *= cooling_rate
            
        # Reduce temperature for next restart
        current_temp *= 0.85
        if current_temp < 0.01:
            current_temp = 0.01

    # --- 4. Final Polish: Hill Climbing (Strict Improvement) ---
    if time.time() < DEADLINE - 1:
        polish_scores = best_scores.copy()
        polish_val = best_val
        rng_polish = np.random.default_rng(99)
        no_improve_polish = 0
        max_polish_iters = 10000
        
        while time.time() < DEADLINE - 0.5 and no_improve_polish < max_polish_iters:
            i, j = rng_polish.integers(0, N, size=2)
            if i == j:
                continue
                
            polish_scores[i], polish_scores[j] = polish_scores[j], polish_scores[i]
            new_val = eval_mis_size(polish_scores)
            
            if new_val > polish_val:
                polish_val = new_val
                no_improve_polish = 0
                if polish_val > best_val:
                    best_val = polish_val
                    best_scores = polish_scores.copy()
            else:
                polish_scores[i], polish_scores[j] = polish_scores[j], polish_scores[i]
                no_improve_polish += 1

    return best_scores
