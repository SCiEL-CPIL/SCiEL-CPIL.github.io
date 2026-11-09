import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in an ACO for MCKP.
    Strategy: Learn a linear combination of features using Simulated Annealing
    to maximize the best ACO performance.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are correct: (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
        
    n_items = P.shape[0]
    n_constraints = w.shape[0]
    capacity = 25.0
    
    # Time management
    t0 = time.time()
    deadline = t0 + 300.0  # Leave 30s buffer for the 330s kill
    
    # Feature Engineering
    # We want a flexible heuristic.
    # Features:
    # 1. Prize
    # 2. Total Volume (sum of weights)
    # 3. Max Component Weight
    # 4. Prize / Total Volume
    # 5. Prize / Max Component
    # 6. 1 / Total Volume (penalty for large items)
    
    V_sum = w.sum(axis=0)
    V_max = w.max(axis=0)
    
    # Normalize features to [0, 1] to stabilize optimization
    P_norm = P / (P.max() + 1e-9)
    V_sum_norm = V_sum / (V_sum.max() + 1e-9)
    V_max_norm = V_max / (V_max.max() + 1e-9)
    
    # Create feature matrix (n_features, n_items)
    # We will compute h = exp(w_feat @ features) or similar?
    # Let's stick to a positive linear combination in log-space or just positive linear.
    # h[i] = exp( sum_k alpha_k * feature_k[i] )
    # This ensures h[i] > 0 always.
    
    features = np.vstack([
        np.ones(n_items),        # Bias
        P_norm,                  # Prize
        -V_sum_norm,             # Neg Volume (penalty)
        -V_max_norm,             # Neg Max Comp (penalty)
        P_norm / (V_sum_norm + 0.1), # Ratio
        P_norm / (V_max_norm + 0.1)  # Ratio Max
    ])
    
    n_features = features.shape[0]
    
    # Fast ACO Simulation
    # We need this to be very fast.
    def fast_colony_sim(h, seed):
        """
        Simulates the ACO process.
        Returns the best total prize found.
        """
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n_items)
        best_total = 0.0
        
        for _ in range(30):
            max_ant_total = 0.0
            for _ in range(20):
                remaining_cap = np.full(n_constraints, capacity)
                used = np.zeros(n_items, dtype=bool)
                total_prize = 0.0
                
                while True:
                    # Check which items fit
                    # w is (5, 100), remaining_cap is (5,)
                    fits_mask = np.all(w <= remaining_cap[:, None], axis=0) & ~used
                    
                    if not np.any(fits_mask):
                        break
                    
                    fitting_indices = np.where(fits_mask)[0]
                    
                    # Calculate probabilities: pheromone * heuristic
                    probs = pheromone[fitting_indices] * h[fitting_indices]
                    sum_probs = probs.sum()
                    if sum_probs < 1e-9:
                        probs = np.ones(len(fitting_indices))
                        sum_probs = len(fitting_indices)
                    probs = probs / sum_probs
                    
                    # Select one item
                    idx = rng.choice(fitting_indices, p=probs)
                    used[idx] = True
                    remaining_cap -= w[:, idx]
                    total_prize += P[idx]
                
            if total_prize > max_ant_total:
                max_ant_total = total_prize
                
            if max_ant_total > best_total:
                best_total = max_ant_total
                
            # Update pheromone
            pheromone = pheromone * 0.9 + max_ant_total / 10.0
            
        return best_total

    # Simulated Annealing Optimization
    # Objective: Maximize best_total from fast_colony_sim
    
    # Initialize alpha (weights for features)
    # Start with a reasonable guess: high bias, positive prize, negative volume
    alpha = np.array([0.0, 1.0, -1.0, -0.5, 1.0, 0.5])
    
    best_alpha = alpha.copy()
    best_h = np.exp(features.T @ alpha)
    best_h = np.clip(best_h, 1e-9, 1e6)
    best_score = fast_colony_sim(best_h, 42)
    
    current_alpha = alpha.copy()
    current_h = np.exp(features.T @ current_alpha)
    current_h = np.clip(current_h, 1e-9, 1e6)
    current_score = best_score
    
    T = 1.0  # Initial temperature
    cooling_rate = 0.95
    min_T = 0.01
    
    step_size = 0.5
    
    while T > min_T:
        if time.time() > deadline - 20.0:
            break
            
        # Propose a new alpha
        delta_alpha = np.random.randn(n_features) * step_size
        new_alpha = current_alpha + delta_alpha
        
        # Compute new heuristic
        new_h = np.exp(features.T @ new_alpha)
        new_h = np.clip(new_h, 1e-9, 1e6)
        
        # Evaluate
        new_score = fast_colony_sim(new_h, 42)
        
        # Acceptance probability
        if new_score > current_score:
            current_alpha = new_alpha
            current_h = new_h
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_alpha = new_alpha.copy()
                best_h = new_h.copy()
        else:
            # Metropolis criterion
            # Note: We are maximizing, so delta is new - current (negative)
            # We want to accept if exp(delta/T) > random
            # delta is negative, so exp(delta/T) < 1
            delta = new_score - current_score
            if np.log(np.random.rand()) < delta / T:
                current_alpha = new_alpha
                current_h = new_h
                current_score = new_score
        
        T *= cooling_rate
        # Reduce step size slightly
        step_size *= 0.99
        
    # Final verification and return
    # Ensure the returned heuristic is the best one found
    # Re-compute to be safe against any state issues
    final_h = np.exp(features.T @ best_alpha)
    final_h = np.where(final_h > 0, final_h, 1e-9)
    final_h = np.clip(final_h, 1e-9, 1e6)
    
    return final_h
