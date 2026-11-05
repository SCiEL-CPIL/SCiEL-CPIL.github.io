"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy:
    1. Initialize scores using a structural heuristic that favors low-degree nodes 
       with high neighbor-degree (hub clearing).
    2. Use a fast local search (Simulated Annealing with swap moves) to directly 
       maximize the size of the independent set produced by the greedy decoder.
       This combines the robust structural initialization of the current program 
       with the optimization machinery from the high-scoring inspirations.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s limit
    
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
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- 1. Initial Heuristic ---
    # We want a strong structural prior.
    # Low degree is good (less likely to be blocked).
    # High average neighbor degree is good (removes hubs efficiently).
    
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors (safe division)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Heuristic: 
    # Score = -w1 * deg - w2 * avg_neighbor_deg
    # Based on inspiration 1 (score 1.1541) which used -deg - 0.5 * avg_neighbor_deg
    # We will use this as the starting point.
    
    raw_scores = -(deg + 0.5 * avg_neighbor_deg)
    
    # Normalize scores to [0, 1] to make the annealing process more stable and 
    # to ensure that swaps have a meaningful impact on the ordering.
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.zeros(N)
    
    # --- 2. Fast Simulation Function ---
    def simulate(s):
        """
        Simulates the greedy decoder and returns the size of the independent set.
        """
        # Get the order of vertices by descending score
        order = np.argsort(-s)
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                # Deactivate neighbors
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
                    
        return count
    
    # --- 3. Local Search (Simulated Annealing) ---
    # We use simulated annealing to escape local minima in the score space.
    # The objective is to maximize the independent set size.
    
    best_size = simulate(scores)
    current_size = best_size
    current_scores = scores.copy()
    
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    max_time = DEADLINE - start_time
    
    # We will run the loop until the deadline is reached.
    # We need to be careful with the cooling schedule.
    # Let's estimate the number of steps we can do.
    # For N=200, one simulation is ~200 operations. 
    # Python overhead is significant. Let's just run as many as we can.
    
    last_step_time = time.time()
    steps = 0
    total_steps_estimate = 50000 # Just a guess for cooling schedule, will adjust dynamically
    
    while time.time() - start_time < DEADLINE:
        # Calculate current temperature
        elapsed = time.time() - start_time
        # Ensure progress is between 0 and 1
        progress = min(1.0, elapsed / max_time)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick two random distinct vertices to swap
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
            
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        # Evaluate
        new_size = simulate(current_scores)
        
        delta = new_size - current_size
        
        if delta > 0:
            # Accept better move
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                scores = current_scores.copy()
        else:
            # Accept worse move with probability exp(delta / T)
            # delta is negative or zero
            if T > 1e-9:
                # Avoid overflow/underflow issues
                if delta / T < -100:
                    prob = 0.0
                else:
                    prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if rng.random() < prob:
                current_size = new_size
                # Do not update best_size if worse
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                
        steps += 1
        
        # Safety check: if we are getting close to the deadline, break
        if time.time() - start_time > DEADLINE - 2.0:
            break

    return scores
# EVOLVE-BLOCK-END
