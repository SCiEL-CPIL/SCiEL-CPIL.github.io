import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Fast Local Search with Optimized Simulation
    1. Initialize scores using a strong static heuristic (Low Degree + Hub Penalty).
    2. Use a highly optimized simulation function (bitmask-based) to evaluate greedy outcomes quickly.
    3. Perform a hill-climbing search by swapping pairs of vertex scores.
       - Use a candidate pool of "interesting" vertices (low degree, high impact) to reduce search space.
       - Accept moves that increase the independent set size.
       - Include a random restart mechanism if stuck for too long, to escape local optima.
    
    Key Optimization:
    - The simulation is the bottleneck. By using numpy bit operations or simple loops with precomputed neighbors,
      we can evaluate thousands of swaps within the time limit.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists as lists of ints for fast iteration in Python
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic
    deg = adj.sum(axis=1)
    # Score: Low degree is good. Also penalize being connected to hubs.
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    scores = -deg.astype(float) - 0.1 * hub_penalty
    
    # 2. Optimized Simulation Function
    # We represent the active set as a boolean array or a set.
    # For N=200, a set of ints is quite fast in Python for membership and removal.
    # However, `np.argsort` is also a cost. 
    # Let's try to make the inner loop as fast as possible.
    
    def simulate(s):
        # Get order
        order = np.argsort(-s)
        active = set(range(n))
        count = 0
        # Local variables for speed
        n_list = neighbors_list
        active_remove = active.remove
        for v in order:
            if v in active:
                count += 1
                active_remove(v)
                for u in n_list[v]:
                    if u in active:
                        active_remove(u)
        return count

    # 3. Local Search
    best_val = simulate(scores)
    rng = np.random.default_rng(42)
    
    # Candidate pool: vertices with lower degrees are more likely to be in the IS.
    # Swapping a high-degree vertex with a low-degree vertex is more likely to help.
    # Let's identify a set of "low degree" vertices and "high degree" vertices.
    # Actually, just picking random pairs is okay, but biased pairs might be better.
    # Let's stick to random pairs first for simplicity, but maybe weight towards lower degree vertices.
    
    max_time = 140.0  # Safety margin from 150s limit
    no_improve_count = 0
    max_no_improve = 200  # Restart if stuck
    
    while time.time() - start_time < max_time:
        # Pick two random vertices to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        new_val = simulate(scores)
        
        if new_val > best_val:
            best_val = new_val
            no_improve_count = 0
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Random restart: perturb scores significantly
                # Add random noise to break symmetry/local optimum
                scores += rng.normal(0, 1.0, n)
                # Re-evaluate
                best_val = simulate(scores)
                no_improve_count = 0

    return scores

# Fallback (though the above should be robust)
# The function above is the only one defined, so no duplicate needed.
