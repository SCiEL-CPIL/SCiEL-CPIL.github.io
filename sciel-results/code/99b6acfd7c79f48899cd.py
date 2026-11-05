import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulated Annealing over Priority Vectors
    1. Initialize scores using a strong heuristic (Low Degree + Hub Penalty).
    2. Use Simulated Annealing to optimize the scores.
       - Move: Perturb a random subset of vertices' scores (or swap a random pair, 
         but SA allows 'bad' moves to escape local minima).
       - Actually, swapping pairs is a discrete move. A better move for SA in this 
         context might be 'reversing' a segment of the order or 'shifting' a vertex.
         However, the decoder relies on relative ordering. 
         Let's stick to swaps but allow accepting worse swaps based on temperature.
         Alternatively, we can treat the scores as continuous and use Gaussian noise,
         then re-sort. But the decoder only cares about the order.
         So, the state space is permutations.
         A good move for permutations: Pick two random indices i, j and swap the 
         elements. 
         To make it more powerful than random walk, we can also try 'block moves' 
         (reverse a segment) or 'insertion moves' (move one element to another position).
         Let's use a mix: 50% swap, 50% insertion (move v to position u).
         
    3. Acceptance criterion: Metropolis criterion.
       - If new_size > old_size: Accept.
       - If new_size < old_size: Accept with probability exp((new_size - old_size) / T).
       - Since size is integer, diff is at least 1. 
       - T starts high (e.g., 5.0) and cools down.
       
    4. Time management: Run until time limit (140s), keeping track of best solution.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Pre-compute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # Initial Heuristic: Low Degree + Hub Penalty
    deg = adj.sum(axis=1)
    scores = -deg.astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    scores = scores - 0.1 * hub_penalty
    
    def simulate(s):
        """
        Simulate greedy decoder with given scores.
        Returns the size of the independent set.
        """
        order = np.argsort(-s)
        active_set = set(range(n))
        count = 0
        for v in order:
            if v in active_set:
                count += 1
                active_set.remove(v)
                for u in neighbors_list[v]:
                    if u in active_set:
                        active_set.remove(u)
        return count
    
    # Get initial best
    best_scores = scores.copy()
    best_val = simulate(best_scores)
    
    # Annealing parameters
    T_start = 2.0  # Since delta is integer, exp(-1/2) ~ 0.6, exp(-1/1) ~ 0.37
    T_end = 0.01
    max_time = 145.0  # Safety margin from 150s limit
    
    rng = np.random.default_rng(42)
    
    # We need a cooling schedule.
    # Let's estimate the number of steps we can do.
    # One step: argsort (N log N), set operations (O(N * avg_deg)).
    # For N=200, this is very fast. Maybe 0.1ms per step?
    # 145s / 0.0001s = 1,450,000 steps.
    # We can use a geometric cooling schedule based on steps, or just linear time-based.
    # Let's use time-based cooling to be safe.
    
    total_duration = max_time - start_time
    if total_duration <= 0:
        return best_scores
        
    # We will run steps until time limit.
    # T(t) = T_start * (T_end / T_start) ** (t / total_duration)
    # Or simply T = T_start * (1 - t/total_duration) ** 2
    
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        progress = elapsed / total_duration
        if progress >= 1.0:
            break
            
        # Geometric cooling
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a move
        # Option 1: Swap two random vertices
        # Option 2: Move a vertex to a random position (Insertion)
        # Insertion is often better for permutation optimization.
        # Let's do: Pick random i, j. If i == j, skip.
        # Swap scores[i] and scores[j]? No, that changes the order.
        # Actually, the scores define the order. Swapping scores is equivalent to swapping positions in the sorted list?
        # Not exactly. If scores are distinct, swapping scores changes the relative order of i and j,
        # but also potentially changes their position relative to others if the scores are close?
        # No, if we just swap the values, the sorted order changes.
        # Let's just manipulate the scores directly.
        
        # A better move for the score vector:
        # Pick two random vertices u, v.
        # Swap their scores.
        u, v = rng.integers(0, n, size=2)
        if u == v:
            continue
            
        # Save current scores
        su = scores[u]
        sv = scores[v]
        
        # Apply move
        scores[u] = sv
        scores[v] = su
        
        # Evaluate
        new_val = simulate(scores)
        delta = new_val - best_val
        
        if delta >= 0:
            # Accept
            best_scores = scores.copy()
            best_val = new_val
        else:
            # Probabilistic accept
            # delta is negative integer.
            # Prob = exp(delta / T)
            # If T is very small, prob is ~0.
            if T > 0.001:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    # Accept worse solution
                    pass # Scores are already updated
                else:
                    # Revert
                    scores[u] = su
                    scores[v] = sv
            else:
                # Revert
                scores[u] = su
                scores[v] = sv

    return best_scores

# Fallback for safety (in case of errors)
def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    scores = -deg.astype(float)
    neighbor_deg_sum = adj @ deg
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    scores = scores - 0.1 * hub_penalty
    
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    def simulate(s):
        order = np.argsort(-s)
        active_set = set(range(n))
        count = 0
        for v in order:
            if v in active_set:
                count += 1
                active_set.remove(v)
                for u in neighbors_list[v]:
                    if u in active_set:
                        active_set.remove(u)
        return count
    
    best_scores = scores.copy()
    best_val = simulate(best_scores)
    
    T_start = 2.0
    T_end = 0.01
    max_time = 145.0
    
    rng = np.random.default_rng(42)
    total_duration = max_time - start_time
    if total_duration <= 0:
        return best_scores
        
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        progress = elapsed / total_duration
        if progress >= 1.0:
            break
            
        T = T_start * (T_end / T_start) ** progress
        
        u, v = rng.integers(0, n, size=2)
        if u == v:
            continue
            
        su = scores[u]
        sv = scores[v]
        
        scores[u] = sv
        scores[v] = su
        
        new_val = simulate(scores)
        delta = new_val - best_val
        
        if delta >= 0:
            best_scores = scores.copy()
            best_val = new_val
        else:
            if T > 0.001:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    pass
                else:
                    scores[u] = su
                    scores[v] = sv
            else:
                scores[u] = su
                scores[v] = sv

    return best_scores
