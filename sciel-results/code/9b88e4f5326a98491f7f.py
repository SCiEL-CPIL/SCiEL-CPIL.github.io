"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _greedy_decode(adjacency, scores):
    """
    Run the fixed greedy decoder.
    Returns: set of selected vertices (indices)
    """
    n = adjacency.shape[0]
    # Order vertices by descending score
    # In case of ties, we need a deterministic tie-breaker to ensure reproducibility 
    # during the optimization phase, usually index or random. 
    # The task says "fixed greedy decoder", implying the decoder is fixed given the scores.
    # We assume standard sort behavior.
    order = np.argsort(-scores, kind='stable')
    
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        # Mark neighbors as unavailable
        # adjacency[v] is a boolean vector of neighbors
        available[adjacency[v]] = False
        
    return selected

def _evaluate_score(adjacency, scores):
    """
    Calculate the size of the independent set produced by the scores.
    """
    selected = _greedy_decode(adjacency, scores)
    return np.sum(selected)

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    
    Strategy: Iterative Local Search on the Score Vector.
    1. Start with a heuristic (e.g., negative degree).
    2. Simulate greedy decoding.
    3. Identify opportunities to improve the set size by swapping priorities 
       or adjusting scores of specific vertices.
    4. Update scores.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin from 120s limit
    
    n = adjacency.shape[0]
    
    # 1. Initial Heuristic: Negative degree (low degree first)
    # This is a good starting point.
    degrees = adjacency.sum(axis=1).astype(float)
    best_scores = -degrees
    best_set_size = _evaluate_score(adjacency, best_scores)
    
    # 2. Local Search Loop
    # We will try to perturb the scores and see if we get a better set.
    # A common effective move is "2-opt" style swaps or small random perturbations.
    # However, since the decoder is deterministic given scores, we can analyze 
    # the "critical path" of the greedy choice.
    
    # Let's use a simulated annealing-like approach on the score vector.
    # But pure random walk is slow. 
    # Better: Identify vertices that are "blocking" many others.
    
    # Alternative Strategy: 
    # The greedy algorithm picks v if it has the highest score among available.
    # If picking v prevents picking u, w, x (3 neighbors), but u, w, x could have been picked 
    # if v was skipped, we might want to lower v's score.
    
    # Let's implement a heuristic-based local search:
    # 1. Get current set S.
    # 2. For each v in S, check if removing v (by lowering its score) allows 
    #    more neighbors to be added.
    # 3. For each v not in S, check if raising its score allows it to be picked 
    #    and whether it blocks too many others.
    
    # To make it fast, we'll do a few iterations of "swap" based on local benefit.
    
    current_scores = best_scores.copy()
    current_size = best_set_size
    
    # We will perform a limited number of steps.
    # Each step involves analyzing the current greedy trace.
    
    for step in range(500): # Max iterations
        if time.time() > deadline:
            break
            
        # Get the order of selection
        order = np.argsort(-current_scores, kind='stable')
        selected = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        # Store the "effective" score at time of selection? No, just the order.
        
        # To find improvements, we can look at pairs (i, j) where i is picked before j,
        # but j is a neighbor of i.
        # If we swap i and j, does the set size increase?
        
        # Efficient approach: 
        # Simulate the greedy process and record for each picked vertex v, 
        # how many vertices it "blocks" (neighbors that become unavailable).
        # If a vertex v blocks many vertices, it might be a "bad" pick if those 
        # blocked vertices could have formed a larger set among themselves.
        
        # Let's try a simpler heuristic: 
        # Randomly swap two vertices in the priority order and check if size improves.
        # This is a form of local search.
        
        # To be faster, we can restrict swaps to "close" vertices in the order 
        # or vertices involved in conflicts.
        
        # Let's try: Pick a random vertex v from the selected set. 
        # Lower its score slightly to push it down. See if the set size increases.
        # Or pick a vertex u NOT in the set, raise its score.
        
        # Actually, a very effective heuristic for Max Independent Set with Greedy is:
        # "Weighted Degree" or "Local Density".
        # Let's try to optimize scores towards a combination of:
        # - Degree (lower is better)
        # - Neighbor Degree Sum (lower is better, i.e., neighbors are low degree)
        
        # Let's compute a static "quality" score and blend it.
        # Quality(v) = -degree(v) - alpha * sum(degrees(neighbors))
        
        # But the task asks for a *different strategy* from just tuning static heuristics.
        # The local search IS the different strategy.
        
        # Let's implement a specific local move:
        # 1. Find the vertex v in the current selected set that has the highest 
        #    "opportunity cost" (number of neighbors it blocks that are NOT selected).
        # 2. Find a neighbor u of v that is NOT selected.
        # 3. Try swapping v and u in the priority order (or adjusting scores to achieve it).
        # 4. If the new set size is larger, accept the change.
        
        selected = _greedy_decode(adjacency, current_scores)
        
        # Find candidates for improvement
        # For each selected vertex v, count unselected neighbors.
        # If v has many unselected neighbors, it's "wasting" its slot.
        
        # We will do a limited number of random perturbations per step to avoid O(n^2)
        # full re-evaluations if n is large, but n=200 is small.
        # n=200, n^2=40000. 500 iterations * 40000 ops is fine in Python? 
        # _greedy_decode is O(n^2) worst case. 500 * 40000 = 20M ops. Should be < 1s.
        
        # Let's do a hill-climbing step:
        improved = False
        # Try swapping a random pair of vertices in the score vector
        # To be smarter, bias towards vertices that are "contested".
        
        # Simple Random Swap Hill Climbing:
        # Pick two random indices i, j.
        # Swap scores[i] and scores[j].
        # Evaluate new size.
        # If better, keep. If worse, revert.
        
        # To make it more directed, we can swap a selected vertex with an unselected neighbor.
        selected_indices = np.where(selected)[0]
        unselected_indices = np.where(~selected)[0]
        
        if len(selected_indices) > 0 and len(unselected_indices) > 0:
            # Pick a random selected vertex
            v_idx = np.random.choice(selected_indices)
            # Get its neighbors
            neighbors = np.where(adjacency[v_idx])[0]
            # Filter neighbors that are unselected
            unselected_neighbors = neighbors[~selected[neighbors]]
            
            if len(unselected_neighbors) > 0:
                # Pick a random unselected neighbor
                u_idx = np.random.choice(unselected_neighbors)
                
                # Try swapping their scores
                old_sv = current_scores[v_idx]
                old_su = current_scores[u_idx]
                
                # Swap
                current_scores[v_idx] = old_su
                current_scores[u_idx] = old_sv
                
                new_size = _evaluate_score(adjacency, current_scores)
                
                if new_size > current_size:
                    current_size = new_size
                    if new_size > best_set_size:
                        best_set_size = new_size
                        best_scores = current_scores.copy()
                    improved = True
                else:
                    # Revert
                    current_scores[v_idx] = old_sv
                    current_scores[u_idx] = old_su
        
        # Occasionally do a random perturbation to escape local minima
        if not improved and np.random.random() < 0.1:
            i, j = np.random.choice(n, 2, replace=False)
            old_si = current_scores[i]
            old_sj = current_scores[j]
            current_scores[i] = old_sj
            current_scores[j] = old_si
            new_size = _evaluate_score(adjacency, current_scores)
            if new_size > current_size:
                current_size = new_size
                if new_size > best_set_size:
                    best_set_size = new_size
                    best_scores = current_scores.copy()
            else:
                current_scores[i] = old_si
                current_scores[j] = old_sj

    return best_scores

# EVOLVE-BLOCK-END
