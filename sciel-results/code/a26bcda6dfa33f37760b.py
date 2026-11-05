import numpy as np
import time
import random

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Local Search Optimization
    1. Initialize scores using a strong heuristic: Lowest Degree First (LDF).
       This provides a good starting point for the independent set.
    2. Perform local search by iteratively swapping the priority scores of pairs of vertices.
    3. For each potential swap, simulate the greedy decoder to calculate the resulting
       independent set size.
    4. Accept the swap if it increases the independent set size.
    5. Continue until no improvements are found or a time limit is reached.
    
    The final scores are the ones that yielded the best independent set found.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Initialize scores: Lower degree -> Higher priority (higher score value)
    # We use negative degree so that lower degree vertices have higher scores.
    # Adding a small random noise to break ties and encourage exploration.
    deg = adj.sum(axis=1).astype(float)
    scores = -deg + np.random.uniform(0, 0.1, n)
    
    # Helper function to evaluate the greedy decoder
    def get_is_size(score_vec):
        # Get order of vertices by descending score
        # argsort returns indices that would sort the array.
        # We want descending order, so we reverse it.
        order = np.argsort(-score_vec)
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            if not selected[v]:
                # Select v
                selected[v] = True
                count += 1
                # Mark neighbors as blocked
                # adj[v] is a boolean vector of neighbors
                # We can vectorize the blocking, but the loop over order is the bottleneck.
                # For n=200, n^2 operations is 40,000, which is fast.
                # However, checking 'if not selected[v]' inside the loop is necessary.
                
                # Optimization: Instead of updating 'selected' array immediately for neighbors,
                # we can just check if any neighbor is already selected.
                # But the standard greedy decoder skips neighbors of *picked* vertices.
                # So if v is not picked, we skip it. If v is picked, we mark its neighbors.
                
                # Let's stick to the standard logic but optimize the inner loop.
                # Actually, for n=200, a simple loop is fine.
                # adj[v] gives neighbors.
                # We need to ensure we don't pick a vertex if it's a neighbor of a previously picked one.
                # The 'selected' array tracks picked vertices.
                # When we pick v, we don't immediately mark neighbors in 'selected' as True,
                # we just need to know they are blocked.
                # A faster way: maintain a 'blocked' array.
                
                pass # Placeholder for logic refinement below
        
        # Let's implement a faster evaluator using bitsets or just efficient numpy.
        # Since n=200, we can represent the selected set as a bitset (Python int) or just a bool array.
        # The bottleneck is the O(n^2) simulation.
        
        # Re-implementation of evaluator for speed:
        order = np.argsort(-score_vec)
        selected_mask = np.zeros(n, dtype=bool)
        is_size = 0
        
        for v in order:
            if not selected_mask[v]:
                is_size += 1
                selected_mask[v] = True
                # Block neighbors
                # adj[v] is bool array. 
                # We can do: selected_mask[adj[v]] = True 
                # But wait, if a neighbor was already selected, it stays True.
                # If a neighbor was not selected, it becomes True (blocked).
                # This correctly implements "skip neighbours of picked vertices".
                selected_mask[adj[v]] = True
                
        return is_size

    # Initial evaluation
    current_size = get_is_size(scores)
    
    # Local Search
    # We will try random swaps.
    # To be efficient, we can limit the number of swaps or use a timeout.
    
    # Pre-compute adjacency list for faster neighbor access if needed, 
    # but matrix-vector or boolean indexing is fast enough for n=200.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 1000 # Stop if we hit this many swaps without improvement
    
    while improved and (time.time() - start_time) < deadline:
        improved = False
        no_improve_count = 0
        
        # Shuffle indices to avoid bias
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        # Try swapping pairs
        # We don't need to try all pairs, just a random subset each iteration
        # Or iterate through a random permutation of pairs.
        
        # Let's do a pass of random swaps
        for i in range(n):
            if (time.time() - start_time) > deadline:
                break
                
            v1 = indices[i]
            # Pick a random other vertex
            v2 = np.random.randint(0, n)
            if v1 == v2:
                continue
                
            # Perform swap
            scores[v1], scores[v2] = scores[v2], scores[v1]
            
            # Evaluate
            new_size = get_is_size(scores)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                no_improve_count = 0
            else:
                # Revert
                scores[v1], scores[v2] = scores[v2], scores[v1]
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    break
        
        # If we didn't improve in this entire pass, we might be stuck in a local optimum.
        # We can try a more aggressive perturbation (e.g., random restart of scores)
        # but for now, if no improvement in a full pass, we stop to save time.
        if not improved:
            break
            
    return scores
