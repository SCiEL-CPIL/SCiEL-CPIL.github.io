import numpy as np
import time

def _greedy_mis(adjacency, order):
    """Run greedy MIS with given vertex order (descending priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency[v]] = False
    return count

def _fast_greedy_size(adjacency, available_mask):
    """
    Estimate MIS size of the subgraph defined by available_mask 
    using a simple Min-Degree heuristic.
    """
    n = adjacency.shape[0]
    avail = available_mask.copy()
    avail_indices = np.where(avail)[0]
    if len(avail_indices) == 0:
        return 0
        
    # Compute degrees in subgraph
    degs = np.sum(adjacency[:, avail], axis=1)
    degs = degs[avail_indices]
    
    # Sort available indices by degree (ascending)
    sorted_indices = avail_indices[np.argsort(degs)]
    
    # Greedy simulation on this order
    avail_sim = np.zeros(n, dtype=bool)
    avail_sim[avail] = True
    
    cnt = 0
    for v in sorted_indices:
        if avail_sim[v]:
            cnt += 1
            avail_sim[v] = False
            avail_sim[adjacency[v]] = False
            
    return cnt

def priority(adjacency):
    n = adjacency.shape[0]
    deadline = time.time() + 110 # Safety margin
    
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Candidate 1: Min Degree (Standard)
    score_min_deg = -degrees
    
    # Candidate 2: Min Degree + Neighbor Sum
    neighbor_deg_sum = adjacency.dot(degrees)
    
    candidates = []
    
    # Add Min Degree
    candidates.append(score_min_deg)
    
    # Add variations of the Neighbor Sum heuristic
    for lam in [0.1, 0.5, 1.0, 2.0]:
        score = -degrees + lam * neighbor_deg_sum
        candidates.append(score)
        
    # Add variations that penalize neighbor sum
    for lam in [-0.5, -1.0]:
        score = -degrees + lam * neighbor_deg_sum
        candidates.append(score)

    # Candidate: 1-Step Lookahead
    if time.time() < deadline - 15:
        lookahead_scores = np.zeros(n)
        for v in range(n):
            mask = np.ones(n, dtype=bool)
            mask[v] = False
            mask[adjacency[v]] = False
            est_size = _fast_greedy_size(adjacency, mask)
            lookahead_scores[v] = 1 + est_size
        candidates.append(lookahead_scores)

    # Evaluate all candidates to find the best score vector
    best_size = -1
    best_score = candidates[0]
    
    for score in candidates:
        if time.time() > deadline - 5:
            break
        order = np.argsort(-score)
        size = _greedy_mis(adjacency, order)
        if size > best_size:
            best_size = size
            best_score = score

    # Local Search Refinement (from Inspiration 1)
    # Convert best_score to an initial order
    best_order = np.argsort(-best_score)
    
    # We will try to improve this order via adjacent swaps.
    # This is a refinement pass on the specific ordering found by the global heuristics.
    improved = True
    # Limit the number of passes to ensure we finish in time
    # Each pass is O(N^2) greedy evaluations. N=200. 200*200*200 = 8M ops per pass.
    # We have ~10s left (deadline - 5). 8M ops is fast in Python/NumPy if vectorized,
    # but _greedy_mis is a Python loop. 200 iterations * 200 swaps = 40k greedy calls.
    # Each greedy call is O(N^2) worst case = 40k * 40k = 1.6B ops? No, greedy is O(N^2) total for the whole run.
    # So 40k calls * O(200^2) = 1.6B ops. Might be too slow in pure Python.
    
    # Let's estimate time for _greedy_mis.
    # N=200. Loop 200 times. Inner update is vectorized on bool array.
    # It should be very fast. Let's try a few passes.
    
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline - 2:
                break
            # Swap i and i+1
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_size = _greedy_mis(adjacency, new_order)
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True

    # Convert best_order back to scores
    # The greedy decoder picks vertices in descending score order.
    # So vertex at index 0 in best_order should have the highest score.
    # Vertex at index 1 should have the next highest, etc.
    # We can assign scores as n-1, n-2, ..., 0 or simply 1/n, 1/(n+1)... 
    # Using rank is sufficient.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
