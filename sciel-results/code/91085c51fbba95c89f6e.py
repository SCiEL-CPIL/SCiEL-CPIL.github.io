"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex for a fixed greedy decoder that picks
    vertices in descending score order, skipping neighbours of picked vertices.
    
    Strategy: We want to pick vertices that are 'easy' to include (low degree)
    but also 'valuable' in the sense that they don't block too many other
    promising vertices. 
    
    A strong heuristic for Maximum Independent Set is to prioritize vertices
    with low degree, but more specifically, we can use a 'core' based or
    'independent set potential' metric.
    
    A known effective heuristic is to rank by:
    1. Degree (lower is better)
    2. But also consider the structure.
    
    Let's try a more sophisticated static score:
    Score = - (Degree + alpha * (Sum of degrees of neighbors))
    
    Or even better, use the concept of 'independent set density'.
    
    Actually, a very common and effective static heuristic for MIS is to
    sort by degree ascending. The seed program does exactly this (-degree).
    
    To improve, we can look at the second-order effects. If a vertex has low
    degree, it's good. But if its neighbors have high degree, picking it
    removes high-degree vertices from the pool, which might be bad because
    high-degree vertices are hard to place later (they block many others).
    Conversely, if a vertex has high degree, it's bad to pick early.
    
    Let's try: Score = - (degree + mean_neighbor_degree)
    This penalizes vertices that are connected to hubs.
    
    Another idea: Use the eigenvector centrality or similar spectral method?
    For 200 nodes, computing eigenvalues is fast.
    However, the greedy decoder is fixed. We just need to provide the order.
    
    Let's refine the local heuristic.
    degree = A.sum(axis=1)
    neighbor_degree_sum = A @ degree
    mean_neighbor_degree = neighbor_degree_sum / (degree + 1e-9)
    
    Score = -(degree + mean_neighbor_degree)
    
    This tends to pick isolated vertices first, then vertices connected to
    low-degree vertices.
    
    Let's add a deadline guard just in case, though this is O(N^2) or O(N^3)
    depending on implementation. N=200 is tiny.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 15s margin
    
    A = np.asarray(adjacency)
    n = A.shape[0]
    
    # Basic degree
    degree = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    # A @ degree gives sum of degrees of neighbors for each vertex
    neighbor_deg_sum = A @ degree
    
    # Mean degree of neighbors (avoid division by zero)
    # If degree is 0, mean is 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degree > 0, neighbor_deg_sum / degree, 0.0)
    
    # Heuristic 1: - (degree + mean_neighbor_deg)
    # Heuristic 2: - (degree * (1 + mean_neighbor_deg))
    
    # Let's try a combination that emphasizes low degree heavily.
    # The seed is just -degree.
    # Adding mean_neighbor_deg should help break ties and handle structure.
    
    scores = -(degree + mean_neighbor_deg)
    
    # Optional: If time permits, we could try a few variations and pick the best
    # by simulating the greedy decoder on a subset or the whole graph?
    # The task says "fixed greedy decoder". The evaluator runs it.
    # We can simulate it locally to check which scoring function works best?
    # "Scored by mean independent-set size on five generated Erdos-Renyi graphs"
    # We only get ONE graph in `priority`. We don't know the other 4.
    # So we can't tune to the specific 5 graphs. We must use a general heuristic.
    
    # However, we CAN simulate the greedy decoder on THIS graph for different
    # scoring strategies and pick the one that gives the largest independent set
    # for THIS graph?
    # The prompt says: "Write a Python priority(adjacency) function that gives
    # every vertex a score; a fixed greedy decoder picks vertices..."
    # "Scored by mean independent-set size on five generated... graphs"
    # This implies the evaluator calls priority 5 times (or once and uses it
    # for 5 graphs? No, "five generated... graphs" suggests 5 different inputs).
    # If I can simulate the decoder on the current input, I can optimize the
    # score for the current input!
    
    # Does the evaluator allow me to compute the actual independent set size?
    # Yes, I have the adjacency matrix. I can implement the greedy decoder
    # inside priority(), try multiple scoring strategies, and return the scores
    # that yield the best independent set for THIS specific graph.
    
    # This is a meta-heuristic: choose the best static ordering for the given graph.
    
    def greedy_decode(scores):
        # scores: array of length n. Higher score = picked first.
        # Returns size of independent set.
        selected = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        count = 0
        
        # Sort indices by score descending
        order = np.argsort(-scores)
        
        for v in order:
            if not remaining[v]:
                continue
            # If v is still available (not neighbor of selected)
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as unavailable
                # selected is bool array. neighbors are where A[v] is True.
                # We need to ensure we don't pick them later.
                # The decoder "skips neighbours of picked vertices".
                # This usually means once a vertex is picked, its neighbors
                # are removed from consideration.
                # Let's update remaining.
                # remaining[v] = False is implicit.
                # neighbors = np.where(A[v])[0]
                # remaining[neighbors] = False
                # But we also need to make sure we don't pick v again? 
                # The loop iterates through order. 
                # If a neighbor was already picked? No, independent set.
                # If a neighbor is picked later in the order? 
                # The standard greedy:
                # 1. Order vertices.
                # 2. Iterate. If vertex is not adjacent to any *already selected* vertex, select it.
                
                # Let's re-implement greedy_decode correctly.
                pass
        return count

    def greedy_decode_size(scores):
        selected = np.zeros(n, dtype=bool)
        count = 0
        order = np.argsort(-scores)
        for v in order:
            # Check if v is adjacent to any selected vertex
            # selected is a bool vector. A[v] is the row of neighbors.
            # If any neighbor is selected, skip.
            if not np.any(selected & A[v]):
                selected[v] = True
                count += 1
        return count

    # Candidate scoring strategies
    strategies = []
    
    # 1. Seed: -degree
    strategies.append(-degree)
    
    # 2. - (degree + mean_neighbor_deg)
    strategies.append(-(degree + mean_neighbor_deg))
    
    # 3. - (degree * (1 + mean_neighbor_deg))
    strategies.append(-(degree * (1 + mean_neighbor_deg)))
    
    # 4. - (degree + neighbor_deg_sum)
    strategies.append(-(degree + neighbor_deg_sum))
    
    # 5. - (degree ** 1.5)
    strategies.append(-(degree ** 1.5))
    
    # 6. - (degree + 0.5 * mean_neighbor_deg)
    strategies.append(-(degree + 0.5 * mean_neighbor_deg))
    
    # 7. - (degree / (1 + mean_neighbor_deg))
    with np.errstate(divide='ignore', invalid='ignore'):
        strat7 = -(degree / (1 + mean_neighbor_deg))
        strat7[~np.isfinite(strat7)] = 0
    strategies.append(strat7)
    
    # 8. - (degree + 0.1 * neighbor_deg_sum)
    strategies.append(-(degree + 0.1 * neighbor_deg_sum))
    
    # 9. - (degree + 2.0 * mean_neighbor_deg)
    strategies.append(-(degree + 2.0 * mean_neighbor_deg))
    
    # 10. Randomized? No, deterministic is safer for reproducibility in scoring.
    # But we can try a few more structural ones.
    
    # 11. - (degree + np.log1p(neighbor_deg_sum))
    strategies.append(-(degree + np.log1p(neighbor_deg_sum)))
    
    # 12. - (degree + 0.5 * neighbor_deg_sum)
    strategies.append(-(degree + 0.5 * neighbor_deg_sum))
    
    # Evaluate each strategy on the current graph
    best_score_val = -1
    best_scores = strategies[0]
    
    for s in strategies:
        # Ensure no NaNs
        if np.any(~np.isfinite(s)):
            s = np.where(np.isfinite(s), s, 0)
        
        size = greedy_decode_size(s)
        if size > best_score_val:
            best_score_val = size
            best_scores = s
            
        if time.time() > deadline:
            break

    return best_scores

# EVOLVE-BLOCK-END
