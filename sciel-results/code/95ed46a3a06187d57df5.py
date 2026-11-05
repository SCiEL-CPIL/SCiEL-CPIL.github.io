import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Instead of optimizing a static ordering via local search, we simulate the greedy
    decoding process to determine a dynamic selection order. At each step of the
    simulation, we select the vertex from the remaining available set that has the
    minimum degree within that available set (Minimum Available Degree).
    
    This "dynamic LDF" approach adapts to the changing state of the graph as vertices
    are selected and blocked, often outperforming static degree-based orderings because
    it accounts for the specific interactions that occur during the actual greedy run.
    
    The resulting selection order is then converted into priority scores (earlier
    selection = higher score) for the fixed greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # We will simulate the greedy process to find a good order.
    # available: boolean mask of vertices not yet picked or blocked.
    # selected_order: list of vertices in the order they were picked in simulation.
    
    available = np.ones(n, dtype=bool)
    selected_order = []
    
    # Optimization: Pre-compute adjacency lists or use matrix ops.
    # For n=200, O(n^2) per step is fine (200*200*200 = 8M ops total), which is fast in Python/NumPy.
    
    # To make it faster and potentially better, we can look ahead slightly or use a more complex metric,
    # but simple dynamic LDF is a strong baseline for this structural change.
    # Let's add a small tie-breaker: among min-degree vertices, pick the one with lower global degree
    # or just random/stable to ensure determinism.
    
    global_deg = adj.sum(axis=1)
    
    for _ in range(n):
        if not np.any(available):
            break
            
        # Calculate degree of each available vertex within the available subgraph
        # deg_avail[v] = sum(adj[v, available])
        # This is a matrix-vector multiplication or sum over axis.
        deg_avail = adj[:, available].sum(axis=1)
        
        # We only care about vertices that are currently available.
        # Set degrees of non-available vertices to infinity so they are never picked.
        deg_avail[~available] = np.inf
        
        # Find the vertex with the minimum degree in the available subgraph.
        # If there are ties, we want a stable, deterministic choice.
        # np.argmin returns the first occurrence.
        min_idx = np.argmin(deg_avail)
        
        # It's possible all remaining are blocked? No, 'available' tracks that.
        # If min_idx is not available (shouldn't happen if available has True), skip.
        if not available[min_idx]:
            continue

        # Pick this vertex
        selected_order.append(min_idx)
        available[min_idx] = False
        
        # Block its neighbors
        neighbors = adj[min_idx]
        available[neighbors] = False
        
        # Time check (unlikely to be needed for n=200, but good practice)
        if time.time() > deadline:
            break

    # If we didn't select all vertices due to time or logic, fill the rest with arbitrary order
    # to ensure we return a valid permutation/score set.
    if len(selected_order) < n:
        remaining = np.where(available)[0]
        # Sort remaining by global degree as a fallback
        remaining_sorted = remaining[np.argsort(global_deg[remaining], kind='stable')]
        selected_order.extend(remaining_sorted.tolist())
        
    # Ensure we have exactly n elements (in case of early break or other issues)
    # If selected_order has duplicates or missing, this could fail. 
    # Let's construct a robust score array.
    scores = np.zeros(n)
    
    # Assign scores based on the order they were selected.
    # Earlier selection -> Higher Score.
    # Score = n - rank
    for rank, v in enumerate(selected_order):
        if 0 <= v < n:
            scores[v] = n - rank
            
    # If any vertex was missed (should not happen with the fill logic above), give it 0
    # But the greedy decoder needs a total order. If scores are 0, it might pick them arbitrarily.
    # To be safe, let's ensure all vertices have a unique score if possible, or at least valid.
    # The fill logic above should handle it.
    
    return scores

