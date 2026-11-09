"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])
    
    # Precompute neighbor masks for fast vectorized blocking
    neighbor_masks = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if len(neighbors[i]) > 0:
            neighbor_masks[i, neighbors[i]] = True
    
    def greedy_mis_size(order):
        """
        order: list of vertices in the order they should be considered (first is highest priority)
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if len(neighbors[v]) > 0:
                    selected |= neighbor_masks[v]
        return count
    
    # --- STRATEGY: Dynamic Weighting via Greedy Removal Layers ---
    # Model the problem as Maximum Weight Independent Set where weights are
    # dynamically assigned based on a greedy removal process.
    # We simulate a greedy MIS construction to identify "layers" of removal.
    # Vertices removed in earlier layers (when the graph is dense) have high "potential".
    # Vertices removed in later layers (sparse graph) have lower potential but are "safer".
    
    # Step 1: Generate a reference order using Min-Degree heuristic
    # This serves as the baseline for layer assignment.
    ref_order = list(np.argsort(degrees))
    
    # Step 2: Simulate greedy removal to assign layers
    # active[v] = True if v is still in the graph
    active = np.ones(n, dtype=bool)
    layers = np.zeros(n, dtype=int)
    current_layer = 0
    
    # We process in batches to be efficient, but for n=200, individual steps are fine.
    # However, we want to assign a score based on the "value" of picking v.
    # Let's refine the score using a combination of:
    # 1. Degree in the original graph (lower is better for inclusion)
    # 2. Degree in the "remaining" graph during a simulated greedy pass
    
    # Let's do a more sophisticated scoring:
    # Score = - (degree in original graph) * w1 + (layer index) * w2 + (local density) * w3
    
    # Actually, the prompt suggests: "iteratively select the vertex with the maximum 'independent potential' ... remove it and its neighbors, and assign scores proportional to the layer in which they were removed"
    # Wait, usually for MIS, you want to pick vertices that don't block many others.
    # If we remove v, we lose v and its neighbors.
    # "Independent potential" could be defined as: 1 / (1 + degree_remaining)
    
    # Let's generate a few different "reference" orders to create a robust score.
    # Order 1: Min Degree
    # Order 2: Max Degree (for comparison, usually bad for MIS but good for Vertex Cover)
    # Order 3: Random
    
    # We will use a "Voting" or "Aggregation" approach.
    # For each vertex v, calculate its rank in Min-Degree order.
    # Also calculate its rank in a "Coreness" order (k-core decomposition).
    
    # --- Structural Change: Coreness-based Scoring ---
    # The k-core of a graph is the maximal subgraph where every vertex has degree at least k.
    # Vertices in higher k-cores are more "central" and thus "more blocking".
    # Therefore, vertices with LOWER coreness should have HIGHER priority (score).
    
    # Compute Coreness (k-shell)
    coreness = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg_rem = degrees.copy()
    
    k = 0
    while remaining.any():
        # Find min degree in remaining graph
        min_deg = np.inf
        for v in range(n):
            if remaining[v] and deg_rem[v] < min_deg:
                min_deg = deg_rem[v]
        
        if min_deg == 0:
            # Remove all degree 0 vertices
            for v in range(n):
                if remaining[v] and deg_rem[v] == 0:
                    coreness[v] = k
                    remaining[v] = False
            k += 1
            # Update degrees
            for u in range(n):
                if remaining[u]:
                    # Count how many removed neighbors u had
                    removed_nei = [v for v in range(n) if not remaining[v] and adj[u, v]]
                    if removed_nei:
                        deg_rem[u] -= len(removed_nei)
        else:
            # Find all vertices with degree == min_deg
            candidates = [v for v in range(n) if remaining[v] and deg_rem[v] == min_deg]
            for v in candidates:
                coreness[v] = k
                remaining[v] = False
            
            k += 1
            # Update degrees
            for u in range(n):
                if remaining[u]:
                    removed_nei = [v for v in candidates if adj[u, v]]
                    if removed_nei:
                        deg_rem[u] -= len(removed_nei)
    
    # Normalize coreness to [0, 1]
    max_core = coreness.max() if coreness.max() > 0 else 1
    core_score = 1.0 - (coreness / max_core)
    
    # Degree score: lower degree is better
    max_deg = degrees.max() if degrees.max() > 0 else 1
    deg_score = 1.0 - (degrees / max_deg)
    
    # Combine scores
    # Weights: Coreness is often a better predictor of "centrality" than degree.
    # Let's try a weighted sum.
    w_core = 0.6
    w_deg = 0.4
    
    scores = w_core * core_score + w_deg * deg_score
    
    # --- Refinement Machinery: Local Search on the Score Vector ---
    # Instead of optimizing a permutation directly (which is expensive),
    # we optimize the score vector by perturbing the weights or individual scores.
    # However, the decoder sorts by score. Small changes in score can change the order.
    
    # Let's use the simulated annealing approach from inspiration 1, but on the permutation
    # derived from the initial scores.
    
    # Initial order from combined scores
    current_order = list(np.argsort(-scores))
    current_size = greedy_mis_size(current_order)
    
    best_order = current_order[:]
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    num_restarts = 3
    steps_per_restart = 15000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            order = current_order[:]
        else:
            order = best_order[:]
            for _ in range(5):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
        
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        temp = 2.0
        cooling = 0.9998
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = current_size - new_size
            
            if delta < 0:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            elif delta == 0:
                current_size = new_size
            else:
                if rng.random() < np.exp(-delta / temp):
                    current_size = new_size
                else:
                    order[i], order[j] = order[j], order[i]
                    
            temp *= cooling
            if temp < 0.01:
                break
    
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    final_scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        final_scores[v] = n - rank
        
    return final_scores.astype(float)

# EVOLVE-BLOCK-END
