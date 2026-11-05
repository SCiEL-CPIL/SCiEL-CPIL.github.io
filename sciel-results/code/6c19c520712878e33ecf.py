import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct Permutation-based Simulated Annealing
    1. Start with a heuristic ordering (e.g., degree-based).
    2. Use Simulated Annealing to optimize the vertex permutation directly.
    3. The objective is the size of the independent set produced by the greedy decoder.
    4. Use incremental updates to efficiently evaluate the impact of swaps.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    # 1. Initial Heuristic: Degree-based ordering
    deg = adj.sum(axis=1).astype(float)
    # Low degree vertices are generally better to include
    initial_order = np.argsort(deg, kind='stable')
    
    def simulate(order):
        """Greedy simulation on a given order."""
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    for nb in nbrs:
                        if active[nb]:
                            active[nb] = False
        return count
    
    # 2. Baseline
    best_order = initial_order.copy()
    best_val = simulate(best_order)
    current_order = best_order.copy()
    current_val = best_val
    
    # 3. Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995  # Cooling rate
    
    # We will perform a fixed number of steps or until time runs out
    # Let's estimate how many steps we can do. 
    # A swap evaluation is O(1) if we just swap in the array and re-simulate? 
    # No, re-simulating is O(n * avg_deg). 
    # We need a faster evaluation or a smarter move.
    
    # Alternative: Use a "greedy repair" or "local search" on the permutation.
    # Since SA on permutations with full re-simulation is slow, let's use a hybrid:
    # 1. Start with heuristic order.
    # 2. Perform local search: try swapping each vertex with its neighbors in the order.
    # 3. If stuck, do a random restart or perturbation.
    
    # Let's try a more efficient local search: 
    # "2-opt" style moves: swap two vertices in the ordering.
    # To make it fast, we can precompute the "contribution" of each vertex? 
    # That's hard because the greedy process is sequential.
    
    # Let's stick to SA but with a faster inner loop if possible, or just accept slower steps but fewer of them.
    # Actually, the prompt mentions "incremental delta-evaluation". 
    # If we swap two vertices i and j in the ordering, how much does the MIS size change?
    # It's complex. 
    
    # Let's try a different structural approach: **Iterated Local Search (ILS)**
    # 1. Start with a good heuristic order.
    # 2. Do a local search (swap adjacent or nearby vertices) until no improvement.
    # 3. Perturb the order (random swaps).
    # 4. Repeat.
    
    def local_search(order):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n - 1):
                # Try swapping i and i+1
                order[i], order[i+1] = order[i+1], order[i]
                new_val = simulate(order)
                if new_val > current_val:
                    current_val = new_val
                    improved = True
                else:
                    order[i], order[i+1] = order[i+1], order[i]
            # Also try swapping i and i+2, etc? Maybe just adjacent is enough for now.
            # Let's add a second pass with larger jumps if time permits.
        return current_val
    
    # Let's refine the local search to be more effective.
    # Instead of just adjacent, try swapping any two vertices? That's O(n^2) per pass.
    # For n=200, n^2 = 40000. Simulate is ~200*10 = 2000 ops. Total 80M ops per pass. 
    # That's too slow for many passes.
    
    # Let's use a smarter heuristic for the initial order and a targeted local search.
    # Heuristic: Order by (degree, -common_neighbors) or similar.
    
    # Let's try: 
    # 1. Compute a "greedy score" for each vertex based on how many neighbors it blocks.
    # 2. Sort by this score.
    # 3. Local search: Try to move a vertex to a better position.
    
    # Actually, the inspiration mentioned "Simulated Annealing directly on the vertex permutation with an incremental delta-evaluation".
    # How to do incremental delta?
    # If we swap two vertices u and v in the order:
    # - The vertices before u are unchanged.
    # - The vertices between u and v are unchanged in their relative order, but their "active" status might change? No, the greedy process is deterministic.
    # - The change only affects the segment from min(pos_u, pos_v) to max(pos_u, pos_v).
    # This is still complex to compute incrementally without re-simulating that segment.
    
    # Given the time constraint, let's implement a robust ILS with adjacent swaps and occasional larger jumps, relying on the fact that the landscape is relatively smooth.
    
    # Revised Strategy:
    # 1. Initial order: Sort by degree (low to high).
    # 2. Local Search: 
    #    - Pass 1: Swap adjacent vertices if it improves.
    #    - Pass 2: Swap vertices at distance 2 if it improves.
    #    - Repeat until no improvement.
    # 3. Perturbation: Randomly swap 5-10 pairs of vertices.
    # 4. Repeat ILS.
    
    def simulate_and_track(order):
        """Simulate and return count."""
        return simulate(order)

    # Initial Order
    current_order = initial_order.copy()
    current_val = simulate(current_order)
    best_order = current_order.copy()
    best_val = current_val
    
    # ILS Loop
    while time.time() < deadline - 10:
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Adjacent swaps
            for i in range(n - 1):
                if time.time() > deadline - 5:
                    break
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                new_val = simulate(current_order)
                if new_val > current_val:
                    current_val = new_val
                    improved = True
                else:
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            
            if not improved:
                # Try distance 2 swaps if time permits
                for i in range(n - 2):
                    if time.time() > deadline - 5:
                        break
                    current_order[i], current_order[i+2] = current_order[i+2], current_order[i]
                    new_val = simulate(current_order)
                    if new_val > current_val:
                        current_val = new_val
                        improved = True
                    else:
                        current_order[i], current_order[i+2] = current_order[i], current_order[i+2]
        
        # Update Best
        if current_val > best_val:
            best_val = current_val
            best_order = current_order.copy()
        
        # Perturbation
        if time.time() < deadline - 10:
            # Randomly swap 10 pairs
            for _ in range(10):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
            current_val = simulate(current_order)
            
    # 4. Convert best_order back to scores
    # The score is the inverse of the rank in the order.
    # Higher score = selected earlier.
    # If best_order[0] is the first selected, it should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank  # So best_order[0] gets n, best_order[-1] gets 1
    
    return scores

