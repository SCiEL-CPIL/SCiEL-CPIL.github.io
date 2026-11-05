import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    A = adjacency.astype(np.int8)
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Precompute neighbor lists as sets for fast membership checks
    neighbors = [set(np.where(A[i])[0]) for i in range(n)]
    
    # --- Phase 1: Stochastic Game Value Estimation ---
    # For each vertex v, estimate its expected contribution to the independent set
    # by sampling random priority orderings and checking if v would be selected.
    # A vertex's "game value" is the probability it gets picked under a random
    # greedy decoder, weighted by the resulting set size.
    
    n_samples = 500  # number of random orderings to sample
    pick_counts = np.zeros(n)
    total_set_sizes = np.zeros(n)  # sum of set sizes in samples where v is picked
    
    for s in range(n_samples):
        if time.time() > deadline:
            break
        
        # Random priority ordering
        order = np.random.permutation(n)
        
        # Greedy decode: pick vertex if no neighbor already picked
        picked = np.zeros(n, dtype=bool)
        for v in order:
            if not picked[v]:
                # Check if any neighbor is already picked
                nb = neighbors[v]
                if not any(picked[u] for u in nb):
                    picked[v] = True
        
        # Record contributions
        pick_counts += picked
        set_size = picked.sum()
        # Weight: vertices picked in larger sets are more valuable
        # Use set_size as weight for each picked vertex
        for v in range(n):
            if picked[v]:
                total_set_sizes[v] += set_size
    
    # Game value: expected contribution = (sum of set sizes where picked) / (number of samples * expected set size)
    # Simpler: use weighted pick frequency
    # value[v] = total_set_sizes[v] / (n_samples * avg_set_size)
    avg_set_size = pick_counts.sum() / n_samples if n_samples > 0 else 1
    game_values = total_set_sizes / (n_samples * max(avg_set_size, 1e-10))
    
    # --- Phase 2: Local Improvement via Swap-based Search ---
    # Start with the ordering implied by game_values, then try swaps to improve
    # the greedy decoder output.
    
    # Initial ordering: sort by game value descending
    order = np.argsort(-game_values)
    
    def evaluate_order(ord):
        """Run greedy decoder on given order, return set size."""
        picked = np.zeros(n, dtype=bool)
        for v in ord:
            if not picked[v]:
                nb = neighbors[v]
                if not any(picked[u] for u in nb):
                    picked[v] = True
        return int(picked.sum())
    
    current_size = evaluate_order(order)
    best_order = order.copy()
    best_size = current_size
    
    # Local search: try pairwise swaps
    improved = True
    max_swap_rounds = 50
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Try swapping positions i and j
                new_order = best_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_size = evaluate_order(new_order)
                if new_size > best_size:
                    best_size = new_size
                    best_order = new_order
                    improved = True
                    break  # restart inner loop after a swap
            if improved:
                break
    
    # --- Phase 3: Multi-restart with perturbation ---
    # Try a few perturbed starts to escape local optima
    if time.time() < deadline:
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Perturb the best order
            perturbed = best_order.copy()
            # Randomly swap a few pairs
            n_swaps = max(1, n // 20)
            for _ in range(n_swaps):
                a = np.random.randint(0, n)
                b = np.random.randint(0, n)
                perturbed[a], perturbed[b] = perturbed[b], perturbed[a]
            
            # Local search from perturbed start
            cur_size = evaluate_order(perturbed)
            cur_order = perturbed
            imp = True
            while imp and time.time() < deadline:
                imp = False
                for i in range(n - 1):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n):
                        new_order = cur_order.copy()
                        new_order[i], new_order[j] = new_order[j], new_order[i]
                        new_size = evaluate_order(new_order)
                        if new_size > cur_size:
                            cur_size = new_size
                            cur_order = new_order
                            imp = True
                            break
                    if imp:
                        break
                # Limit iterations per restart
                if not imp:
                    break
            
            if cur_size > best_size:
                best_size = cur_size
                best_order = cur_order
    
    # --- Final Score Assignment ---
    # Assign scores such that the greedy decoder picks vertices in best_order
    # Score = position-based (earlier = higher score)
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
    
    return scores
