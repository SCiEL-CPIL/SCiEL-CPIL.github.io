# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_2opt(dist, tour):
    """
    Fast 2-opt implementation.
    Returns the improved tour.
    """
    n = len(tour)
    if n <= 2:
        return tour
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            # If we are at the end, the next edge wraps around
            if i == n - 2:
                c = tour[0]
                d = tour[n - 1]
            else:
                c = tour[i + 2]
                d = tour[i + 1] # Wait, standard 2-opt logic:
                # Edge 1: (tour[i], tour[i+1])
                # Edge 2: (tour[i+1], tour[i+2]) -- No, 2-opt removes (i, i+1) and (j, j+1)
                # Let's stick to the standard O(n^2) or O(n^3) logic carefully.
                # Standard 2-opt: reverse segment [i+1, j]
                # Gain = dist[tour[i], tour[i+1]] + dist[tour[j], tour[j+1]] - dist[tour[i], tour[j]] - dist[tour[i+1], tour[j+1]]
                # Here, we fix i and vary j > i+1.
                pass
        
        # Let's use a simpler, robust O(n^2) per pass implementation for clarity and safety
        # Actually, let's just do a standard pass.
        # For speed, we can vectorize the gain calculation for a fixed i.
        
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            
            # j ranges from i+2 to n-1 (if j==n-1, the second edge is (n-1, 0))
            # If i == n-2, there is no j > i+1 in the linear sense, but we consider the wrap.
            # Actually, if i=n-2, b=tour[n-1]. The next node is tour[0].
            # So j can be 0? No, j must be > i in the reversed segment logic.
            # Let's just iterate j from i+2 to n-1.
            # And handle the wrap case for j=n-1 (edge to 0) and i=0?
            
            # Gain for reversing segment (i+1 ... j)
            # Old edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1 % n])
            # New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1 % n])
            
            # We can vectorize the gain for all j > i+1
            # j_indices = np.arange(i+2, n)
            # This gets complicated with the wrap-around for j=n-1.
            # Let's just loop j. n=100, so 100*100 = 10k ops per pass. 20 passes = 200k ops. Very fast.
            
            for j in range(i + 2, n):
                c = tour[j]
                # d is the node after c in the current tour
                if j == n - 1:
                    d = tour[0]
                else:
                    d = tour[j + 1]
                
                # Current edges: (a, b) and (c, d)
                # New edges: (a, c) and (b, d)
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse tour[i+1 : j+1]
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    improved = True
                    # Break inner loop? Or continue? 
                    # Usually, after a move, we restart the i loop or continue carefully.
                    # Restarting i loop is safer and standard for simple implementations.
                    # But for speed, we can break and let the while loop handle it.
                    # However, breaking immediately might miss other improvements in the same pass.
                    # Let's just set improved=True and break out of j loop, then i loop continues?
                    # No, the tour changed, so indices are invalid.
                    # Best practice: break out of both loops and restart the pass.
                    break
            if improved:
                # We made a change. Restart the pass? Or just continue?
                # If we continue, the tour has changed. The previous i is now different.
                # It is safer to restart the entire pass (break outer loop too) to ensure consistency.
                # Given n=100, restarting is cheap.
                break
                
    return tour

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search with Internal 2-Opt.
    
    Strategy:
    1. Compute additive penalties for all edges based on edge_n_used.
    2. Create a perturbed distance matrix.
    3. Run 2-opt on the perturbed matrix to find the best tour in the perturbed space.
    4. Return the perturbed matrix (or a matrix that guides the evaluator's 2-opt to the same place).
    
    Wait, the interface requires returning a matrix. The evaluator then runs 2-opt on THIS matrix.
    If I return the perturbed matrix, the evaluator's 2-opt will refine it.
    However, if I can find a *better* tour under the perturbed metric myself, I can't return the tour.
    I must return a matrix.
    
    Alternative Strategy:
    The evaluator runs 2-opt on my returned matrix.
    If I return a matrix where the optimal tour under that matrix is a specific good tour, 
    the evaluator's 2-opt should find it (or close to it).
    
    Let's try this:
    1. Apply standard GLS penalties: P(e) = alpha * count(e).
    2. Perturbed Dist D' = D + P.
    3. Run my own 2-opt on D' to get a candidate tour T'.
    4. Now, I want the evaluator's 2-opt (on D') to converge to T'.
    5. If I just return D', the evaluator's 2-opt starts from `local_opt_tour` (the input tour).
       It might get stuck in a local optimum of D' that is not T'.
       
    To force the evaluator's 2-opt to find T', I can "guide" the matrix further.
    But I can only return a matrix.
    
    Actually, the prompt says: "run 2-opt ... on the matrix you return".
    It starts from `local_opt_tour`? 
    "It then repeats 20 times: ... call your function with copies ... run 2-opt ... on the matrix you return".
    Usually, 2-opt is run on the *current* tour. The "current tour" is the one kept from the previous iteration (best so far).
    So the evaluator starts 2-opt from the current best tour, using my perturbed matrix.
    
    If I use a standard GLS penalty, the landscape is changed. The current best tour is a local opt in the TRUE metric, but likely not in the PERTURBED metric (because its edges are penalized). So 2-opt on the perturbed metric will likely move away from the current tour.
    
    The key is: How strong should the penalty be?
    Standard GLS uses a small penalty.
    If the penalty is too strong, it distorts the metric too much and the resulting tour might be bad in true distance.
    If the penalty is too weak, 2-opt won't escape.
    
    Let's try an adaptive penalty based on the "stagnation" or just a fixed reasonable alpha.
    Also, I will run my own 2-opt on the perturbed matrix to see if it finds a better tour.
    If it does, I can potentially "encode" that tour into the matrix? No, that's hard.
    
    Let's stick to the standard GLS formulation but tuned.
    Penalty = alpha * count.
    D' = D + Penalty.
    
    What alpha?
    Typical values for GLS are small, e.g., 0.01 to 0.1 relative to mean distance.
    Let's calculate the mean distance.
    
    Additionally, I will run 2-opt on D' internally to check the quality.
    If the internal 2-opt finds a tour with true length better than the input `local_opt_tour`'s true length, 
    it suggests the perturbation is working.
    
    Actually, I can do better.
    I can run 2-opt on D' to get T_perturbed.
    Then I can compute the true length of T_perturbed.
    If T_perturbed is better than local_opt_tour, great.
    
    But I just return D'. The evaluator runs 2-opt on D' starting from local_opt_tour.
    Will it reach T_perturbed?
    If T_perturbed is a local optimum of D', and local_opt_tour is far from it, 2-opt might get stuck.
    
    To help the evaluator's 2-opt, I can add a "pull" towards T_perturbed?
    No, I can only modify distances.
    
    Let's try a different angle:
    The current program (score 0.99) multiplies current edges by 5-15. This is a "kick".
    My new strategy: Additive penalty on ALL edges based on usage.
    This is the textbook GLS.
    
    Let's implement textbook GLS.
    alpha = 0.05 * mean_dist.
    D' = D + alpha * edge_n_used.
    
    To ensure the evaluator's 2-opt escapes, the penalty on the current tour's edges must be significant.
    Since edge_n_used accumulates, after 20 rounds, used edges have high counts.
    If we use alpha=0.05, and count=20, penalty = 1.0. Mean dist ~ 0.5?
    Then penalty is 2x the distance. That's a strong kick.
    
    Let's refine alpha.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()
    
    # Calculate mean distance to scale penalty
    # Use a sample or full matrix mean? Full is O(n^2) = 10k ops. Fine.
    mean_dist = np.mean(edge_distance)
    
    # Adaptive alpha
    # If usage is low, use smaller penalty? Or constant?
    # Let's try alpha = 0.05 * mean_dist
    # But we need to ensure we escape.
    # Let's look at the max usage in the current tour.
    tour_edges = [(local_opt_tour[i], local_opt_tour[(i+1)%n]) for i in range(n)]
    current_max_usage = max(edge_n_used[u, v] for u, v in tour_edges)
    
    # If current tour edges are heavily used, they are "traps".
    # We want the penalty to be strong enough to break them.
    # Let's set the penalty such that the most used edge in the current tour
    # becomes significantly more expensive than its alternatives.
    
    # Strategy: Set alpha such that alpha * current_max_usage is a significant fraction of the edge cost.
    # e.g., alpha * current_max_usage = 0.5 * mean_dist
    # alpha = 0.5 * mean_dist / max(1, current_max_usage)
    
    if current_max_usage > 0:
        alpha = 0.5 * mean_dist / current_max_usage
    else:
        alpha = 0.0 # No usage yet, no penalty? Or small random kick?
        # If no usage, we are in round 1. The tour is NN+2opt.
        # We should still perturb to find a better basin.
        # Let's use a small default alpha.
        alpha = 0.05 * mean_dist

    # Create perturbed matrix
    perturbed_dist = edge_distance + alpha * edge_n_used
    
    # Ensure symmetry and finite values
    # edge_n_used is symmetric, edge_distance is symmetric.
    
    # Optional: Run internal 2-opt to verify?
    # If we have time, we can run 2-opt on perturbed_dist to see if it finds a good tour.
    # But we can't return the tour.
    # However, we can use the result to adjust alpha? No, too complex for one call.
    
    # Let's just return the perturbed matrix.
    # The evaluator's 2-opt will do the rest.
    
    # One optimization: If the current tour is very good, maybe we don't want to perturb too much?
    # But we don't know if it's "very good" globally, only locally.
    
    return perturbed_dist
# EVOLVE-BLOCK-END
