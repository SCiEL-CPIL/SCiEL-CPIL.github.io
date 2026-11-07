"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Uses a 'regret' / opportunity-cost based heuristic:
    For each edge (i, j), the attractiveness is inversely proportional to
    the maximum prize of any other node k that is also reachable from i
    within the remaining budget, explicitly biasing ants away from choices
    that lead to lower opportunity costs compared to immediate alternatives.
    """
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    eps = 1e-9
    
    # Precompute return costs: cost to go from j back to 0
    return_cost = d[:, 0].copy()  # return_cost[j] = d[j, 0]
    
    # Heuristic matrix H[i, j]
    H = np.zeros((n, n), dtype=float)
    
    # For each source node i, compute the attractiveness of moving to each j
    for i in range(n):
        if i == 0:
            # From node 0, the "length so far" is 0
            # We can move to j if d[0, j] + d[j, 0] <= max_len
            reachable = (d[i, :] + return_cost) <= max_len
            reachable[i] = False  # Can't stay at i
            
            # For each reachable j, compute the regret-based attractiveness
            for j in range(n):
                if not reachable[j]:
                    H[i, j] = eps
                    continue
                
                # The "remaining budget" after moving to j is:
                # max_len - d[i, j] - d[j, 0]
                # Any other node k is "competing" if it's also reachable from i
                # i.e., d[i, k] + d[k, 0] <= max_len
                
                # Find all other reachable nodes k != j
                other_reachable = reachable.copy()
                other_reachable[j] = False
                
                if np.any(other_reachable):
                    # The maximum prize among other reachable nodes
                    max_other_prize = np.max(prizes[other_reachable])
                else:
                    max_other_prize = 0.0
                
                # Regret: how much worse is j compared to the best alternative?
                # If j has a high prize and the best alternative has a low prize,
                # the regret of NOT picking j is high, so j is more attractive.
                # Attractiveness ~ prize[j] / (1 + max_other_prize) or similar.
                # Let's use: H[i,j] = prize[j] / (1 + max_other_prize * alpha)
                # where alpha controls how strongly we penalize having good alternatives.
                
                # Actually, the directive says: inversely proportional to the
                # maximum prize value of any other unvisited node k.
                # So: H[i,j] = prize[j] / (1 + max_other_prize)
                
                H[i, j] = prizes[j] / (1.0 + max_other_prize)
        
        else:
            # From node i (not 0), the "length so far" depends on the path taken.
            # Since the heuristic is static (computed once), we need a proxy.
            # A reasonable proxy: assume the ant has used some fraction of the budget.
            # The worst case is that the ant has already traveled a significant distance.
            # Let's use a conservative estimate: assume the remaining budget is
            # max_len - d[0, i] (the cost to reach i from 0).
            
            # Actually, for a static heuristic, we can consider the "effective"
            # remaining budget as max_len - d[0, i] - d[i, 0] is not quite right.
            # Let's think: the ant is at i. It has spent some length L_so_far.
            # The constraint is: L_so_far + d[i, j] + d[j, 0] <= max_len.
            # Since we don't know L_so_far, let's use the minimum possible L_so_far,
            # which is d[0, i] (the direct distance from 0 to i).
            # So the effective remaining budget is: max_len - d[0, i]
            
            effective_budget = max_len - d[0, i]
            
            # j is reachable if d[i, j] + d[j, 0] <= effective_budget
            reachable = (d[i, :] + return_cost) <= effective_budget
            reachable[i] = False
            
            for j in range(n):
                if not reachable[j]:
                    H[i, j] = eps
                    continue
                
                other_reachable = reachable.copy()
                other_reachable[j] = False
                
                if np.any(other_reachable):
                    max_other_prize = np.max(prizes[other_reachable])
                else:
                    max_other_prize = 0.0
                
                H[i, j] = prizes[j] / (1.0 + max_other_prize)
    
    # Ensure all non-positive values become 1e-9
    H[H <= 0] = eps
    H[~np.isfinite(H)] = eps
    
    return H
# EVOLVE-BLOCK-END
