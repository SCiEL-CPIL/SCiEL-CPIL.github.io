"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    n = len(demands)
    rng = np.random.default_rng(seed)
    # Precompute which items fit after which item (i.e., demands[i]+demands[j] <= capacity)
    fits = (demands[:, None] + demands[None, :] <= capacity)
    # Also precompute which items can be placed in an empty bin (always true since demands <= 99 < 150)
    
    best_bins = float('inf')
    pheromone = np.ones((n, n))
    
    for it in range(n_iters):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current = -1  # -1 means new bin
            
            while placed.sum() < n:
                if current == -1:
                    # Open a new bin, place a random fitting item
                    candidates = np.where(~placed)[0]
                    if len(candidates) == 0:
                        break
                    current = candidates[rng.integers(len(candidates))]
                    placed[current] = True
                    bins_used += 1
                else:
                    # Try to place an item after current
                    candidates = np.where((~placed) & fits[current, :])[0]
                    if len(candidates) == 0:
                        # Open a new bin
                        current = -1
                        continue
                    
                    # Compute probabilities proportional to pheromone[current, j] * heuristic[current, j]
                    probs = pheromone[current, candidates] * heuristic[current, candidates]
                    # Ensure positive
                    probs = np.where(probs > 0, probs, 1e-9)
                    total = probs.sum()
                    if total <= 0 or not np.isfinite(total):
                        # Fallback: random
                        current = candidates[rng.integers(len(candidates))]
                        placed[current] = True
                    else:
                        probs /= total
                        current = candidates[rng.choice(len(candidates), p=probs)]
                        placed[current] = True
            
        # Track min bins for this iteration
        min_bins_this_iter = min(min_bins_this_iter, bins_used)
        
        # Update pheromone
        pheromone *= 0.9
        if min_bins_this_iter > 0 and np.isfinite(min_bins_this_iter):
            pheromone += 1.0 / min_bins_this_iter
        
        best_bins = min(best_bins, min_bins_this_iter)
    
    return best_bins


def heuristics(demands, capacity):
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Start with a base heuristic: prefer pairs that leave good remaining space
    # pair[i,j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Base: smaller pair sum is better (leaves more room)
    # Use inverse: 1/(pair_sum + epsilon) for pairs that fit, 1e-9 otherwise
    eps = 1e-6
    base = np.where(pair <= capacity, 1.0 / (pair + eps), 1e-9)
    
    # Add a component that rewards pairing large items together (to consolidate large items)
    # This is like a "big items together" heuristic
    max_demand = demands.max()
    if max_demand > 0:
        large_score = (demands[:, None] + demands[None, :]) / (2 * max_demand)
    else:
        large_score = np.zeros((n, n))
    
    # Combine: we want small pair sums AND we want to group large items
    # Use a weighted combination
    # Actually, for bin packing, it's often better to pack large items together
    # to avoid wasting space with small items mixed in.
    # Let's use: heuristic = base * (1 + alpha * large_score)
    # where alpha controls the strength of the "group large" effect
    
    alpha = 1.0
    heuristic = base * (1.0 + alpha * large_score)
    
    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
