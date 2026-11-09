"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute the heuristic matrix for ACO bin packing.
    
    The heuristic favors placing item j after item i if they fit together,
    with stronger preference for pairs that fill the bin more tightly.
    We use a best-fit style heuristic: 1/slack^alpha where slack is the
    remaining space after placing both items, amplified for tight fits.
    
    Returns a matrix where entry [i,j] is the attractiveness of placing
    item j right after item i.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pair sums: demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Mask for pairs that fit in the bin
    mask = pair <= capacity
    
    # For fitting pairs, compute slack = capacity - pair
    # We want to favor tight fits (small slack), so use 1/slack^alpha
    # But avoid division by zero; add a small epsilon
    slack = np.where(mask, capacity - pair, 1.0)
    slack = np.maximum(slack, 1e-6)  # Avoid division by zero
    
    # Use a power to amplify preference for tight fits
    alpha = 2.0
    heuristic_fitting = 1.0 / (slack ** alpha)
    
    # For non-fitting pairs, use a very small value
    heuristic = np.where(mask, heuristic_fitting, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


def run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Run the Ant Colony Optimization for bin packing.
    
    Returns the minimum number of bins used by any ant in the best iteration.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        ant_results = []
        
        for ant in range(n_ants):
            # State
            used = np.zeros(n, dtype=bool)
            current_bin_remaining = capacity
            bins_used = 1
            current_item = -1
            
            # Place a random fitting item first
            available = np.where(~used)[0]
            if len(available) == 0:
                ant_results.append(bins_used)
                continue
            
            # Find items that fit in the current bin
            fitting = available[demands[available] <= current_bin_remaining]
            if len(fitting) == 0:
                # Open a new bin with a random item
                current_bin_remaining = capacity
                bins_used = 1
                fitting = available[demands[available] <= current_bin_remaining]
                if len(fitting) == 0:
                    ant_results.append(bins_used)
                    continue
            
            # Pick a random fitting item
            idx = rng.choice(fitting)
            used[idx] = True
            current_bin_remaining -= demands[idx]
            current_item = idx
            
            # Now repeatedly pick fitting items
            while True:
                # Find available items that fit
                available = np.where(~used)[0]
                if len(available) == 0:
                    break
                
                fitting_mask = demands[available] <= current_bin_remaining
                fitting = available[fitting_mask]
                
                if len(fitting) == 0:
                    # No more items fit; open a new bin
                    current_bin_remaining = capacity
                    bins_used += 1
                    
                    # Find items that fit in the new bin
                    fitting = available[demands[available] <= current_bin_remaining]
                    if len(fitting) == 0:
                        break
                    
                    # Pick randomly from fitting items
                    idx = rng.choice(fitting)
                    used[idx] = True
                    current_bin_remaining -= demands[idx]
                    current_item = idx
                    continue
                
                # Compute probabilities: pheromone * heuristic
                if current_item == -1:
                    # First item in bin, use uniform or heuristic-based
                    probs = heuristic[0, fitting]  # Arbitrary reference
                else:
                    probs = tau[current_item, fitting] * heuristic[current_item, fitting]
                
                # Ensure positive probabilities
                probs = np.maximum(probs, 1e-9)
                probs = probs / probs.sum()
                
                # Roulette wheel selection
                r = rng.random()
                cumulative = np.cumsum(probs)
                idx_in_fitting = np.searchsorted(cumulative, r)
                if idx_in_fitting >= len(fitting):
                    idx_in_fitting = len(fitting) - 1
                
                idx = fitting[idx_in_fitting]
                used[idx] = True
                current_bin_remaining -= demands[idx]
                current_item = idx
            
            ant_results.append(bins_used)
        
        # Find the best ant in this iteration
        best_ant_bins = min(ant_results)
        best_overall = min(best_overall, best_ant_bins)
        
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / best_ant_bins
    
    return best_overall


def _solve_instance(demands, capacity, seed):
    """Solve a single instance with ACO."""
    heuristic = heuristics(demands, capacity)
    return run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=seed)


def main(demands_list, capacity):
    """
    Main entry point that processes multiple instances.
    """
    results = []
    start_time = time.time()
    deadline = 300.0  # Safety margin
    
    for i, demands in enumerate(demands_list):
        # Check time
        if time.time() - start_time > deadline - 30:
            # Not enough time for remaining instances, use a quick heuristic
            # as fallback
            for j in range(i, len(demands_list)):
                # Simple first-fit decreasing as fallback
                d = np.sort(demands_list[j])[::-1]
                bins = []
                for item in d:
                    placed = False
                    for b in bins:
                        if b + item <= capacity:
                            bins[bins.index(b)] = b + item
                            placed = True
                            break
                    if not placed:
                        bins.append(item)
                results.append(len(bins))
            break
        
        seed = 42 + i
        result = _solve_instance(demands, capacity, seed)
        results.append(result)
    
    return np.mean(results)


# EVOLVE-BLOCK-END
