# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute the item-pair attractiveness matrix for ACO bin packing.
    
    The heuristic rewards placing item j after item i if they fit together,
    with higher reward for tighter fits (less wasted capacity).
    We use (1 - slack)^alpha where slack = 1 - (d_i + d_j)/capacity.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    # For fitting pairs, compute how tight the fit is
    # ratio = (d_i + d_j) / capacity, closer to 1 is better
    ratio = pair / capacity
    # Use a power to amplify preference for tight fits
    # alpha=3 gives strong preference for near-full bins
    heuristic = np.where(mask, ratio ** 3, 1e-9)
    # Ensure no negative or zero values cause issues
    heuristic = np.maximum(heuristic, 1e-9)
    return heuristic


def run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Run the ACO simulation and return the minimum bins used by any ant.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Pheromone matrix starts at 1
    tau = np.ones((n, n), dtype=np.float64)
    
    # Pre-compute which items can follow which (fitting constraint)
    # This is static - doesn't change during the run
    # But we need to track used items per ant
    
    rng = np.random.default_rng(seed)
    
    best_bins_overall = np.inf
    
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        best_bins_this_iter = np.inf
        best_ant_path = None  # Store the sequence of (i, j) pairs for the best ant
        
        for ant in range(n_ants):
            if time.time() > deadline:
                break
            
            used = np.zeros(n, dtype=bool)
            bins = 1
            current = None
            current_bin_items = []
            ant_path = []
            
            # Place first item randomly
            available = np.where(~used)[0]
            first_idx = rng.choice(available)
            used[first_idx] = True
            current = first_idx
            current_bin_items = [first_idx]
            
            # Repeatedly pick next fitting item
            while True:
                # Find all items that fit after current
                # An item j fits if d[j] <= capacity - sum(current_bin_items)
                # But the heuristic already encodes pairwise fit.
                # Actually, the task says "picking a fitting item" - 
                # an item fits if it can be added to the current bin.
                # The current bin's remaining capacity is capacity - sum(d[current_bin_items])
                remaining = capacity - np.sum(d[current_bin_items])
                
                # Items that fit: d[j] <= remaining and not used
                fit_mask = (~used) & (d <= remaining)
                
                # Also need the heuristic to be positive (non-fitting pairs are 1e-9)
                # But the task says "probability proportional to pheromone[current, j] * heuristic[current, j]"
                # and "when none fits, it opens a new bin"
                # So we only consider items where the pair (current, j) fits
                # The heuristic already has 1e-9 for non-fitting pairs, but we should
                # also respect the bin capacity constraint.
                
                # The pairwise constraint: d[current] + d[j] <= capacity
                # But the bin constraint is: sum of all items in bin + d[j] <= capacity
                # These are different! The bin constraint is tighter.
                
                # Let's use the bin constraint: d[j] <= remaining
                # And also the pairwise constraint from the heuristic
                # Actually, if d[current] + d[j] <= capacity (pairwise), and all items in the bin
                # are non-negative, then the bin constraint is: sum(bin) + d[j] <= capacity
                # which is stricter than just d[current] + d[j] <= capacity.
                
                # So the fitting condition is: d[j] <= remaining AND not used
                # The heuristic value for (current, j) should also be positive.
                
                candidates = np.where(fit_mask)[0]
                
                if len(candidates) == 0:
                    # No more items fit in this bin, open a new bin
                    bins += 1
                    if bins > best_bins_overall:
                        # Early termination for this ant
                        break
                    # Pick a random item for the new bin
                    available = np.where(~used)[0]
                    if len(available) == 0:
                        break
                    first_idx = rng.choice(available)
                    used[first_idx] = True
                    current = first_idx
                    current_bin_items = [first_idx]
                    ant_path.append((current, first_idx))  # Mark transition
                    continue
                
                # Compute probabilities: tau[current, j] * heuristic[current, j]
                probs = tau[current, candidates] * heuristic[current, candidates]
                
                # Ensure all are positive
                probs = np.maximum(probs, 1e-9)
                total = np.sum(probs)
                if total <= 0:
                    break
                probs = probs / total
                
                # Pick next item
                next_idx = rng.choice(candidates, p=probs)
                used[next_idx] = True
                current_bin_items.append(next_idx)
                ant_path.append((current, next_idx))
                current = next_idx
            
            if bins < best_bins_this_iter:
                best_bins_this_iter = bins
                best_ant_path = ant_path
        
        if best_bins_this_iter < best_bins_overall:
            best_bins_overall = best_bins_this_iter
        
        # Update pheromone
        # Multiply by 0.9
        tau *= 0.9
        # Every entry gains 1/(fewest bins of that iteration)
        if best_ant_path is not None and best_bins_this_iter > 0:
            reward = 1.0 / best_bins_this_iter
            for (i, j) in best_ant_path:
                tau[i, j] += reward
    
    return best_bins_overall


def solve_instance(demands, capacity, seed):
    """Solve a single instance and return the minimum bins used."""
    heuristic = heuristics(demands, capacity)
    result = run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=seed)
    return result


# The main interface
def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    ratio = pair / capacity
    heuristic = np.where(mask, ratio ** 3, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
