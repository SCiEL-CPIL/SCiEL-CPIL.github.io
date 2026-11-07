"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _fast_aco_eval(d, capacity, heuristic, n_ants=5, n_iters=5, seed=123):
    """
    Fast ACO simulation to evaluate a heuristic matrix.
    Returns the fewest bins found.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    # Ensure heuristic is valid for ACO (positive where it fits, tiny otherwise)
    # We assume the heuristic passed in is already masked for fits, 
    # but we re-mask to be safe against optimizer noise
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    
    # Clip to avoid log issues or zero probs if optimizer produces weird values
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    h = np.where(fits, np.maximum(heuristic, 1e-9), 1e-9)
    
    best_bins = float('inf')
    
    # Pheromone initialization
    pheromone = np.ones((n, n))
    
    for _ in range(n_iters):
        iter_best = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # Pick random start
            idx = rng.randint(n)
            placed[idx] = True
            current_bin_sum = d[idx]
            last_item = idx
            bins_used = 1
            
            while not np.all(placed):
                remaining_space = capacity - current_bin_sum
                fits_mask = (~placed) & (d <= remaining_space)
                
                if not np.any(fits_mask):
                    bins_used += 1
                    available = np.where(~placed)[0]
                    idx = available[rng.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                # Probability proportional to pheromone * heuristic
                row_ph = pheromone[last_item, :]
                row_he = h[last_item, :]
                
                scores = row_ph * row_he
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = np.sum(scores)
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[rng.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    idx = rng.choice(n, p=probs)
                    
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
                
            if bins_used < iter_best:
                iter_best = bins_used
            if bins_used < best_bins:
                best_bins = bins_used
        
        # Pheromone update
        pheromone *= 0.9
        pheromone += 1.0 / iter_best
        
    return best_bins

def _cma_es_optimize(d, capacity, init_h, max_time=280.0, pop_size=10, sigma=1.0):
    """
    Simple CMA-ES implementation to optimize the heuristic matrix.
    Objective: Minimize bins from _fast_aco_eval.
    """
    n = len(d)
    # We only optimize the upper triangle (or full matrix, but symmetry helps)
    # Actually, H[i,j] and H[j,i] can be different in ACO if directed, but here
    # "placing item j right after item i" implies direction. 
    # However, usually bin packing heuristics are symmetric. 
    # The task says "item j right after item i". 
    # Let's optimize the full matrix to be safe, or just the unique pairs.
    # Full matrix is 120*120 = 14400 params. 
    # CMA-ES with 14400 dims is heavy. 
    # Let's try to optimize a lower-dimensional representation or use a very small pop.
    # Given time constraints, maybe we optimize a "temperature" or "scaling" of a good base?
    # No, the directive says "different design family". 
    # Let's try optimizing the raw matrix but with a very aggressive time budget and small pop.
    # Or, better: optimize a "rank-based" heuristic where we only adjust the top K partners?
    
    # Alternative: Use a "Learning" approach where we simulate, identify "good" pairs, and reinforce them.
    # This is essentially what ACO does, but we can do it *before* the real ACO run to seed the heuristic.
    # Wait, the heuristic is computed ONCE. 
    # So we can use a "Pre-ACO" phase to learn the heuristic.
    
    # Let's stick to a simpler, more robust "Learning" strategy:
    # 1. Run a greedy FFD to get a good solution.
    # 2. Extract the adjacency matrix of items that ended up in the same bin in that solution.
    # 3. Use this structure to shape the heuristic.
    # This is a "Greedy-Induced Heuristic".
    
    # Actually, let's try a different structural change:
    # **Co-occurrence Heuristic**: 
    # Run multiple random greedy packings. Count how often pairs (i,j) appear in the same bin.
    # H[i,j] = 1 + count(i,j) * weight.
    # This learns the "affinity" directly from the problem structure without parametric assumptions.
    
    pass # Placeholder, we will implement the Greedy-Induced approach below

def _greedy_induced_heuristic(d, capacity, n_runs=50, seed=42):
    """
    Generates a heuristic based on co-occurrence in random greedy packings.
    Items that frequently fit together in good greedy solutions get higher heuristic values.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    # Count co-occurrences
    co_occurrence = np.zeros((n, n))
    
    # Also track "quality" of the bin they were in (e.g., fill ratio) to weight the counts?
    # For simplicity, just count. Or weight by (1 / bins_used) to favor better packings.
    
    for _ in range(n_runs):
        # Randomized FFD
        # Sort indices randomly, then place each in the first bin it fits?
        # Or: Pick random item, place in bin with least remaining space that fits? (Best Fit)
        # Let's use Randomized Best Fit Decreasing (R-BFD)
        
        # Sort by size descending, but add noise
        sizes = d + rng.normal(0, 1, n)
        order = np.argsort(-sizes)
        
        bins = []
        bin_sums = []
        
        for idx in order:
            item_size = d[idx]
            best_bin_idx = -1
            best_remaining = float('inf')
            
            for b_idx in range(len(bins)):
                remaining = capacity - bin_sums[b_idx]
                if item_size <= remaining:
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_bin_idx = b_idx
            
            if best_bin_idx == -1:
                bins.append([idx])
                bin_sums.append(item_size)
            else:
                bins[best_bin_idx].append(idx)
                bin_sums[best_bin_idx] += item_size
        
        # Update co-occurrence
        # Weight by inverse number of bins (better packing -> higher weight)
        weight = 1.0 / len(bins)
        
        for b in bins:
            for i in b:
                for j in b:
                    if i != j:
                        co_occurrence[i, j] += weight

    # Normalize
    # Max possible co-occurrence is roughly n_runs
    max_co = np.max(co_occurrence)
    if max_co > 0:
        co_occurrence = co_occurrence / max_co
    
    # Construct heuristic
    # Base 1.0, plus co-occurrence signal
    h = 1.0 + 10.0 * co_occurrence
    
    # Mask non-fitting pairs
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness.
    Strategy: Greedy-Induced Heuristic via Co-occurrence Learning.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Time budget check
    if _time_budget_remaining() < 5.0:
        # Fallback: simple complementarity
        pair_sums = d[:, None] + d[None, :]
        fits = pair_sums <= capacity
        gaps = np.abs(pair_sums - capacity)
        h = 1.0 + 5.0 * np.exp(-gaps / 10.0)
        h = np.where(fits, h, 1e-9)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return h

    # 1. Generate candidate heuristics with different numbers of greedy runs
    # and different random seeds to capture different structures.
    
    candidates = []
    
    # We have ~280s budget.
    # Each greedy run is O(n^2) roughly. 120^2 = 14400 ops. Very fast.
    # ACO eval is O(ants * iters * n^2). 
    # 5 ants * 5 iters * 14400 = 360k ops. Also fast.
    # So we can afford many candidates.
    
    run_configs = [
        (20, 1),
        (50, 2),
        (100, 3),
        (200, 4),
        (50, 10), # Different seed
        (100, 20),
    ]
    
    for n_runs, seed in run_configs:
        if _time_budget_remaining() < 10.0:
            break
        
        h = _greedy_induced_heuristic(d, capacity, n_runs=n_runs, seed=seed)
        
        # Evaluate with fast ACO
        score = _fast_aco_eval(d, capacity, h, n_ants=5, n_iters=5, seed=99)
        candidates.append((score, h))
        
        # Also try a "mixed" heuristic: 50% greedy-induced, 50% complementarity
        pair_sums = d[:, None] + d[None, :]
        fits = pair_sums <= capacity
        gaps = np.abs(pair_sums - capacity)
        comp_h = 1.0 + 5.0 * np.exp(-gaps / 10.0)
        comp_h = np.where(fits, comp_h, 1e-9)
        
        mixed_h = 0.5 * h + 0.5 * comp_h
        mixed_h = np.where(fits, mixed_h, 1e-9)
        mixed_h = np.where(np.isfinite(mixed_h) & (mixed_h > 0), mixed_h, 1e-9)
        
        score_mixed = _fast_aco_eval(d, capacity, mixed_h, n_ants=5, n_iters=5, seed=99)
        candidates.append((score_mixed, mixed_h))

    if not candidates:
        # Fallback
        pair_sums = d[:, None] + d[None, :]
        fits = pair_sums <= capacity
        gaps = np.abs(pair_sums - capacity)
        h = 1.0 + 5.0 * np.exp(-gaps / 10.0)
        h = np.where(fits, h, 1e-9)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return h

    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # Pick the best
    best_score, best_h = candidates[0]
    
    # If we have time, refine the best one with a slightly larger ACO eval?
    # Or just return it. The "learning" phase is the key structural change.
    
    return best_h

# EVOLVE-BLOCK-END
