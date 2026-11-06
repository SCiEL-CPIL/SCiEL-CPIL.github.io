"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Rank-based scoring using the distribution of remainders.
    We want to minimize fragmentation. A good heuristic is to prefer bins
    that leave a remainder which is "common" or "fillable" by future items.
    However, a simpler and often effective approach for L1 lower bound excess
    is to keep the remainders as large as possible (to allow for large future items)
    OR to fill bins as completely as possible (Best Fit).
    
    The prompt suggests: "score each candidate bin by the negative count of other bins 
    whose remainder is smaller than the resulting gap".
    
    Let's analyze this:
    Let R be the array of current remainders.
    For a candidate bin i with remainder r_i, if we place item 'item', the new remainder is r_i' = r_i - item.
    The "resulting gap" is r_i'.
    The strategy says to score by -count(other bins with remainder < r_i').
    
    If we place the item in a bin that leaves a small remainder, that remainder is "small".
    If we place it in a bin that leaves a large remainder, that remainder is "large".
    
    Wait, the logic in the prompt is: "maximizing the likelihood that the residual space can be exactly filled by future items".
    Usually, having many small remainders is bad because they can't be filled by large items.
    Having one large remainder is good because it can accommodate a large item.
    
    If we choose a bin such that the resulting gap (r_i - item) is SMALL, we are "closing" a bin.
    If we choose a bin such that the resulting gap is LARGE, we are "keeping" space.
    
    The prompt's specific strategy: "negative count of other bins whose remainder is smaller than the resulting gap".
    Let's call the resulting gap g.
    Score = - (number of bins j != i such that r_j < g).
    
    If g is very small, few bins have remainder < g. So score is close to 0 (or small negative).
    If g is very large, many bins have remainder < g. So score is large negative.
    
    Since we want the HIGHEST score, we prefer the score closest to 0 (least negative).
    This means we prefer a SMALL resulting gap g.
    This corresponds to Best Fit Decreasing (BFD) logic: fill the bin that leaves the least space.
    This minimizes fragmentation by closing bins quickly, which is generally good for the L1 lower bound because it reduces the number of "open" bins with wasted space that don't contribute to packing efficiency.
    
    However, simple Best Fit (which is effectively this for a single item if we just look at min remaining) is the baseline.
    The rank-based approach adds a global context.
    
    Let's implement exactly what is described:
    1. Filter bins with enough capacity.
    2. Calculate resulting gaps for these bins.
    3. For each resulting gap g_i, count how many OTHER bins (in the full set? or just valid ones?) have remainder < g_i.
       The prompt says "other bins". It likely means all other bins in the current state.
    4. Score = -count.
    5. Return scores (invalid bins get -inf).
    """
    start_time = time.time()
    
    # Ensure bins is a numpy array for fast ops
    bins = np.asarray(bins, dtype=float)
    
    # Identify valid bins: those with remaining capacity >= item
    valid_mask = bins >= item
    
    # If no valid bins, return all -inf (or some invalid value)
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    # Calculate resulting gaps for valid bins
    # We only need to consider valid bins for the "resulting gap" calculation
    valid_indices = np.where(valid_mask)[0]
    valid_remainders = bins[valid_mask]
    resulting_gaps = valid_remainders - item
    
    # We need to count, for each resulting gap g, the number of OTHER bins (from the full set) 
    # that have a remainder strictly smaller than g.
    # Note: "Other bins" usually implies excluding the bin we are placing into.
    # However, since the bin we are placing into has remainder r_i, and g = r_i - item,
    # and item > 0, g < r_i.
    # So the bin itself has remainder r_i which is > g. So the bin itself is NOT counted in "remainder < g".
    # Therefore, we can just count ALL bins with remainder < g.
    
    # To make this efficient:
    # Sort the full bins array.
    sorted_bins = np.sort(bins)
    
    # For each resulting gap g, find the count of bins < g.
    # np.searchsorted(sorted_bins, g, side='left') gives the index of the first element >= g.
    # This index is exactly the count of elements < g.
    
    scores = np.full_like(bins, -np.inf)
    
    # Optimization: If the number of valid bins is small, we can just do it directly.
    # But searchsorted is O(N log N) for sort + O(M log N) for searches.
    # N is number of bins. M is number of valid bins.
    
    # Let's check time budget.
    # If N is large, sorting every call might be expensive if called 1000 times.
    # 1000 * (N log N). If N=100, 1000 * 100 * 7 = 700,000 ops. Very fast.
    
    # Calculate counts
    counts = np.searchsorted(sorted_bins, resulting_gaps, side='left')
    
    # Score is negative count
    valid_scores = -counts
    
    # Assign back to the result array
    scores[valid_mask] = valid_scores
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 10:  # Arbitrary safety margin, should be well under 120s
        pass # We are done anyway
        
    return scores
# EVOLVE-BLOCK-END
