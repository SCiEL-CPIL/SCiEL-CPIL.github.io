import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Valid bins have non-negative remaining capacity
    valid_mask = rem >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_rem = rem[valid_mask]
    
    # Strategy: Score by negative count of other bins whose remainder is smaller 
    # than the resulting gap. This favors creating gaps that are likely to be filled 
    # by future items (matching the distribution of current gaps).
    
    # For each valid bin i, resulting gap = valid_rem[i]
    # We want to maximize the count of other bins j (j != i) such that bins[j] <= valid_rem[i]
    # Note: bins[j] is the current remainder of bin j.
    # If we put item in bin i, the new remainder of bin i becomes valid_rem[i].
    # We want this new remainder to be "common" or "fillable".
    # The directive says: "negative count of other bins whose remainder is smaller than the resulting gap"
    # Wait, "maximizing the likelihood that the residual space can be exactly filled"
    # If we pick a bin where the resulting gap is small, it's hard to fill.
    # If we pick a bin where the resulting gap matches many other bins' current gaps, it might be good?
    # Let's re-read carefully: 
    # "score each candidate bin by the negative count of other bins whose remainder is smaller than the resulting gap"
    # Score = - (count of j != i such that bins[j] < valid_rem[i])
    # Since we want the highest score (most positive or least negative), we want to MINIMIZE the count of bins with smaller remainder.
    # This means we prefer bins where the resulting gap is SMALLER than most other bins' current gaps.
    # This is essentially Best-Fit (smallest gap), but using a rank-based score instead of the raw gap size.
    
    # Let's compute the ranks of valid_rem among all bins (including invalid ones? No, "other bins" implies all bins in the state)
    # The bins array contains the current remainders of ALL bins.
    # We need to count for each valid bin i, how many bins j (j != i) have bins[j] < valid_rem[i].
    
    # Efficient way:
    # 1. Sort all bin remainders.
    # 2. For a given gap g = valid_rem[i], the count of bins with remainder < g is the rank of g in the sorted list (minus 1 if unique, or careful with ties).
    # Actually, np.searchsorted on sorted bins gives the index where g would be inserted to maintain order.
    # The number of elements strictly less than g is the insertion index.
    
    sorted_bins = np.sort(bins)
    
    # For each valid_rem value, find the number of bins with remainder < valid_rem
    # np.searchsorted(sorted_bins, valid_rem, side='left') gives the count of elements < valid_rem
    counts_less = np.searchsorted(sorted_bins, valid_rem, side='left')
    
    # We need to exclude the bin itself if its current remainder is less than the new gap?
    # The bin i currently has remainder bins[i].
    # After placing item, its remainder is valid_rem[i] = bins[i] - item.
    # Since item > 0, valid_rem[i] < bins[i].
    # So bins[i] is NOT less than valid_rem[i].
    # Therefore, the bin i is not counted in "bins with remainder < valid_rem[i]".
    # So counts_less is exactly the count of OTHER bins with remainder < valid_rem[i].
    
    # Score = -counts_less
    valid_scores = -counts_less.astype(float)
    
    scores[valid_indices] = valid_scores
    return scores
