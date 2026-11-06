import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard
    # The evaluator kills after 150s. We should be well under that.
    # But we add a check just in case N is huge.
    start_time = time.time()
    deadline = start_time + 1.0  # 1 second limit per call is generous
    
    n = len(bins)
    if n == 0:
        return np.array([])
        
    scores = np.full(n, -np.inf)
    
    # Precompute sorted bins? 
    # For each j, we need count of bins < gap_j.
    # gap_j = bins[j] - item
    
    # Optimization:
    # If we sort bins, we can use binary search.
    # But we need to exclude j?
    # We established that bins[j] is never < gap_j.
    # So count = number of elements in bins strictly less than gap_j.
    
    # We can sort the bins once.
    # sorted_bins = np.sort(bins)
    # For each j:
    #    gap = bins[j] - item
    #    count = np.searchsorted(sorted_bins, gap, side='left')
    #    scores[j] = -count
    
    # This is O(N log N) for sort + O(N log N) for searches.
    # Much faster than O(N^2) for large N.
    
    if n > 0:
        sorted_bins = np.sort(bins)
        
        for j in range(n):
            if bins[j] >= item:
                gap = bins[j] - item
                # Number of bins with remainder < gap
                # searchsorted returns index of first element >= gap
                # So number of elements < gap is exactly the index returned by side='left'
                count = np.searchsorted(sorted_bins, gap, side='left')
                scores[j] = -count
            else:
                scores[j] = -np.inf
                
            if time.time() > deadline:
                break
                
    return scores
