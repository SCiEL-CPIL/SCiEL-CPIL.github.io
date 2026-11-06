"""Rewritten program: Greedy inversion-minimization construction for 17 wires."""
import time
import numpy as np

N = 17
DEADLINE = 55.0  # Strict safety margin
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _make_all_inputs():
    """All 2^N boolean inputs as uint8 array shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)


_ALL_INPUTS = None


def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS


def count_inversions(data):
    """Count total inversions in all rows of data. 
    Inversion: pair (i, j) with i < j such that data[i] > data[j].
    For binary data, this is the number of (1, 0) pairs where 1 comes before 0.
    """
    # data shape (M, N)
    # For each row, count inversions.
    # Vectorized approach: 
    # An inversion exists at (r, i, j) if i < j and data[r, i] == 1 and data[r, j] == 0.
    # Sum over all r, i, j.
    # This is equivalent to summing for each 1 at position i, the number of 0s to its right.
    # Or simply: 
    # Let's use a simple loop over columns to accumulate counts? 
    # M = 131072, N = 17. 
    # A full O(N^2 * M) check is 17*17*131072 ~ 37M ops. Numpy can do this fast.
    
    # Create a matrix of comparisons? 
    # data[:, i] > data[:, j] for i < j.
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            total += np.sum(data[:, i] > data[:, j])
    return total


def simulate_network(comps, initial_data=None):
    """Simulate network on all boolean inputs. Returns final uint8 array (2^N, N)."""
    if initial_data is None:
        data = _get_all_inputs().copy()
    else:
        data = initial_data.copy()
    for a, b in comps:
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    return data


def is_sorted_network(comps):
    """Check if network sorts all boolean inputs (zero-one principle)."""
    final = simulate_network(comps)
    return np.all(final[:, :-1] <= final[:, 1:])


def construct():
    """Construct a sorting network using greedy inversion minimization."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with empty network
    comps = []
    
    # Get all inputs
    all_inputs = _get_all_inputs()
    current_data = all_inputs.copy()
    
    # Pre-compute all possible pairs
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Greedy Loop
    # We will keep adding comparators that reduce the total inversion count the most.
    # Stop when no comparator reduces inversions (local optimum) or time is up.
    
    # Optimization: We can stop early if we reach a known good size, but let's let it run.
    # However, greedy might get stuck. We will run until no improvement.
    
    while _time_left() > 2.0:
        best_gain = 0
        best_pair = None
        
        # Evaluate each pair
        for i, j in pairs:
            # Simulate compare-exchange on current_data
            # We don't want to copy the whole array for every pair.
            # We can just look at columns i and j.
            # gain = number of rows where data[i] > data[j]
            # Because if data[i] > data[j], swapping (min to i, max to j) fixes this inversion.
            # Does it create new inversions? 
            # For binary 0/1:
            # If data[i]=1, data[j]=0: 
            #   Before: ... 1 ... 0 ...
            #   After:  ... 0 ... 1 ...
            #   Inversions fixed: 1 (the pair i,j itself).
            #   Inversions created: 
            #     With k < i: if data[k]=0, before (0,1) no inv, after (0,0) no inv. If data[k]=1, before (1,1) no inv, after (1,0) inv created? 
            #     Wait, let's look at the structure.
            #     The comparator only changes wires i and j.
            #     Pairs involving other wires k:
            #     If k < i: 
            #       Before: (data[k], data[i]) and (data[k], data[j])
            #       After:  (data[k], min) and (data[k], max)
            #     If data[i]=1, data[j]=0: min=0, max=1.
            #       Before: (data[k], 1) and (data[k], 0)
            #       After:  (data[k], 0) and (data[k], 1)
            #       Change in inversions for k:
            #         If data[k]=0: Before (0,1)=0 inv, (0,0)=0 inv. After (0,0)=0 inv, (0,1)=0 inv. Net 0.
            #         If data[k]=1: Before (1,1)=0 inv, (1,0)=1 inv. After (1,0)=1 inv, (1,1)=0 inv. Net 0.
            #     If i < k < j:
            #       Before: (data[i], data[k]) and (data[j], data[k])
            #       After:  (min, data[k]) and (max, data[k])
            #       With min=0, max=1.
            #       Before: (1, data[k]) and (0, data[k])
            #       After:  (0, data[k]) and (1, data[k])
            #       If data[k]=0: Before (1,0)=1 inv, (0,0)=0 inv. After (0,0)=0 inv, (1,0)=1 inv. Net 0.
            #       If data[k]=1: Before (1,1)=0 inv, (0,1)=0 inv. After (0,1)=0 inv, (1,1)=0 inv. Net 0.
            #     If k > j:
            #       Before: (data[i], data[k]) and (data[j], data[k])
            #       After:  (min, data[k]) and (max, data[k])
            #       With min=0, max=1.
            #       Before: (1, data[k]) and (0, data[k])
            #       After:  (0, data[k]) and (1, data[k])
            #       If data[k]=0: Before (1,0)=1 inv, (0,0)=0 inv. After (0,0)=0 inv, (1,0)=1 inv. Net 0.
            #       If data[k]=1: Before (1,1)=0 inv, (0,1)=0 inv. After (0,1)=0 inv, (1,1)=0 inv. Net 0.
            
            # Conclusion: For binary inputs, a comparator (i,j) where data[i]=1 and data[j]=0 
            # reduces the total inversion count by exactly 1. 
            # It does NOT create any new inversions with other wires.
            # Therefore, the gain is simply the count of such pairs.
            
            gain = np.sum(current_data[:, i] > current_data[:, j])
            
            if gain > best_gain:
                best_gain = gain
                best_pair = (i, j)
        
        if best_pair is None:
            # No improvement possible
            break
        
        # Apply the best comparator
        i, j = best_pair
        lo = np.minimum(current_data[:, i], current_data[:, j])
        hi = np.maximum(current_data[:, i], current_data[:, j])
        current_data[:, i] = lo
        current_data[:, j] = hi
        comps.append(best_pair)
        
        # Check if sorted
        # If sorted, we can stop adding, but we might still have redundant comparators?
        # No, if it's sorted, any further comparator is redundant? 
        # Not necessarily. But if it's sorted, we can't reduce inversions further (already 0).
        # So the loop will break naturally on next iteration (best_gain will be 0).
        
        # Optimization: If inversion count is 0, break.
        # We can check if current_data is sorted.
        # This check is O(M*N).
        if np.all(current_data[:, :-1] <= current_data[:, 1:]):
            break

    # Now we have a valid network.
    # Try to remove redundant comparators to minimize count.
    # We will try to remove from the end.
    improved = True
    while improved and _time_left() > 1.0:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if _time_left() < 0.5:
                break
            # Try removing comps[idx]
            new_comps = comps[:idx] + comps[idx + 1:]
            if is_sorted_network(new_comps):
                comps = new_comps
                improved = True
                break

    return comps


if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")
