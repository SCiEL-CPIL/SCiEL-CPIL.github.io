import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin
    
    n = 12
    # Start with powers of 2
    arr = [1 << i for i in range(n)]
    
    def get_subset_sums(arr):
        sums = {0}
        for x in arr:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums |= new_sums
        return sums

    def try_reduce(arr, idx):
        """Try to reduce arr[idx] to the smallest possible value."""
        current = arr[idx]
        others = arr[:idx] + arr[idx+1:]
        other_sums = get_subset_sums(others)
        used = set(others)
        
        # We want the smallest v < current
        # Check v from 1 to current - 1
        # Optimization: if current is small, range is small.
        # If current is large, we might find a small v quickly.
        
        for v in range(1, current):
            if v in used:
                continue
            
            # Check collision: other_sums & (other_sums + v)
            # Equivalent to: does v exist in difference set of other_sums?
            # Fast check using set intersection
            shifted = {s + v for s in other_sums}
            if not (other_sums & shifted):
                return v
        return None

    # Local search loop
    # Randomize index order to avoid bias
    indices = list(range(n))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            new_val = try_reduce(arr, idx)
            if new_val is not None and new_val < arr[idx]:
                arr[idx] = new_val
                improved = True
    
    return arr
