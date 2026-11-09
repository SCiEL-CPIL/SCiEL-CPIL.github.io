import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Return 12 distinct positive integers with all 2^12=4096 subset sums distinct,
    minimizing the largest element.
    
    Phase 1: Fast greedy minimal extension using NumPy vectorized collision checks.
    Phase 2: Local search to try reducing the largest element.
    """
    deadline = time.time() + 85  # safety margin before 90s kill

    # Phase 1: Greedy minimal extension with fast collision checking
    elements = [1]
    subset_sums = np.array([0, 1], dtype=np.int64)

    for _ in range(11):  # need 12 total, already have 1
        candidate = 2
        while True:
            # Fast collision check: does (subset_sums + candidate) intersect subset_sums?
            shifted = subset_sums + candidate
            # Check intersection using sorting
            combined = np.concatenate([subset_sums, shifted])
            combined_sorted = np.sort(combined)
            has_collision = np.any(np.diff(combined_sorted) == 0)

            if not has_collision:
                break

            candidate += 1
            if time.time() > deadline:
                return [1 << i for i in range(12)]

        elements.append(candidate)
        subset_sums = np.concatenate([subset_sums, subset_sums + candidate])

    # Phase 2: Local search to try to reduce the largest element
    # Try to replace elements (from the largest backward) with smaller values
    # that still maintain distinct subset sums
    
    def check_distinct_sums(elems):
        """Check if all 2^n subset sums are distinct. Returns True if valid."""
        sums = np.array([0], dtype=np.int64)
        for a in elems:
            sums = np.concatenate([sums, sums + a])
        if len(sums) != 4096:
            return False
        sorted_sums = np.sort(sums)
        return np.all(np.diff(sorted_sums) > 0)

    best = elements[:]
    best_max = max(best)
    
    # Try to improve: for each position from the end, try smaller values
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for pos in range(11, -1, -1):  # from largest to smallest position
            if time.time() > deadline - 5:
                break
            
            current_val = best[pos]
            # Try smaller values for this position
            # The value must be distinct from all other elements
            used = set(best)
            used.discard(current_val)
            
            # Try values from current_val - 1 down to 1
            for new_val in range(current_val - 1, 0, -1):
                if new_val in used:
                    continue
                
                trial = best[:]
                trial[pos] = new_val
                
                if check_distinct_sums(trial):
                    best = trial
                    best_max = max(best)
                    improved = True
                    break
        
        if not improved:
            break

    return best
# EVOLVE-BLOCK-END
