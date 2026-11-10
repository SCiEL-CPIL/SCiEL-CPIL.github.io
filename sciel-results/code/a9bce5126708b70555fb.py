"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _subset_sums_unique(arr):
    """Check if all subset sums are distinct."""
    n = len(arr)
    sums = set()
    sums.add(0)
    for x in arr:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        if len(new_sums) != len(sums):
            return False
        sums = new_sums
    return len(sums) == (1 << n)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Known optimal or near-optimal solutions for distinct subset sums
    # For n=12, the minimal largest element is known to be 2047 (powers of 2)
    # But let's try to find better solutions using greedy and local search
    
    # Start with powers of 2
    best = [1 << i for i in range(12)]
    best_max = max(best)
    
    # Try greedy construction
    def greedy_construct(n):
        """Greedy: pick smallest possible value at each step."""
        arr = []
        for i in range(n):
            found = False
            for candidate in range(1, 1 << 20):
                test_arr = arr + [candidate]
                if _subset_sums_unique(test_arr):
                    arr.append(candidate)
                    found = True
                    break
            if not found:
                return None
        return arr
    
    # Try greedy first
    greedy_result = greedy_construct(12)
    if greedy_result:
        greedy_max = max(greedy_result)
        if greedy_max < best_max:
            best = greedy_result
            best_max = greedy_max
    
    # Local search: try to reduce elements
    def local_search(arr, deadline):
        """Try to reduce the largest element via local search."""
        arr = list(arr)
        n = len(arr)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to reduce each element
            for i in range(n - 1, -1, -1):
                if arr[i] <= 1:
                    continue
                for candidate in range(arr[i] - 1, 0, -1):
                    if candidate in arr[:i] or candidate in arr[i+1:]:
                        continue
                    test_arr = arr[:i] + [candidate] + arr[i+1:]
                    if _subset_sums_unique(test_arr):
                        arr[i] = candidate
                        improved = True
                        break
        return arr
    
    # Apply local search
    if time.time() < deadline:
        improved_arr = local_search(best, deadline)
        if max(improved_arr) < best_max:
            best = improved_arr
            best_max = max(improved_arr)
    
    return best
# EVOLVE-BLOCK-END
