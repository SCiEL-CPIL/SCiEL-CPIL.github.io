import time
import random
import itertools

def _check_distinct_subset_sums(arr):
    """Check if all subset sums are distinct."""
    n = len(arr)
    num_sums = 1 << n
    sums = [0] * num_sums
    for i in range(n):
        for mask in range(1 << i):
            sums[(1 << i) | mask] = sums[mask] + arr[i]
    return len(set(sums)) == num_sums

def construct():
    """
    Returns a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    Uses randomized local search starting from powers of 2, with greedy reduction
    passes to find smaller valid configurations.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    # Start with powers of 2 (known valid solution, largest = 2048)
    best_arr = [1 << i for i in range(n)]
    best_max = best_arr[-1]
    
    # First, do a greedy reduction pass to get a better starting point
    def greedy_reduce(arr):
        arr = list(arr)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    break
                old_val = arr[i]
                found = False
                # Try to reduce arr[i] to a smaller value
                for new_val in range(1, old_val):
                    if time.time() > deadline:
                        break
                    # Check distinctness
                    if new_val in arr[:i] or new_val in arr[i+1:]:
                        continue
                    arr[i] = new_val
                    if _check_distinct_subset_sums(arr):
                        found = True
                        improved = True
                        break
                if not found:
                    arr[i] = old_val
        return arr
    
    best_arr = greedy_reduce(best_arr)
    best_max = max(best_arr)
    
    # Now do randomized local search
    random.seed(42)
    
    def mutate(arr):
        """Create a mutated version of arr by adjusting one or two elements."""
        arr = list(arr)
        num_mutations = random.choice([1, 1, 2])
        indices = random.sample(range(n), num_mutations)
        for i in indices:
            old_val = arr[i]
            # Try to change to a nearby value
            delta = random.randint(-max(1, old_val // 4), max(1, old_val // 4))
            new_val = old_val + delta
            if new_val >= 1:
                arr[i] = new_val
        # Ensure distinctness of elements
        seen = set()
        for i in range(n):
            while arr[i] in seen:
                arr[i] += 1
            seen.add(arr[i])
        return arr
    
    def local_search(arr):
        """Perform local search starting from arr."""
        arr = list(arr)
        current_max = max(arr)
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Try to reduce the maximum element
            max_idx = arr.index(max(arr))
            old_val = arr[max_idx]
            
            # Try reducing max_idx
            found = False
            for new_val in range(1, old_val):
                if time.time() > deadline:
                    break
                if new_val in arr[:max_idx] or new_val in arr[max_idx+1:]:
                    continue
                arr[max_idx] = new_val
                if _check_distinct_subset_sums(arr):
                    found = True
                    break
            if not found:
                arr[max_idx] = old_val
                # Try other positions
                for i in range(n - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    old_val = arr[i]
                    found = False
                    for new_val in range(1, old_val):
                        if time.time() > deadline:
                            break
                        if new_val in arr[:i] or new_val in arr[i+1:]:
                            continue
                        arr[i] = new_val
                        if _check_distinct_subset_sums(arr):
                            found = True
                            break
                    if not found:
                        arr[i] = old_val
                    else:
                        break
        
        return arr
    
    # Run multiple local search iterations
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # Start from a mutated version of the best solution
        candidate = mutate(best_arr)
        
        # Ensure candidate is valid
        if not _check_distinct_subset_sums(candidate):
            # Try to repair
            candidate = greedy_reduce(candidate)
            if not _check_distinct_subset_sums(candidate):
                continue
        
        # Local search
        candidate = local_search(candidate)
        
        # Check if improved
        candidate_max = max(candidate)
        if candidate_max < best_max:
            best_arr = candidate
            best_max = candidate_max
            # Do another greedy reduction pass
            best_arr = greedy_reduce(best_arr)
            best_max = max(best_arr)
    
    # Final verification
    if _check_distinct_subset_sums(best_arr):
        return best_arr
    
    # Fallback to powers of 2
    return [1 << i for i in range(n)]
