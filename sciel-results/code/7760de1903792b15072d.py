import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N
    K = 2  # Max representations allowed

    # State variables
    # chosen_arr: NumPy array of elements in the set
    # counts: NumPy array of size MAX_SUM+1, counts[s] = number of pairs {a,b} in set with a+b=s
    # avail_set: Python set of elements not in the set, for fast membership and sampling
    
    best_size = 0
    best_set_list = []
    
    def _can_add(x, chosen_arr, counts):
        """Check if x can be added without violating the B2[2] constraint."""
        if chosen_arr.size == 0:
            return counts[2 * x] < K
        # Check if adding x would cause any sum to exceed K
        # Sums formed by x and existing elements
        sums = chosen_arr + x
        if np.any(counts[sums] >= K):
            return False
        # Check 2x
        if counts[2 * x] >= K:
            return False
        return True

    def _add(x, chosen_arr, counts):
        """Add x to the set and update counts."""
        if chosen_arr.size > 0:
            sums = chosen_arr + x
            counts[sums] += 1
        counts[2 * x] += 1
        return np.append(chosen_arr, x)

    def _remove(x, chosen_arr, counts):
        """Remove x from the set and update counts."""
        # Create mask for elements not equal to x
        mask = chosen_arr != x
        remaining = chosen_arr[mask]
        if remaining.size > 0:
            sums = remaining + x
            counts[sums] -= 1
        counts[2 * x] -= 1
        return remaining

    def _greedy_init(order):
        """Run greedy selection in the given order."""
        chosen = np.empty(0, dtype=np.int64)
        counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
        for x in order:
            if _can_add(int(x), chosen, counts):
                chosen = _add(int(x), chosen, counts)
        return chosen, counts

    def _local_search(initial_chosen, initial_counts, rng, deadline):
        """
        Perform local search starting from an initial set.
        Returns the best set found.
        """
        chosen = initial_chosen.copy()
        counts = initial_counts.copy()
        
        chosen_set = set(int(v) for v in chosen)
        avail_set = set(range(1, N + 1)) - chosen_set
        
        best_size = len(chosen)
        best_chosen = chosen.copy()
        best_counts = counts.copy()
        
        # Pre-allocate a list for random sampling from avail_set if needed, 
        # but since avail_set changes, we use list(avail_set) occasionally.
        # For efficiency, we can maintain a list of available elements.
        avail_list = list(avail_set)
        rng.shuffle(avail_list)
        avail_idx = 0
        
        while time.time() < deadline:
            # Try to insert
            inserted = False
            # Try a few random candidates from the available list
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts and avail_idx < len(avail_list):
                x = int(avail_list[avail_idx])
                avail_idx += 1
                attempts += 1
                
                if x in chosen_set:
                    continue
                
                if _can_add(x, chosen, counts):
                    chosen = _add(x, chosen, counts)
                    chosen_set.add(x)
                    avail_set.remove(x)
                    # Remove from avail_list efficiently? 
                    # For simplicity, we just mark it and skip if encountered, 
                    # or rebuild periodically. 
                    # Given N=1000, rebuilding is cheap if done rarely.
                    # But we can just let avail_list contain stale entries and check membership.
                    # Actually, since we increment avail_idx, we might run out.
                    # If avail_idx reaches end, shuffle and reset.
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_chosen = chosen.copy()
                        best_counts = counts.copy()
                    inserted = True
                    break
            
            if not inserted:
                # If we ran out of candidates in the current shuffled list, reshuffle
                if avail_idx >= len(avail_list):
                    if avail_set:
                        avail_list = list(avail_set)
                        rng.shuffle(avail_list)
                        avail_idx = 0
                    else:
                        # No more elements to add, must swap
                        pass
                
                # Try a swap: remove a random element, then try to add a different one
                if not chosen_set:
                    break
                
                # Remove a random element
                # Pick a random index from chosen
                rem_idx = rng.randrange(len(chosen))
                u = int(chosen[rem_idx])
                
                # Remove u
                chosen = _remove(u, chosen, counts)
                chosen_set.remove(u)
                avail_set.add(u)
                # Note: u is now in avail_set. We don't explicitly add to avail_list here, 
                # but if we reshuffle, it will be included.
                
                # Try to add a different element
                swapped = False
                # Try to find a valid element to add
                # We can try the removed element again? No, that's a no-op.
                # Try random elements from avail_set.
                # To avoid bias, we can try a few random ones.
                if avail_set:
                    # Sample a few candidates
                    # Using list(avail_set) is O(N), but N=1000 is small.
                    # However, we have avail_list which might be stale.
                    # Let's just pick a random element from avail_set directly for the swap attempt.
                    # This is O(1) average if we use a list, but avail_set is a set.
                    # Converting to list for sampling is O(N).
                    # Given the tight loop, maybe we should maintain a list of avail.
                    # Let's stick to the simple approach: pick random from avail_set.
                    # Python's random.choice on a set requires conversion.
                    # Let's use a small sample.
                    sample_size = min(10, len(avail_set))
                    candidates = random.sample(avail_set, sample_size)
                    for v in candidates:
                        if _can_add(int(v), chosen, counts):
                            chosen = _add(int(v), chosen, counts)
                            chosen_set.add(int(v))
                            avail_set.remove(int(v))
                            swapped = True
                            # Size is same as before swap (removed 1, added 1)
                            # So best_size doesn't change, but we might escape a local optimum
                            break
                    
                    if not swapped:
                        # Put u back if we couldn't find a replacement
                        chosen = _add(u, chosen, counts)
                        chosen_set.add(u)
                        avail_set.remove(u)

        return best_chosen, best_counts, best_size

    # 1. Deterministic Ascending Greedy Baseline
    asc_order = np.arange(1, N + 1, dtype=np.int64)
    chosen, counts = _greedy_init(asc_order)
    best_size = len(chosen)
    best_set_list = chosen.tolist()
    
    # 2. Local Search on Baseline
    rng = np.random.default_rng(42)
    if time.time() < deadline - 2.0:
        improved_chosen, improved_counts, improved_size = _local_search(chosen, counts, rng, deadline)
        if improved_size > best_size:
            best_size = improved_size
            best_set_list = improved_chosen.tolist()
    
    # 3. Random Restarts with Local Search
    # Use a deadline margin to allow for the final return
    search_deadline = deadline - 1.0
    
    while time.time() < search_deadline:
        # Random order
        order = np.arange(1, N + 1, dtype=np.int64)
        rng.shuffle(order)
        
        chosen, counts = _greedy_init(order)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set_list = chosen.tolist()
            
        # Local search
        if time.time() < search_deadline - 0.5:
            improved_chosen, improved_counts, improved_size = _local_search(chosen, counts, rng, search_deadline)
            if improved_size > best_size:
                best_size = improved_size
                best_set_list = improved_chosen.tolist()

    return best_set_list
