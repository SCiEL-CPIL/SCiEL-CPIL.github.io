import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    N = 1000
    MAX_SUM = 2 * N
    K = 2

    all_nums = np.arange(1, N + 1, dtype=np.int64)

    def _can_add_vec(x, chosen_arr, counts):
        """Check if x can be added without violating the B2[2] constraint."""
        if chosen_arr.size == 0:
            return counts[2 * x] < K
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
        if counts[2 * x] >= K:
            return False
        return True

    def _add_vec(x, chosen_arr, counts):
        """Add x to the set and update counts. Returns new array and counts."""
        if chosen_arr.size > 0:
            s = chosen_arr + x
            counts[s] += 1
        counts[2 * x] += 1
        return np.concatenate([chosen_arr, np.array([x], dtype=np.int64)]), counts

    def _remove_vec(x, chosen_arr, counts):
        """Remove x from the set and update counts. Returns new array and counts."""
        mask = chosen_arr != x
        remaining = chosen_arr[mask]
        if remaining.size > 0:
            s = remaining + x
            counts[s] -= 1
        counts[2 * x] -= 1
        return remaining, counts

    def _saturating_greedy(order, counts):
        """Greedy that adds any valid element in the given order."""
        chosen = np.empty(0, dtype=np.int64)
        for x in order:
            if _can_add_vec(int(x), chosen, counts):
                chosen, counts = _add_vec(int(x), chosen, counts)
        return chosen, counts

    def _local_search(chosen, counts, rng, deadline, max_iter=500000):
        """
        Hybrid local search:
        1. Attempt to INSERT a valid element (grow size).
        2. If stuck, attempt a SWAP (remove one, add another) to escape local optima.
        3. If a swap succeeds, continue trying to insert.
        """
        n = len(chosen)
        chosen_set = set(int(v) for v in chosen)
        best_size = n
        best_chosen = chosen.copy()
        best_counts = counts.copy()

        avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
        rng.shuffle(avail)

        start = time.time()
        # Track consecutive failures to decide when to try swaps
        fail_count = 0

        for it in range(max_iter):
            if (it & 255) == 0:
                if time.time() > deadline:
                    break
            if avail.size == 0:
                break

            # 1. Try to insert
            # Pick a random candidate from avail
            idx = rng.randrange(avail.size)
            x_add = int(avail[idx])
            
            if x_add in chosen_set:
                # Should not happen, but safety check
                continue

            if _can_add_vec(x_add, chosen, counts):
                chosen, counts = _add_vec(x_add, chosen, counts)
                chosen_set.add(x_add)
                avail = np.delete(avail, idx)
                fail_count = 0
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = chosen.copy()
                    best_counts = counts.copy()
                continue

            # 2. Cannot insert. If we have failed many times, try a swap.
            # Strategy: Try a few random swaps. If a swap allows an immediate insert, great.
            # Otherwise, just do the swap if it doesn't make things "worse" (it doesn't, size is same).
            # We'll try a few random swaps to see if they unlock an insert.
            
            if fail_count > 10 or (it % 100 == 0 and fail_count > 0):
                # Try up to 5 random swaps
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    if chosen.size == 0:
                        break
                    
                    # Pick a random element to remove
                    rm_idx = rng.randrange(chosen.size)
                    x_rm = int(chosen[rm_idx])
                    
                    # Tentatively remove
                    temp_chosen, temp_counts = _remove_vec(x_rm, chosen, counts)
                    
                    # Try to add a random available element
                    # Pick a random available element
                    if avail.size == 0:
                        # Restore and break
                        chosen, counts = _add_vec(x_rm, temp_chosen, temp_counts)
                        break
                        
                    add_idx = rng.randrange(avail.size)
                    x_try = int(avail[add_idx])
                    
                    if _can_add_vec(x_try, temp_chosen, temp_counts):
                        # Swap successful! Apply it.
                        # x_rm is removed, x_try is added.
                        # current state is temp_chosen, temp_counts
                        # We need to add x_try to temp_chosen
                        final_chosen, final_counts = _add_vec(x_try, temp_chosen, temp_counts)
                        
                        # Update state
                        chosen = final_chosen
                        counts = final_counts
                        chosen_set.discard(x_rm)
                        chosen_set.add(x_try)
                        
                        # Update avail
                        # x_try was at add_idx in avail
                        avail = np.delete(avail, add_idx)
                        # x_rm is now available
                        avail = np.concatenate([avail, np.array([x_rm], dtype=np.int64)])
                        
                        # Reset fail count since we made a move
                        fail_count = 0
                        break
                    else:
                        # Swap failed, restore state
                        chosen, counts = _add_vec(x_rm, temp_chosen, temp_counts)
                
                fail_count += 1
            else:
                fail_count += 1

        return best_chosen, best_counts

    # --- Phase 1: Multiple Greedy Seeds ---
    best_chosen = np.empty(0, dtype=np.int64)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    best_size = 0

    # Deterministic orders
    orders = [
        all_nums,
        all_nums[::-1],
    ]
    
    # Randomized orders
    for seed in (123, 7, 999, 2024, 555, 111):
        o = all_nums.copy()
        r = random.Random(seed)
        r.shuffle(o)
        orders.append(o)

    for o in orders:
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct

    # --- Phase 2: Local Search from Best Seed ---
    rng = random.Random(42)
    ch, ct = _local_search(best_chosen, best_counts, rng, deadline, max_iter=1000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        best_counts = ct

    # --- Phase 3: Try other good seeds with Local Search if time remains ---
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Generate a new seed
        seed = 10000 + trial * 101
        o = all_nums.copy()
        rt = random.Random(seed)
        rt.shuffle(o)
        
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        
        # Only proceed if this seed is competitive
        if len(ch2) < best_size - 2:
            continue
            
        # Run local search with a tighter time budget for this trial
        # Reserve 10s for the final return
        current_deadline = min(deadline, time.time() + 10)
        if current_deadline <= time.time():
            break
            
        ch2, ct2 = _local_search(ch2, ct2, rng, current_deadline, max_iter=200000)
        
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = ch2
            best_counts = ct2

    # Final check: ensure we return a valid list of ints
    return [int(v) for v in best_chosen]

# EVOLVE-BLOCK-END
