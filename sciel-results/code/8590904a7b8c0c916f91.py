import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def find_sum_free(elements):
        """Find a large sum-free subset using bitmask optimization and local search."""
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}

        # Precompute conflict structure
        # conflicts_as_sum[k] = list of (i, j) where elements[i] + elements[j] == elements[k]
        # We store these as bitmasks of pairs? No, list of pairs is fine for iteration, 
        # but for "is pair active", we check bits.
        
        # To speed up "count_conflicts", we can precompute for each k, the list of pairs.
        # But checking "is i and j active" is O(1) with bitmasks.
        
        conflicts_as_sum = defaultdict(list)
        conflicts_as_left = defaultdict(list)
        conflicts_as_right = defaultdict(list)

        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    conflicts_as_sum[k].append((i, j))
                    conflicts_as_left[i].append((j, k))
                    conflicts_as_right[j].append((i, k))

        # Helper to create bitmask from set of indices
        def set_to_mask(idx_set):
            mask = 0
            for i in idx_set:
                mask |= (1 << i)
            return mask

        def mask_to_set(mask):
            # Not strictly needed if we keep the set, but useful for debugging
            s = set()
            for i in range(n):
                if mask & (1 << i):
                    s.add(i)
            return s

        def count_conflicts(mask):
            """Count number of violated sum-free constraints using bitmask."""
            count = 0
            # Iterate over all k that are active
            # Optimization: Only iterate k where bit is set
            # We can just iterate all k in range(n) and check if bit is set
            for k in range(n):
                if mask & (1 << k):
                    for (i, j) in conflicts_as_sum[k]:
                        if (mask & (1 << i)) and (mask & (1 << j)):
                            count += 1
            return count

        def can_add(idx, mask):
            """Check if adding idx to active set (mask) creates any new conflict."""
            # 1. idx as sum: i + j = idx
            for (i, j) in conflicts_as_sum[idx]:
                if (mask & (1 << i)) and (mask & (1 << j)):
                    return False
            # 2. idx as left: idx + j = k
            for (j, k) in conflicts_as_left[idx]:
                if (mask & (1 << j)) and (mask & (1 << k)):
                    return False
            # 3. idx as right: i + idx = k
            for (i, k) in conflicts_as_right[idx]:
                if (mask & (1 << i)) and (mask & (1 << k)):
                    return False
            return True

        # Strategy 1: Greedy removal from full set
        def greedy_removal():
            active_mask = (1 << n) - 1
            # Iteratively remove element with highest conflict degree
            # This is expensive to recompute every time.
            # Alternative: Simple greedy removal of random conflicts
            for _ in range(n * 5):
                if time.time() > deadline - 5:
                    break
                # Find a conflict
                conflict_found = False
                for k in range(n):
                    if active_mask & (1 << k):
                        for (i, j) in conflicts_as_sum[k]:
                            if (active_mask & (1 << i)) and (active_mask & (1 << j)):
                                # Conflict (i, j, k)
                                # Remove one of them randomly or the one with most conflicts?
                                # Let's remove k for now (the sum)
                                active_mask &= ~(1 << k)
                                conflict_found = True
                                break
                        if conflict_found:
                            break
                if not conflict_found:
                    break
            return active_mask

        # Strategy 2: Greedy addition with various orderings
        def greedy_addition(order):
            active_mask = 0
            for idx in order:
                if not (active_mask & (1 << idx)):
                    if can_add(idx, active_mask):
                        active_mask |= (1 << idx)
            return active_mask

        # Strategy 3: Local search with swaps
        def local_search(initial_mask, max_iters=5000):
            active_mask = initial_mask
            current_conflicts = count_conflicts(active_mask)
            best_mask = active_mask
            best_size = bin(active_mask).count('1')
            rng = random.Random(12345)
            
            # Precompute list of indices for faster iteration
            all_indices = list(range(n))
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    break

                if current_conflicts == 0:
                    # Try to add more elements
                    improved = False
                    # Try all inactive elements
                    for idx in all_indices:
                        if not (active_mask & (1 << idx)):
                            if can_add(idx, active_mask):
                                active_mask |= (1 << idx)
                                current_conflicts = count_conflicts(active_mask)
                                if bin(active_mask).count('1') > best_size:
                                    best_size = bin(active_mask).count('1')
                                    best_mask = active_mask
                                improved = True
                                break
                    if not improved:
                        # Try 1-for-2 swap: remove one, add two
                        swap_improved = False
                        # Sample a few removal candidates
                        active_indices = [i for i in all_indices if active_mask & (1 << i)]
                        if len(active_indices) > 0:
                            for remove_c in rng.sample(active_indices, min(5, len(active_indices))):
                                new_mask = active_mask & ~(1 << remove_c)
                                # Try to add two inactive elements
                                inactive_indices = [i for i in all_indices if not (new_mask & (1 << i))]
                                if len(inactive_indices) < 2:
                                    continue
                                # Try random pairs
                                for _ in range(10):
                                    i1, i2 = rng.sample(inactive_indices, 2)
                                    test_mask = new_mask | (1 << i1) | (1 << i2)
                                    if count_conflicts(test_mask) == 0:
                                        active_mask = test_mask
                                        current_conflicts = 0
                                        if bin(active_mask).count('1') > best_size:
                                            best_size = bin(active_mask).count('1')
                                            best_mask = active_mask
                                        swap_improved = True
                                        break
                                if swap_improved:
                                    break
                        if not swap_improved:
                            break
                    continue

                # Try to fix conflicts
                # Find a conflict
                conflict_found = None
                for k in range(n):
                    if active_mask & (1 << k):
                        for (i, j) in conflicts_as_sum[k]:
                            if (active_mask & (1 << i)) and (active_mask & (1 << j)):
                                conflict_found = (i, j, k)
                                break
                        if conflict_found:
                            break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                candidates = [i, j, k]
                rng.shuffle(candidates)

                best_new_conflicts = current_conflicts
                best_remove = None
                for c in candidates:
                    new_mask = active_mask & ~(1 << c)
                    new_conflicts = count_conflicts(new_mask)
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_remove = c

                if best_remove is not None:
                    active_mask &= ~(1 << best_remove)
                    current_conflicts = best_new_conflicts
                    if bin(active_mask).count('1') > best_size and current_conflicts == 0:
                        best_size = bin(active_mask).count('1')
                        best_mask = active_mask
                else:
                    # Try swap: remove one from conflict, add a random inactive
                    swap_found = False
                    for c in candidates:
                        new_mask = active_mask & ~(1 << c)
                        inactive = [x for x in all_indices if not (new_mask & (1 << x))]
                        if not inactive:
                            continue
                        rng.shuffle(inactive)
                        for add_candidate in inactive[:10]:
                            test_mask = new_mask | (1 << add_candidate)
                            new_conflicts = count_conflicts(test_mask)
                            if new_conflicts < current_conflicts:
                                active_mask = test_mask
                                current_conflicts = new_conflicts
                                if bin(active_mask).count('1') > best_size and current_conflicts == 0:
                                    best_size = bin(active_mask).count('1')
                                    best_mask = active_mask
                                swap_found = True
                                break
                        if swap_found:
                            break

            return best_mask

        # Run strategies
        best_mask = 0
        best_size = 0

        # Strategy 1: Greedy removal
        mask1 = greedy_removal()
        size1 = bin(mask1).count('1')
        if size1 > best_size:
            best_size = size1
            best_mask = mask1

        # Strategy 2: Multiple greedy additions
        rng = random.Random(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            order = list(range(n))
            rng.shuffle(order)
            mask2 = greedy_addition(order)
            size2 = bin(mask2).count('1')
            if size2 > best_size:
                best_size = size2
                best_mask = mask2

        # Try structured orders
        for order_func in [
            lambda: list(range(n)),
            lambda: sorted(range(n), key=lambda i: elements[i]),
            lambda: sorted(range(n), key=lambda i: abs(elements[i])),
            lambda: sorted(range(n), key=lambda i: -elements[i]),
            lambda: sorted(range(n), key=lambda i: elements[i] % 2), # Odds first?
        ]:
            if time.time() > deadline - 10:
                break
            order = order_func()
            mask2 = greedy_addition(order)
            size2 = bin(mask2).count('1')
            if size2 > best_size:
                best_size = size2
                best_mask = mask2

        # Strategy 3: Local search from best
        if best_mask:
            max_iters = 5000 if n <= 120 else 3000
            improved_mask = local_search(best_mask, max_iters)
            size_imp = bin(improved_mask).count('1')
            if size_imp > best_size:
                best_size = size_imp
                best_mask = improved_mask

        # Convert mask to list of elements
        result = [elements[i] for i in range(n) if best_mask & (1 << i)]
        
        # Final verification and cleanup (just in case)
        result_set = set(result)
        # Check sum-free
        valid = True
        for a in result:
            for b in result:
                if a + b in result_set:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: simple odd numbers
            result = [x for x in elements if x % 2 != 0]
            # Ensure sum-free for odd numbers? 
            # Sum of two odds is even. So if the set contains only odds, 
            # a+b is even. If the set contains no evens, it's sum-free.
            # But what if the set contains evens? 
            # The fallback just takes odds.
            
        return result

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            # Fallback to simple odd-number heuristic
            elems = instance(size, seed)
            results.append([x for x in elems if x % 2 != 0])
            continue

        elems = instance(size, seed)
        result = find_sum_free(elems)
        results.append(result)

    return results
