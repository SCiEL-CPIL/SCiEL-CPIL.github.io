import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    N = 1000
    max_rep = 2
    MAX_SUM = 2 * N  # 2000

    # Pre-allocated arrays for fast count tracking
    # counts[s] = number of representations of sum s
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    in_set = np.zeros(N + 1, dtype=bool)

    def check_valid_add(x, S_list, counts):
        """Check if adding x keeps all sums <= max_rep."""
        if counts[2 * x] >= max_rep:
            return False
        for y in S_list:
            if counts[x + y] >= max_rep:
                return False
        return True

    def add_element(x, S_list, S_set, counts, in_set):
        """Add x to the set and update counts."""
        S_list.append(x)
        S_set.add(x)
        in_set[x] = True
        for y in S_list[:-1]:
            counts[x + y] += 1
        counts[2 * x] += 1

    def remove_element(x, S_list, S_set, counts, in_set):
        """Remove x from the set and update counts."""
        S_list.remove(x)
        S_set.discard(x)
        in_set[x] = False
        for y in S_list:
            counts[x + y] -= 1
        counts[2 * x] -= 1

    def greedy_build(order, counts, in_set):
        """Greedy: add elements from order if valid."""
        S_list = []
        S_set = set()
        for x in order:
            if in_set[x]:
                continue
            if check_valid_add(x, S_list, counts):
                add_element(x, S_list, S_set, counts, in_set)
        return S_list, S_set, counts, in_set

    def local_search(S_list, S_set, counts, in_set, time_budget):
        """Local search: try to add elements, then remove-one-add-two."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            
            # Phase 1: Try to add any candidate
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            if not candidates:
                break
            random.shuffle(candidates)
            for x in candidates[:50]:
                if check_valid_add(x, S_list, counts):
                    add_element(x, S_list, S_set, counts, in_set)
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Remove-one-add-two
            if len(S_list) < 2:
                break

            # Try a few random removals
            for _ in range(8):
                if time.time() >= time_budget:
                    break
                u_idx = random.randint(0, len(S_list) - 1)
                u = S_list[u_idx]

                # Save state for rollback
                old_counts = counts.copy()
                old_in_set = in_set.copy()
                old_S_list = list(S_list)
                old_S_set = set(S_set)

                # Temporarily remove u
                remove_element(u, S_list, S_set, counts, in_set)

                # Now try to add two elements
                added = []
                candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
                random.shuffle(candidates)

                # Try to add first element
                for v1 in candidates[:60]:
                    if check_valid_add(v1, S_list, counts):
                        add_element(v1, S_list, S_set, counts, in_set)
                        added.append(v1)
                        break

                # Try to add second element
                if len(added) == 1:
                    candidates2 = [x for x in range(1, N + 1) if not in_set[x] and x != u and x != added[0]]
                    random.shuffle(candidates2)
                    for v2 in candidates2[:60]:
                        if check_valid_add(v2, S_list, counts):
                            add_element(v2, S_list, S_set, counts, in_set)
                            added.append(v2)
                            break

                # If we added 2 elements, net gain is +1 (removed 1, added 2)
                if len(added) == 2:
                    improved = True
                    break
                else:
                    # Rollback
                    S_list[:] = old_S_list
                    S_set.clear()
                    S_set.update(old_S_set)
                    counts[:] = old_counts
                    in_set[:] = old_in_set
                    break  # Skip remaining removals this iteration

        return S_list, S_set, counts, in_set

    def remove_one_add_two(S_list, S_set, counts, in_set, time_budget):
        """Dedicated remove-one-add-two phase with more attempts."""
        improved = True
        attempts = 0
        while improved and time.time() < time_budget and attempts < 200:
            improved = False
            attempts += 1

            if len(S_list) < 2:
                break

            u_idx = random.randint(0, len(S_list) - 1)
            u = S_list[u_idx]

            # Save state for rollback
            old_counts = counts.copy()
            old_in_set = in_set.copy()
            old_S_list = list(S_list)
            old_S_set = set(S_set)

            # Temporarily remove u
            remove_element(u, S_list, S_set, counts, in_set)

            # Now try to add two elements
            added = []
            candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
            random.shuffle(candidates)

            # Try to add first element
            for v1 in candidates[:80]:
                if check_valid_add(v1, S_list, counts):
                    add_element(v1, S_list, S_set, counts, in_set)
                    added.append(v1)
                    break

            # Try to add second element
            if len(added) == 1:
                candidates2 = [x for x in range(1, N + 1) if not in_set[x] and x != u and x != added[0]]
                random.shuffle(candidates2)
                for v2 in candidates2[:80]:
                    if check_valid_add(v2, S_list, counts):
                        add_element(v2, S_list, S_set, counts, in_set)
                        added.append(v2)
                        break

            # If we added 2 elements, net gain is +1 (removed 1, added 2)
            if len(added) == 2:
                improved = True
            else:
                # Rollback
                S_list[:] = old_S_list
                S_set.clear()
                S_set.update(old_S_set)
                counts[:] = old_counts
                in_set[:] = old_in_set

        return S_list, S_set, counts, in_set

    def swap_improve(S_list, S_set, counts, in_set, time_budget):
        """Try swapping one element for another to enable further additions."""
        improved = False
        if len(S_list) < 1:
            return S_list, S_set, counts, in_set, improved

        # Try a few random swaps
        for _ in range(10):
            if time.time() >= time_budget:
                break

            u_idx = random.randint(0, len(S_list) - 1)
            u = S_list[u_idx]

            # Save state
            old_counts = counts.copy()
            old_in_set = in_set.copy()
            old_S_list = list(S_list)
            old_S_set = set(S_set)

            # Remove u
            remove_element(u, S_list, S_set, counts, in_set)

            # Try to add a different element
            candidates = [x for x in range(1, N + 1) if not in_set[x] and x != u]
            random.shuffle(candidates)

            for v in candidates[:30]:
                if check_valid_add(v, S_list, counts):
                    add_element(v, S_list, S_set, counts, in_set)
                    # Now try to add more elements
                    added_more = 0
                    for w in candidates[:30]:
                        if w == v or in_set[w]:
                            continue
                        if check_valid_add(w, S_list, counts):
                            add_element(w, S_list, S_set, counts, in_set)
                            added_more += 1
                    # Net change: removed 1, added 1 + added_more
                    if added_more >= 1:
                        improved = True
                        break
                    else:
                        # Rollback
                        S_list[:] = old_S_list
                        S_set.clear()
                        S_set.update(old_S_set)
                        counts[:] = old_counts
                        in_set[:] = old_in_set
                        break

        return S_list, S_set, counts, in_set, improved

    best_size = 0
    best_set = []

    num_restarts = 100  # Increased restarts

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Reset state
        counts[:] = 0
        in_set[:] = False

        if r == 0:
            order = list(range(1, N + 1))
        elif r == 1:
            order = list(range(N, 0, -1))
        elif r == 2:
            # Even then odd
            order = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        elif r == 3:
            # Odd then even
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        elif r == 4:
            # Multiples of 3, then others
            order = list(range(3, N + 1, 3)) + [x for x in range(1, N + 1) if x % 3 != 0]
        elif r == 5:
            # Primes first (approximation)
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in range(1, N + 1) if is_prime(x)]
            non_primes = [x for x in range(1, N + 1) if not is_prime(x)]
            order = primes + non_primes
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        S_list, S_set, counts, in_set = greedy_build(order, counts, in_set)

        # Local search
        S_list, S_set, counts, in_set = local_search(
            S_list, S_set, counts, in_set, deadline
        )

        # Additional remove-one-add-two phase
        S_list, S_set, counts, in_set = remove_one_add_two(
            S_list, S_set, counts, in_set, deadline
        )

        # Swap improvement
        S_list, S_set, counts, in_set, _ = swap_improve(
            S_list, S_set, counts, in_set, deadline
        )

        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)

    return best_set
